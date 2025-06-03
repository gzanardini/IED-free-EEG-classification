import os
import torch
from torch.utils.data import Dataset
import utils
from tqdm import tqdm
import numpy as np
import json


class Labels():
    def __init__(self, labels):
        self.id = labels[0]
        self.eeg_label = labels[1]
        self.presence_IEDs = labels[2]
        self.presence_ictal = labels[3]
        self.presence_slowing = labels[4]
        self.patience_state = labels[5]
        self.medication_history = labels[6]
        self.presence_artifacts = labels[7]
        self.epilepsy_history = labels[8]
        self.age = labels[9]
        self.gender = labels[10]
    
    def as_dict(self):
        return {
            'id': self.id,
            'eeg_label': self.eeg_label,
            'presence_IEDs': self.presence_IEDs,
            'presence_ictal': self.presence_ictal,
            'presence_slowing': self.presence_slowing,
            'patience_state': self.patience_state,
            'medication_history': self.medication_history,
            'presence_artifacts': self.presence_artifacts,
            'epilepsy_history': self.epilepsy_history,
            'age': self.age
        }
    
    def __str__(self):
        return str(self.as_dict())

################################################################################################

class EEGSegment():
    def __init__(self, segment_filename, spectrograms_filename, edf_filename, eeg_idx, timesteps):
        self.edf_filename = edf_filename
        self.eeg_filename = segment_filename
        self.spectrograms_filename = spectrograms_filename
        self.eeg_idx = eeg_idx
        self.timesteps = timesteps
    
    def get_data(self):
        return np.load(self.eeg_filename)
    
    def get_spectrograms(self):
        if utils.file_exists(self.spectrograms_filename):
            return np.load(self.spectrograms_filename)
        return self.compute_spectrograms(200) # TODO remove when we are sure all spectrograms are computed

    def compute_spectrograms(self, sampling_freq):
        data = self.get_data()
        Sx, _ = utils.stft(data, sampling_freq)
        return abs(Sx)**2

################################################################################################

class EEGDataset(Dataset):
    def __init__(self, data_dir_path, preprocessed_dir, labels_path, lookup_table_path, original_sampling_freq=250, window_length=10, overlap_factor=0.5, stft_window='hann', stft_nperseg=128, stft_noverlap=None):
        # Dataset paths
        self.data_dir = data_dir_path
        self.labels_path = labels_path
        self.lookup_table_path = lookup_table_path
        self.preprocessed_folder = preprocessed_dir + '/preprocessed/' + self.data_dir.split('/')[-2] + '/'
        self.spectrograms_folder = preprocessed_dir + '/spectrograms/' + self.data_dir.split('/')[-2] + '/'
        
        # Preprocessing parameters
        self.window_length = window_length # seconds
        self.overlap_factor = overlap_factor
        self.n_channels = 16 # TODO deal with different number of channels
        self.original_sampling_freq = original_sampling_freq # Hz
        self.sampling_freq = 200
        self.notch_filter_cutoff = 60
        self.highpass_filter_cutoff = 1
        
        # text data
        self.labels = []                        # one entry per edf file
        # lookup tables
        self.eeg_idx2personal_id = []           # one entry per edf file
        self.personal_id2eeg_idx = {}           # one entry per person
        # EEG data segments
        self.eeg_segments = []                  # one entry for each chunk of EEG data
        self.eeg_idx2edf_filename = []          # one entry per edf file
        self.edf_filename2eeg_idx = {}          # one entry per edf file
        self.eeg_idx2edf_info = []              # one entry per edf file
        
        # self.personal_id2eeg_segments_idx = {}  # one entry per person, list of indices of eeg_segments
        self.filename2eeg_segments_idx = {}     # one entry per filename, list of indices of eeg_segments
        self.personal_id2filenames = {}         # one entry per person, list of filenames
        
        self.channel_name2channel_idx = {}
        # self.channel_idx2channel_name = []
        
        
    def load_labels(self):
        with open(self.labels_path, 'r') as f:
            for line in f:
                line = line[:-1].split(',')
                line[0] = int(line[0]) - 1
                labels = Labels( line )
                self.labels.append(labels)
    
    
    def load_lookup_table(self):
        with open(self.lookup_table_path, 'r') as f:
            for line in f:
                idx, pat_nr, _ = [ int(el) for el in line[:-1].split('\t') ] # what the third number refers to?
                idx -= 1
                self.eeg_idx2personal_id.append(pat_nr)
                if pat_nr not in self.personal_id2eeg_idx: self.personal_id2eeg_idx[pat_nr] = []
                self.personal_id2eeg_idx[pat_nr].append(idx)
                
                
    def load_eeg_data(self): # TODO move the splitting phase to preprocessing function
        eeg_folders = [ file for file in os.listdir(self.preprocessed_folder) if not file.endswith('.txt') ]
        eeg_folders.sort()
        window_len_timesteps = int(self.sampling_freq * (self.window_length))
        step = int(self.sampling_freq * (self.window_length*self.overlap_factor))
        
        for eeg_idx, edf_filename in tqdm(enumerate(eeg_folders), total=len(eeg_folders), desc='Loading EEG data'):
            self.eeg_idx2edf_filename.append(edf_filename)
            self.edf_filename2eeg_idx[edf_filename] = eeg_idx
            
            info_filename = self.preprocessed_folder + edf_filename + '/info.txt'
            with open(info_filename, 'r') as f:
                info = json.load(f)
                self.eeg_idx2edf_info.append(info)
                channels = [ch for ch in info['ch_names']]
                for ch in channels:
                    if 'EEG' not in ch: continue
                    ch = ch.split(' ')[-1].split('-')[0]
                    if ch not in self.channel_name2channel_idx:
                        self.channel_name2channel_idx[ch] = len(self.channel_name2channel_idx)
                        # self.channel_idx2channel_name.append(ch)
                
            
            eeg_segment_files = [ file for file in os.listdir(self.preprocessed_folder + edf_filename) if file.endswith('.npy') ]
            eeg_segment_files.sort()
            start = 0
            for segment_file in eeg_segment_files:
                eeg_file_path = self.preprocessed_folder + edf_filename + '/' + segment_file
                spectrograms_file_path = self.spectrograms_folder + edf_filename + '/' + segment_file
                self.eeg_segments.append( EEGSegment(eeg_file_path, spectrograms_file_path, edf_filename, eeg_idx, (start, start+window_len_timesteps)) )
                personal_id = self.eeg_idx2personal_id[eeg_idx]
                
                # if personal_id not in self.personal_id2eeg_segments_idx: self.personal_id2eeg_segments_idx[personal_id] = []
                # self.personal_id2eeg_segments_idx[personal_id].append(len(self.eeg_segments)-1)
                
                if edf_filename not in self.filename2eeg_segments_idx: self.filename2eeg_segments_idx[edf_filename] = []
                self.filename2eeg_segments_idx[edf_filename].append(len(self.eeg_segments)-1)
                
                if personal_id not in self.personal_id2filenames: self.personal_id2filenames[personal_id] = []
                if edf_filename not in self.personal_id2filenames[personal_id]: self.personal_id2filenames[personal_id].append(edf_filename)
                
                start += step

                
    def load_data(self):
        self.load_labels() # Load text data
        self.load_lookup_table() # Load lookup table
        self.load_eeg_data() # Load EEG data

    
    def __len__(self):
        return len(self.eeg_segments)
    
    
    def __getitem__(self, idx):
        segment = self.eeg_segments[idx]    
        
        data = segment.get_data()
        x = torch.zeros(len(self.channel_name2channel_idx), data.shape[1])
        
        spectrograms = None
        try:
            spectrograms = segment.get_spectrograms()
        except:    
            spectrograms = segment.compute_spectrograms(self.sampling_freq)
        spectrograms_padded = torch.zeros(len(self.channel_name2channel_idx), spectrograms.shape[1], spectrograms.shape[2])
        
        channels = self.eeg_idx2edf_info[segment.eeg_idx]['ch_names']
        for ch in channels:
            if ch in self.channel_name2channel_idx:
                ch_idx = self.channel_name2channel_idx[ch]
                x[ch_idx, :] = torch.tensor(data[channels.index(ch), :])
                spectrograms_padded[ch_idx, :, :] = torch.tensor(spectrograms[channels.index(ch), :, :])
        
        filename = self.eeg_idx2edf_filename[segment.eeg_idx]
        labels = self.labels[segment.eeg_idx].as_dict()

        return {'data': x, 'segment_timesteps': segment.timesteps, 'filename': filename, 'labels': labels, 'spectrograms': spectrograms_padded}
    
    
    def set_channel_name2channel_idx_dictionary(self, channel_name2channel_idx):
        self.channel_name2channel_idx = channel_name2channel_idx
    
    
    def get_subsets(self, personal_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
        assert train_ratio + val_ratio + test_ratio <= 1.
        if seed is not None: np.random.seed(seed)
        segments_idx_per_patient = [self.get_patient_segment_indexes(personal_id) for personal_id in personal_ids if self.is_patient_in_dataset(personal_id)]
        # ref = [self.personal_id2eeg_segments_idx[personal_id] for personal_id in personal_ids if personal_id in self.personal_id2eeg_segments_idx]
        # assert ref == segments_idx_per_patient
        
        train_n_patients = int(train_ratio * len(personal_ids))
        val_n_patients = int(val_ratio * len(personal_ids))
        test_n_patients = int(test_ratio * len(personal_ids))
        
        train_segments_idx = [ idx for segments_idx in segments_idx_per_patient[:train_n_patients] for idx in segments_idx ]
        val_segments_idx = [ idx for segments_idx in segments_idx_per_patient[train_n_patients:train_n_patients+val_n_patients] for idx in segments_idx ]
        test_segments_idx = [ idx for segments_idx in segments_idx_per_patient[train_n_patients+val_n_patients:] for idx in segments_idx ]
        
        train_subset = torch.utils.data.Subset(self, train_segments_idx)
        val_subset = torch.utils.data.Subset(self, val_segments_idx)
        test_subset = torch.utils.data.Subset(self, test_segments_idx)
        
        return train_subset, val_subset, test_subset
    
    def is_patient_in_dataset(self, personal_id):
        return personal_id in self.personal_id2eeg_idx
    
    def get_nr_patients(self):
        return len(self.personal_id2eeg_idx.keys())
    
    def get_patient_ids(self):
        return list(self.personal_id2eeg_idx.keys())
    
    def get_patient_id(self, segment_idx):
        return self.eeg_idx2personal_id[self.eeg_segments[segment_idx].eeg_idx]
    
    def get_patient_eeg_recordings(self, patient_id):
        return self.personal_id2filenames[patient_id]
    
    def get_eeg_recording_segment_indexes(self, filename):
        return self.filename2eeg_segments_idx[filename]
    
    def get_filename_segments_data(self, filename):
        return [ self[idx]['data'] for idx in self.filename2eeg_segments_idx[filename] ]
    
    def get_patient_segment_indexes(self, patient_id):
        # ref = self.personal_id2eeg_segments_idx[patient_id]
        eeg_recordings = self.get_patient_eeg_recordings(patient_id)
        indexes = [ eeg_idx for filename in eeg_recordings for eeg_idx in self.get_eeg_recording_segment_indexes(filename) ]
        return indexes
    
    def get_labels(self, filename):
        return self.labels[self.edf_filename2eeg_idx[filename]].as_dict()
    
    def get_filenames(self):
        return self.filename2eeg_segments_idx.keys()