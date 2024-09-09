import os
import torch
from torch.utils.data import Dataset
import mne
import scipy
from src import utils
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



class EEGSegment():
    def __init__(self, segment_filename, edf_filename, eeg_idx, timesteps):
        self.filename = segment_filename
        self.edf_filename = edf_filename
        self.eeg_idx = eeg_idx
        self.timesteps = timesteps
    
    def get_data(self):
        return np.load(self.filename)
    
    def get_spectrograms(self, stft_obj):
        # TODO load a saved spectrogram if it exists
        data = self.get_data()
        return abs(stft_obj.stft(data))**2
        # if self.spectrograms is None:
        #     self.spectrograms = abs(stft_obj.stft(self.data))**2
        # return self.spectrograms
            


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
        
        # STFT parameters
        self.stft_window = stft_window
        self.stft_nperseg = stft_nperseg
        self.stft_noverlap = stft_noverlap if stft_noverlap is not None else int(stft_nperseg - 7)
        self.stft_obj = scipy.signal.ShortTimeFFT.from_window(self.stft_window, self.sampling_freq, self.stft_nperseg, self.stft_noverlap, scale_to='magnitude')
        
        # text data
        self.labels = []                        # one entry per edf file
        # lookup tables
        self.eeg_idx2personal_id = []           # one entry per edf file
        self.personal_id2eeg_idx = {}           # one entry per person
        # EEG data segments
        self.eeg_segments = []                  # one entry for each chunk of EEG data
        self.eeg_idx2edf_filename = []          # one entry per edf file
        self.eeg_idx2edf_info = []              # one entry per edf file # TODO fix this, json loading not working
        self.personal_id2eeg_segments_idx = {}  # one entry per person, list of indices of eeg_segments
        
        
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
            
            # TODO fix this, json loading not working
            # info_filename = self.preprocessed_folder + eeg_original_filename + '/info.txt'
            # with open(info_filename, 'r') as f:
            #     info = json.load(f)
            #     self.idx2edf_info.append(info)
            
            eeg_segment_files = [ file for file in os.listdir(self.preprocessed_folder + edf_filename) if file.endswith('.npy') ]
            eeg_segment_files.sort()
            start = 0
            for segment_file in eeg_segment_files:
                eeg_file_path = self.preprocessed_folder + edf_filename + '/' + segment_file
                self.eeg_segments.append( EEGSegment(eeg_file_path, edf_filename, eeg_idx, (start, start+window_len_timesteps)) )
                personal_id = self.eeg_idx2personal_id[eeg_idx]
                if personal_id not in self.personal_id2eeg_segments_idx: self.personal_id2eeg_segments_idx[personal_id] = []
                self.personal_id2eeg_segments_idx[personal_id].append(len(self.eeg_segments)-1)
                start += step

                
    def load_data(self):
        self.load_labels() # Load text data
        self.load_lookup_table() # Load lookup table
        self.load_eeg_data() # Load EEG data
                              
    
    def get_nr_patients(self):
        return len(self.personal_id2eeg_segments_idx.keys())
    
    
    def get_patient_id(self, segment_idx):
        return self.eeg_idx2personal_id[self.eeg_segments[segment_idx].eeg_idx]
    
    
    def filter_and_downsample(self, data):
        # Notch filter (to eliminate power line interference)
        data = utils.notch_filter(data, self.original_sampling_freq, cutoff_freq=self.notch_filter_cutoff)
        
        # High pass filter (to remove DC oﬀset and baseline ﬂuctuations)
        data = utils.highpass_filter(data, self.original_sampling_freq, cutoff_freq=self.highpass_filter_cutoff)
        
        # Downsampling (to 200Hz)
        data = utils.downsample(data, self.original_sampling_freq, self.sampling_freq)
        
        # Apply Montage (Common Average Reference)
        average_potential = np.mean(data, axis=0)
        data = data - average_potential
        
        # Noise statistics-based artifact rejection (to remove high amplitude noise)
        # TODO ?
            
        return data
    
    
    def preprocessing(self):
        if not utils.folder_exists(self.preprocessed_folder): os.makedirs(self.preprocessed_folder, exist_ok=False)
        # if os.listdir(self.preprocessed_folder): return # don't do the preprocessing if the folder is not empty
        
        eeg_files = [ file for file in os.listdir(self.data_dir) if file.endswith('.edf') ]
        eeg_files.sort()
        
        
        for idx, file in tqdm(enumerate(eeg_files), total=len(eeg_files), desc='Preprocessing and segmenting EEG data'):
            preprocessed_eeg_folder_path = self.preprocessed_folder + '/' + file.split('.')[0] + '/'
            if not utils.folder_exists(preprocessed_eeg_folder_path): os.mkdir(preprocessed_eeg_folder_path)
            else: continue
            eeg_file_path = os.path.join(self.data_dir, file)
            eeg_edf = mne.io.read_raw_edf(eeg_file_path, preload=True, verbose='ERROR')
            eeg_raw_data = eeg_edf.get_data()
            
            # storing edf info
            info = {}
            # for info_key in ['ch_names', 'nchan', 'chs', 'meas_date', 'highpass', 'lowpass', 'subject_info']: 
            for info_key in ['ch_names']: 
                info[info_key] = eeg_edf.info[info_key]
            info_file_path = preprocessed_eeg_folder_path + 'info.txt'
            if not os.path.exists(info_file_path):
                with open(info_file_path, 'w') as f:
                    json.dump(info, f)
            
            eeg_raw_data = self.filter_and_downsample(eeg_raw_data)
            
            # splitting the data into equal-sized segments 
            window_len_timesteps = int(self.sampling_freq * (self.window_length))
            step = int(self.sampling_freq * (self.window_length*self.overlap_factor))
            for i in range(0, eeg_raw_data.shape[1], step):
                segment = eeg_raw_data[:, i:i+window_len_timesteps]
                if not segment.shape[1] < window_len_timesteps: 
                    np.save(preprocessed_eeg_folder_path + f'{i}.npy', segment)
                    
    
    def __len__(self):
        return len(self.eeg_segments)
    
    
    def __getitem__(self, idx):
        segment = self.eeg_segments[idx]    
        data = segment.get_data()[:self.n_channels, :] # TODO deal with different number of channels
        filename = self.eeg_idx2edf_filename[segment.eeg_idx]
        labels = self.labels[segment.eeg_idx].as_dict()
        # spectrograms = segment.get_spectrograms(self.stft_obj)
        # return {'data': data, 'segment_timesteps': segment.timesteps, 'filename': filename, 'spectrograms': spectrograms, 'labels': labels}
        return {'data': data, 'segment_timesteps': segment.timesteps, 'filename': filename, 'labels': labels}
    

    def get_patient_segment_indexes(self, patient_id):
        return self.personal_id2eeg_segments_idx[patient_id]
    
    
    def get_subsets(self, personal_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
        assert train_ratio + val_ratio + test_ratio <= 1.
        if seed is not None: np.random.seed(seed)
        segments_idx_per_patient = [self.personal_id2eeg_segments_idx[personal_id] for personal_id in personal_ids if personal_id in self.personal_id2eeg_segments_idx]
        
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