import os
import torch
from torch.utils.data import Dataset
import mne

class EEGDataset(Dataset):
    def __init__(self, data_dir_path, window_length=10):
        self.dir = data_dir_path
        self.window_len_seconds = window_length # seconds
        # self.data = [] # TODO remove
        self.infos = []
        self.filenames = []
        self.segments = []
        self.segments_timesteps = []
        self.segments_info_index = []
        
        # Load EEG segments
        info_index = 0
        with open(data_dir_path+'/RECORDS', 'r') as f:
            for line in f:
                eeg_file_path = data_dir_path + '/' + line.strip()
                if os.path.exists(eeg_file_path):
                    eeg_edf = mne.io.read_raw_edf(eeg_file_path, preload=True, verbose='ERROR')
                    
                    info = {}
                    info['sfreq'] = eeg_edf.info['sfreq']                 
                    self.infos.append(info)
                    # TODO save also other not None info
                    # self.infos.append(dict(eeg_edf.info))
                    
                    self.filenames.append(eeg_file_path.split('/')[-1])
                    raw_data = eeg_edf.get_data()
                    # self.data.append(eeg_edf) # TODO remove
                    
                    # TODO add info about seizures
                    
                    # splitting the data into equal-sized segments 
                    sampling_freq = eeg_edf.info['sfreq']
                    window_len_timesteps = int(sampling_freq * (self.window_len_seconds))
                    step = int(sampling_freq * (self.window_len_seconds/2))
                    for i in range(0, raw_data.shape[1], step):
                        segment = raw_data[:, i:i+window_len_timesteps]
                        if not segment.shape[1] < window_len_timesteps: 
                            self.segments.append(segment)
                            self.segments_timesteps.append((i, i+window_len_timesteps))
                            self.segments_info_index.append(info_index)
                    
                    info_index += 1
                    
        self.spectrograms = [None] * len(self.segments)
        
        # Load labels (seizure segments tags)
        self.labels = [0] * len(self.segments)
        self.load_labels() 

    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        segment_timesteps = self.segments_timesteps[idx]
        info_index = self.segments_info_index[idx]
        info = self.infos[info_index]
        filename = self.filenames[info_index]
        spectrograms = self.spectrograms[idx]
        label = self.labels[idx]
        return {'data': segment, 'segment_timesteps': segment_timesteps, 'info': info, 'filename': filename, 'spectrograms': spectrograms, 'labels': label}
        
    
    def preprocessing(self):
        pass # TODO move here the code from the notebook
    
    
    def load_labels(self):
        data_folders = [folder for folder in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, folder))]
        for folder in data_folders:
            summary_file = os.path.join(self.dir, folder, f'{folder}-summary.txt')
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    lines = f.readlines()
                    i = 0
                    while i < len(lines):
                        if lines[i].startswith('Number of Seizures in File:'):
                            nr_seizures = int(lines[i].split(':')[1].strip())
                            if nr_seizures > 0:
                                filename = lines[i-3].split(':')[1].strip()
                                if filename in self.filenames:
                                    file_index = self.filenames.index(filename)
                                    start = int(lines[i+1].split(':')[1].strip().split(' ')[0]) * 60
                                    end = int(lines[i+2].split(':')[1].strip().split(' ')[0]) * 60
                                    
                                    for seg_idx in range(len(self.segments)):
                                        if self.segments_info_index[seg_idx] == file_index:
                                            seg_start, seg_end = self.segments_timesteps[seg_idx]
                                            if (seg_start < start and start < seg_end) or (seg_start < end and end < seg_end) or (seg_start < start and seg_end > end) or (seg_start > start and seg_end < end):
                                                # print(seg_start, seg_end, start, end)
                                                self.labels[seg_idx] = 1
                                    # print(f'File: {filename}, Seizures: {nr_seizures}', start, end)
                        i+=1 # TODO optimize iteration (?)