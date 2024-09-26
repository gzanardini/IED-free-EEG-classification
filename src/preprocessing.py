import os
import mne
from src import utils
from tqdm import tqdm
import numpy as np
import json


def filter_and_downsample(dataset, data):
    # Notch filter (to eliminate power line interference)
    data = utils.notch_filter(data, dataset.original_sampling_freq, cutoff_freq=dataset.notch_filter_cutoff)
    
    # High pass filter (to remove DC oﬀset and baseline ﬂuctuations)
    data = utils.highpass_filter(data, dataset.original_sampling_freq, cutoff_freq=dataset.highpass_filter_cutoff)
    
    # Downsampling (to 200Hz)
    data = utils.downsample(data, dataset.original_sampling_freq, dataset.sampling_freq)
    
    # Apply Montage (Common Average Reference)
    average_potential = np.mean(data, axis=0)
    data = data - average_potential
    
    # Noise statistics-based artifact rejection (to remove high amplitude noise)
    # TODO ?
        
    return data
    

def preprocessing(dataset, save_spectrograms=False):
    if not utils.folder_exists(dataset.preprocessed_folder): os.makedirs(dataset.preprocessed_folder, exist_ok=False)
    if save_spectrograms and not utils.folder_exists(dataset.spectrograms_folder): os.makedirs(dataset.spectrograms_folder, exist_ok=False)
    
    eeg_files = [ file for file in os.listdir(dataset.data_dir) if file.endswith('.edf') ]
    eeg_files.sort()
    
    
    for idx, file in tqdm(enumerate(eeg_files), total=len(eeg_files), desc='Preprocessing and segmenting EEG data'):
        preprocessed_eeg_folder_path = dataset.preprocessed_folder + '/' + file.split('.')[0] + '/'
        if not utils.folder_exists(preprocessed_eeg_folder_path): os.mkdir(preprocessed_eeg_folder_path)
        
        if save_spectrograms:
            eeg_spectrograms_folder_path = dataset.spectrograms_folder + '/' + file.split('.')[0] + '/'
            if not utils.folder_exists(eeg_spectrograms_folder_path): os.mkdir(eeg_spectrograms_folder_path)
        
        eeg_file_path = os.path.join(dataset.data_dir, file)
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
        
        eeg_raw_data = filter_and_downsample(dataset, eeg_raw_data)
        
        # splitting the data into equal-sized segments 
        window_len_timesteps = int(dataset.sampling_freq * (dataset.window_length))
        step = int(dataset.sampling_freq * (dataset.window_length*dataset.overlap_factor))
        for i in range(step, eeg_raw_data.shape[1], step): # skip the first segment
            segment = eeg_raw_data[:, i:i+window_len_timesteps]
            if not segment.shape[1] < window_len_timesteps: 
                np.save(preprocessed_eeg_folder_path + f'{i}.npy', segment)
                if save_spectrograms:
                    Sx, _ = utils.stft(segment, dataset.sampling_freq)
                    np.save(eeg_spectrograms_folder_path + f'{i}.npy', abs(Sx)**2)