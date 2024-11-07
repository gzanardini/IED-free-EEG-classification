import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from braindecode.preprocessing import create_fixed_length_windows
from TUEP import TUHEpilepsy

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

EPILEPSY_PATH='/space/gzanardini/tuh_eeg/tuh_eeg_epilepsy'

tuep=TUHEpilepsy(path=EPILEPSY_PATH,set_montage=True,rename_channels=True,target_name='epilepsy', n_jobs=4, preload=True)

import mne
from copy import deepcopy

def preprocess(in_dataset, target_freq=None, photic_ph=False):
    dataset=deepcopy(in_dataset)
    if photic_ph:
        channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
            'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'PHOTIC PH']
    else:
        channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
            'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        
    for i, sample in enumerate(dataset.datasets):
        #discard samples shorter than 10s
        if sample.raw.n_times/sample.raw.info['sfreq'] < 10:
            print(f"Discarding {i} because it is shorter than 10s")
            dataset.datasets.pop(i)
            continue
        #sample.raw.ch_names.upper()
        if set(channels[:-1]).issubset(ch_name.upper() for ch_name in sample.raw.ch_names) == False:
            print(f"Channels missing in {i}")
            print(set(channels) - set(sample.raw.ch_names))
            dataset.datasets.pop(i)
            continue

        sample.raw.rename_channels(lambda x: x.upper(),verbose=False)
        #check for PHOTIC PH channel
        if 'PHOTIC PH' in sample.raw.ch_names and photic_ph == True:
            #set the channel type to stim
            sample.raw.set_channel_types({'PHOTIC PH': 'stim'}, on_unit_change="ignore")
        elif 'PHOTIC PH' not in sample.raw.ch_names and photic_ph == True: 
            sample.raw.load_data()
            sample.raw.add_channels([mne.io.RawArray(np.zeros((1, sample.raw.n_times)), mne.create_info(['PHOTIC PH'], sample.raw.info['sfreq'], ['stim']), verbose=False)],force_update_info=True)
            print(f"Added PHOTIC PH channel to {i}")

        sample.raw.pick(channels)
        sample.raw.reorder_channels(channels)

        if sample.raw.info['sfreq'] != target_freq and target_freq is not None:
            sample.raw.resample(target_freq)
            print(f"Resampled {i} to {target_freq} Hz")

        sample.raw.notch_filter(60,verbose=False, n_jobs='cuda')

    return dataset

tuep_test=preprocess(tuep, target_freq=250, photic_ph=True)

def select_by_duration(dataset,tmin=10):
    new_dataset=deepcopy(dataset)
    new_dataset.datasets=[sample for sample in dataset.datasets if sample.raw.n_times/sample.raw.info['sfreq'] >= tmin]
    return new_dataset

channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
            'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'PHOTIC PH']

def select_by_channel(dataset,channels):
    new_dataset=deepcopy(dataset)
    new_dataset.datasets=[sample for sample in dataset.datasets if set(channels).issubset(ch_name.upper() for ch_name in sample.raw.ch_names)]
    return new_dataset

tuep_test=select_by_channel(tuep_test,channels)

tuep_test=select_by_duration(tuep_test, tmin=10)

savedir='/space/gzanardini/tuh_eeg/preprocessed/full/'
tuep_test.save(savedir, overwrite=True)

windows_tuep=create_fixed_length_windows(tuep_test, window_size_samples=2500, window_stride_samples=2500, drop_last_window=True)
windows_tuep.save('/space/gzanardini/tuh_eeg/preprocessed/windows10s/', overwrite=True)
