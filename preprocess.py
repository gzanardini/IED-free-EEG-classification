import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from braindecode.preprocessing import create_fixed_length_windows
from utils.TUEP import TUHEpilepsy
import mne
from copy import deepcopy

from braindecode.preprocessing.preprocess import Preprocessor, preprocess as preprocess_bc

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

EPILEPSY_PATH='/space/gzanardini/tuh_eeg/tuh_eeg_epilepsy'

tuep=TUHEpilepsy(path=EPILEPSY_PATH,set_montage=False,rename_channels=True,target_name='epilepsy', n_jobs=2, preload=True)


def select_by_duration(dataset,tmin=10):
    new_dataset=deepcopy(dataset)
    new_dataset.datasets=[sample for sample in dataset.datasets if sample.raw.n_times/sample.raw.info['sfreq'] >= tmin]
    return new_dataset

def select_by_channel(dataset,channels):
    new_dataset=deepcopy(dataset)
    new_dataset.datasets=[sample for sample in dataset.datasets if set(channels).issubset(ch_name.upper() for ch_name in sample.raw.ch_names)]
    return new_dataset

channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
            'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'PHOTIC PH']


def preprocess(in_dataset, target_freq=None, photic_ph=True):
    dataset=deepcopy(in_dataset)
    if photic_ph:
        channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
            'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'PHOTIC PH']
    else:
        channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
            'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        
    for i, sample in enumerate(dataset.datasets):

        #if sample.raw.n_times/sample.raw.info['sfreq'] < 10:        #discard samples shorter than 10s
        #    print(f"Discarding {i} because it is shorter than 10s")
        #    dataset.datasets.pop(i)
        #    continue
        
        #sample.raw.ch_names.upper()
        if set(channels[:-1]).issubset(ch_name.upper() for ch_name in sample.raw.ch_names) == False:
            print(f"Channels missing in {i}")
            print(set(channels) - set(sample.raw.ch_names))
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
        sample.raw.filter(0.1,(target_freq/2)-0.1,verbose=False, n_jobs=8, l_trans_bandwidth=0.1)
        sample.raw.notch_filter(60,verbose=False, n_jobs=8)

        if sample.raw.info['sfreq'] != target_freq and target_freq is not None:
            sample.raw.resample(target_freq, n_jobs=8)
            print(f"Resampled {i} to {target_freq} Hz")

    return dataset

tuep_test=preprocess(tuep, target_freq=250, photic_ph=True)

tuep_test=select_by_channel(tuep_test,channels)

tuep_test=select_by_duration(tuep_test, tmin=10)

savedir='/space/gzanardini/tuh_eeg/preprocessed/full/'
tuep_test.save(savedir, overwrite=True)