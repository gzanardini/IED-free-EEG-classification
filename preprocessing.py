import os 
import mne
import numpy as np 
from src import utils

from src.dataset import EEGDataset

data_folder= '/space/gzanardini/tuh'
h_data_folder = data_folder + '/Non-epileptic EDF/'
h_data_lookup_table = data_folder + '/info_h/TUH_Epilepsy_h_lookup_table.txt'
h_labels = data_folder + '/info_h/TUH_Epilepsy_h_text_label.csv'

pat_data_folder = data_folder + '/Epilepsy EDF/'
pat_data_lookup_table = data_folder + '/info_pat/TUH_Epilepsy_pat_lookup_table.txt'
pat_labels = data_folder + '/info_pat/TUH_Epilepsy_pat_text_label.csv'

# data_folder = 'data/chb-mit'

window_length = 10 # seconds
overlap_factor = 1 # no overlap for the moment

h_dataset = EEGDataset(h_data_folder, h_data_lookup_table, h_labels, window_length, overlap_factor)
pat_dataset = EEGDataset(pat_data_folder, pat_data_lookup_table, pat_labels, window_length, overlap_factor)

do_preprocessing = True
if do_preprocessing:
    h_dataset.preprocessing()
    pat_dataset.preprocessing()