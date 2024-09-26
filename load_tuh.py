import os 
import pickle
import numpy as np
import mne
from multiprocessing import Pool

#get cpu count
ncpus = os.cpu_count()

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

standard_channels = [ "EEG FP1-REF", "EEG F7-REF",   "EEG T3-REF", "EEG T5-REF",    "EEG O1-REF",    "EEG FP2-REF",    "EEG F8-REF",    "EEG T4-REF",    "EEG T6-REF",    "EEG O2-REF",  
                  "EEG FP1-REF",    "EEG F3-REF",    "EEG C3-REF",    "EEG P3-REF",    "EEG O1-REF",    "EEG FP2-REF",    "EEG F4-REF",    "EEG C4-REF",    "EEG P4-REF",    "EEG O2-REF",]


def split_and_dump(params):
    ''' Function to split and dump the data
    Args:
    params: tuple, (fetch_folder, sub, dump_folder, label)
    '''
    fetch_folder, sub, dump_folder, label = params

    if not os.path.exists(fetch_folder):
        raise FileNotFoundError(f"{fetch_folder} does not exist")
    
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)
    
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            try:
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in raw.ch_names:
                            useless_chs.append(ch)
                    raw.drop_channels(useless_chs)
                if chOrder_standard is not None and len(chOrder_standard) == len(raw.ch_names):
                    raw.reorder_channels(chOrder_standard)
                if raw.ch_names != chOrder_standard:
                    raise Exception("channel order is wrong!")

                raw.filter(l_freq=0.1, h_freq=75.0)
                raw.notch_filter(50.0)
                raw.resample(200, n_jobs=5)

                ch_name = raw.ch_names
                raw_data = raw.get_data(units='uV')
                channeled_data = raw_data.copy()
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue
            for i in range(channeled_data.shape[1] // 2000):
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".pkl"
                )
                pickle.dump(
                    {"X": channeled_data[:, i * 2000 : (i + 1) * 2000], "y": label},
                    open(dump_path, "wb"),
                )

def main():
    split_and_dump(('data/edf/train', '01_tcp_ar', 'data/pkl/train', 0))