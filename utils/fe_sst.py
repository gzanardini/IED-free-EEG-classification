import os
import numpy as np
import pandas as pd
from stockwell import st
from scipy.stats import kurtosis, skew
#from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def apply_montage(eeg_data, montage_type):
    """
    Apply the specified montage to EEG data.

    """
    n_ch, n_samples = eeg_data.shape

    if montage_type == 'CAR':
        # Common Average Reference Montage
        mean_signal = np.mean(eeg_data, axis=0)
        return eeg_data - mean_signal

    elif montage_type == 'Cz':
        # Cz Montage 
        cz_reference = eeg_data[9, :]
        return np.delete(eeg_data - cz_reference, 9, axis=0)

    elif montage_type == 'BipolarDB':
        # Bipolar Montage
        eeg_data_bipolar = np.array([
            eeg_data[0] - eeg_data[4],  # FP1 - F7
            eeg_data[4] - eeg_data[5],  # F7 - T3
            eeg_data[5] - eeg_data[6],  # T3 - T5
            eeg_data[6] - eeg_data[7],  # T5 - O1
            eeg_data[0] - eeg_data[1],  # FP1 - F3
            eeg_data[1] - eeg_data[2],  # F3 - C3
            eeg_data[2] - eeg_data[3],  # C3 - P3
            eeg_data[3] - eeg_data[7],  # P3 - O1
            eeg_data[11] - eeg_data[15], # FP2 - F8
            eeg_data[15] - eeg_data[16], # F8 - T4
            eeg_data[16] - eeg_data[17], # T4 - T6
            eeg_data[17] - eeg_data[18], # T6 - O2
            eeg_data[8] - eeg_data[9],   # FZ - CZ
            eeg_data[9] - eeg_data[10],  # CZ - PZ
            eeg_data[11] - eeg_data[12], # FP2 - F4
            eeg_data[12] - eeg_data[13], # F4 - C4
            eeg_data[13] - eeg_data[14], # C4 - P4
            eeg_data[14] - eeg_data[18], # P4 - O2
        ])
        return eeg_data_bipolar

    elif montage_type == 'Laplacian':
        # Laplacian Montage
        eeg_data_laplacian = np.array([
            eeg_data[0] - np.mean(eeg_data[[4, 11, 1]], axis=0),    # FP1 - avg(F7, FP2, F3)
            eeg_data[1] - np.mean(eeg_data[[0, 4, 8, 2]], axis=0), # F3 - avg(FP1, F7, FZ, C3)
            eeg_data[2] - np.mean(eeg_data[[1, 5, 9, 3]], axis=0), # C3 - avg(F3, T3, CZ, P3)
            eeg_data[3] - np.mean(eeg_data[[2, 6, 10, 7]], axis=0), # P3 - avg(C3, T5, PZ, O1)
            eeg_data[4] - np.mean(eeg_data[[0, 1, 5]], axis=0),     # F7 - avg(FP1, F3, T3)
            eeg_data[5] - np.mean(eeg_data[[2, 4, 6]], axis=0),     # T3 - avg(C3, F7, T5)
            eeg_data[6] - np.mean(eeg_data[[5, 3, 7]], axis=0),     # T5 - avg(T3, P3, O1)
            eeg_data[7] - np.mean(eeg_data[[6, 3, 18]], axis=0),    # O1 - avg(T5, P3, O2)
            eeg_data[8] - np.mean(eeg_data[[1, 11, 2, 12, 9]], axis=0), # FZ - avg(F3, FP2, C3, F4, CZ)
            eeg_data[9] - np.mean(eeg_data[[2, 8, 10, 13]], axis=0), # CZ - avg(C3, FZ, PZ, C4)
            eeg_data[10] - np.mean(eeg_data[[18, 3, 14, 7, 9]], axis=0), # PZ - avg(O2, P3, P4, O1, CZ)
            eeg_data[11] - np.mean(eeg_data[[0, 15, 12]], axis=0),   # FP2 - avg(FP1, F8, F4)
            eeg_data[12] - np.mean(eeg_data[[11, 15, 8, 13]], axis=0), # F4 - avg(FP2, F8, FZ, C4)
            eeg_data[13] - np.mean(eeg_data[[12, 16, 9, 14]], axis=0), # C4 - avg(F4, T4, CZ, P4)
            eeg_data[14] - np.mean(eeg_data[[13, 17, 10, 18]], axis=0), # P4 - avg(C4, T6, PZ, O2)
            eeg_data[15] - np.mean(eeg_data[[11, 12, 16]], axis=0),   # F8 - avg(FP2, F4, T4)
            eeg_data[16] - np.mean(eeg_data[[17, 13, 15]], axis=0),   # T4 - avg(T6, C4, F8)
            eeg_data[17] - np.mean(eeg_data[[18, 14, 16]], axis=0),   # T6 - avg(O2, P4, T4)
            eeg_data[18] - np.mean(eeg_data[[14, 17, 7]], axis=0),    # O2 - avg(P4, T6, O1)
        ])
        return eeg_data_laplacian

    else:
        raise ValueError("Invalid montage type specified.")
    
def apply_edge_removal(timeseries):
    n = len(timeseries)
    ind = np.arange(n)
    r = np.polyfit(ind, timeseries, 2)
    fit = np.polyval(r, ind)
    timeseries = timeseries - fit
    
    sh_len = n // 10
    wn = np.hanning(sh_len)
    if sh_len == 0:
        sh_len = n
        wn = np.ones(sh_len)
    
    timeseries[:sh_len//2] *= wn[:sh_len//2]
    timeseries[-sh_len//2:] *= wn[-sh_len//2:]
    return timeseries

def run_sST_seg(data, Fs, epoch_width, MONTAGE):
    input_data = apply_montage(data, MONTAGE)
    n_channels = input_data.shape[0]
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, Fs / 2), (1, Fs / 2)]
    minfreq = 1
    maxfreq = int(Fs / 2)
    all_stats = []

    for ch in range(n_channels):
        x1 = input_data[ch, :]
        L = len(x1)
        n_timeseries_segments = np.floor(L / (Fs * epoch_width)).astype(int)

        sr_power_batches_all = []

        for batch in range(n_timeseries_segments):
            start_time = int(batch * epoch_width * Fs)
            stop_time = int(start_time + epoch_width * Fs)
            timeseries_segment = x1[start_time:stop_time]
            timeseries_segment = apply_edge_removal(timeseries_segment)
            timeseries_segment = st.hilbert(timeseries_segment)
            L_segment = len(timeseries_segment)
            st_out_org = st.st(timeseries_segment, lo=minfreq, hi=maxfreq, gamma=0.9, win_type='gauss')

            sr_power_batches = []

            for fmin, fmax in bands:
                fmin_index, fmax_index = int(fmin), int(fmax)
                st_out_band = st_out_org[fmin_index:fmax_index, :]
                
                n_segments = L_segment // (Fs // 2)
                power_sum = []
                for i in range(n_segments):
                    start = i * (Fs // 2)
                    stop = start + (Fs // 2)
                    segment_power_sum = np.sum(np.sum(np.abs(st_out_band[:, start:stop])))
                    power_sum.append(segment_power_sum)
                    
                power_skew = skew(power_sum, bias=False)

                sr_power_batches.extend([power_skew])

            sr_power_batches_all.append(sr_power_batches)

        if n_timeseries_segments == 0:
            stats = np.zeros((5, len(bands)))
        else:
            stats = np.vstack([
                np.mean(sr_power_batches_all, axis=0),
                np.median(sr_power_batches_all, axis=0),
                np.std(sr_power_batches_all, ddof=1, axis=0),
                skew(sr_power_batches_all, bias=False, axis=0),
                kurtosis(sr_power_batches_all, fisher=False, axis=0)
            ])
        all_stats.append(stats)

    if all_stats:
        all_stats = np.array(all_stats)
        all_stats = all_stats.transpose(1, 0, 2).reshape(5, -1)
    else:
        all_stats = np.zeros((5, len(bands) * n_channels))

    return all_stats

def read_lookup_table(file_path):
    lut = pd.read_csv(file_path, header=None)
    file_ids = lut.iloc[:, 0].tolist()
    return file_ids

def process_file(file_id, patient_type, output_dir, montages, segment_lengths, combiner_names, Fs=200):
    if patient_type == 'h':
        data_dir = '/home/yash@mydre.org/Documents/data/ML_EEG_TUD/data_v1/pp_npy/h_patients/'
    else:
        data_dir = '/home/yash@mydre.org/Documents/data/ML_EEG_TUD/data_v1/pp_npy/pat_patients/'

    file_path = os.path.join(data_dir, f'EMC_{patient_type}_PREP4_200hz_{file_id}.npy')
    if not os.path.exists(file_path):
        #print(f"File {file_path} not found, skipping.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    data = np.load(file_path)

    feature_type = 'sST'

    for montage in montages:
        for sec in segment_lengths:
            all_files_exist = True
            output_filenames = []
            for combiner_name in combiner_names:
                output_filename = f'{file_id}_{montage}_{feature_type}{combiner_name}_{sec}_{patient_type}.npy'
                full_output_filename = os.path.join(output_dir, output_filename)
                output_filenames.append(full_output_filename)
                if not os.path.exists(full_output_filename):
                    all_files_exist = False

            if all_files_exist:
                # print(f"Skipping {file_id} for {montage} at {sec} seconds, already processed.")
                continue

            features = run_sST_seg(data, Fs, sec, montage)
            if features is None:
                continue

            for idx, combiner_feature in enumerate(features):
                np.save(output_filenames[idx], combiner_feature)

    # print('File processed:', file_path)

def process_file_wrapper(args):
    process_file(*args)

def main():
    output_dir = '/home/yash@mydre.org/Documents/data/ML_EEG_TUD/data_v1/features/sST_npy'
    montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
    segment_lengths = [5*60, 2*60, 1*60, 20, 10, 5, 2]
    combiner_names = ['mean', 'median', 'std', 'skewness', 'kurtosis']
    Fs = 200

    lut_h = read_lookup_table('/home/yash@mydre.org/Documents/data/ML_EEG_TUD/data_v1/LUT_h.csv')
    lut_pat = read_lookup_table('/home/yash@mydre.org/Documents/data/ML_EEG_TUD/data_v1/LUT_pat.csv')
    # Combine file IDs and patient types into a list of tuples
    files_to_process = [(file_id, 'h', output_dir, montages, segment_lengths, combiner_names, Fs) for file_id in lut_h] + \
                       [(file_id, 'pat', output_dir, montages, segment_lengths, combiner_names, Fs) for file_id in lut_pat]

    pool_size = os.cpu_count()
    process_map(process_file_wrapper, files_to_process, max_workers=pool_size, chunksize=1)
    print('All files processed.')

if __name__ == "__main__":
    main()