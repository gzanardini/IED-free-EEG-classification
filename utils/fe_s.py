import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
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

def run_spectral_analysis(data, Fs, MONTAGE):
    # Display input data size
    #print(f'Initial data size: {data.shape}')
    # Filter design
    Fd = 4  # Delta
    bl, al = butter(4, Fd / (Fs / 2), 'low')
    #print('Delta band coefficients:')
    #print(bl)
    #print(al)
    Ft = [4, 7.9]  # Theta
    bt, at = butter(4, [f / (Fs / 2) for f in Ft], 'band')
    # print('Theta band coefficients:')
    # print(bt)
    # print(at)

    Fa = [8, 13]  # Alpha
    ba, aa = butter(4, [f / (Fs / 2) for f in Fa], 'band')
    # print('Alpha band coefficients:')
    # print(ba)
    # print(aa)
    Fb = [13, 30]  # Beta
    bb, ab = butter(4, [f / (Fs / 2) for f in Fb], 'band')
    # print('Beta band coefficients:')
    # print(bb)
    # print(ab)
    Fg3 = [31, (Fs / 2) - 0.1]  # Gamma
    bg3, ag3 = butter(4, [f / (Fs / 2) for f in Fg3], 'band')
    # print('Gamma band coefficients:')
    # print(bg3)
    # print(ag3)

    # Apply montage to data
    data_montage = apply_montage(data, MONTAGE)

    # Display montage data size
    #print(f'Data size after montage: {data_montage.shape}')

    # Perform spectral analysis
    pow = np.zeros((data_montage.shape[0], 5))  # Initialize power matrix
    #pow = np.zeros((data_montage.shape[0], 5))  # Assuming 5 frequency bands
    for j in range(data_montage.shape[0]):
        a = data_montage[j, :]
        delta = lfilter(bl, al, a)
        p_delta = np.dot(delta, delta)

        g3 = lfilter(bg3, ag3, a)
        p_g3 = np.dot(g3, g3)

        theta = lfilter(bt, at, a)
        p_theta = np.dot(theta, theta)

        alpha = lfilter(ba, aa, a)
        p_alpha = np.dot(alpha, alpha)

        beta = lfilter(bb, ab, a)
        p_beta = np.dot(beta, beta)

        # Compute total power and relative power
        p_total = p_delta + p_g3 + p_theta + p_alpha + p_beta  # only up to 100Hz
        # p_delta = p_delta / p_total
        # p_beta = p_beta / p_total
        # p_alpha = p_alpha / p_total
        # p_theta = p_theta / p_total
        # p_g3 = p_g3 / p_total
        if p_total > 0:
            p_delta = p_delta / p_total
            p_beta = p_beta / p_total
            p_alpha = p_alpha / p_total
            p_theta = p_theta / p_total
            p_g3 = p_g3 / p_total
        else:
            # Handle the case where p_total is 0 or very close to 0
            # Here, you might set all relative powers to zero or to NaN
            # This choice depends on how you wish to handle these cases
            p_delta = 0
            p_beta = 0
            p_alpha = 0
            p_theta = 0
            p_g3 = 0
        # Store the result for each channel
        pow[j, 0] = p_delta
        pow[j, 1] = p_theta
        pow[j, 2] = p_alpha
        pow[j, 3] = p_beta
        pow[j, 4] = p_g3
        # Uncomment the following lines if g2 and g1 results are needed
        # pow[j, 5] = p_g2
        # pow[j, 6] = p_g1

    # Reshape the pow array to match MATLAB's output
    pow = pow.T.flatten()

    # Display final power vector size
    #print(f'Final power vector size: 1 x {pow.size}')

    return pow

def run_spectral_seg2(data, Fs, MONTAGE, sec):
    data_whole = data.copy()
    Fd = 4  # Delta
    bl, al = butter(4, Fd / (Fs / 2), 'low')
    Ft = [4, 7.9]  # Theta
    bt, at = butter(4, [f / (Fs / 2) for f in Ft], 'band')
    Fa = [8, 13]  # Alpha
    ba, aa = butter(4, [f / (Fs / 2) for f in Fa], 'band')
    Fb = [13, 30]  # Beta
    bb, ab = butter(4, [f / (Fs / 2) for f in Fb], 'band')
    Fg3 = [31, (Fs / 2) - 0.1]  # Gamma
    bg3, ag3 = butter(4, [f / (Fs / 2) for f in Fg3], 'band')
    data = apply_montage(data, MONTAGE)
    # Calculate number of segments
    seg_num = int(data.shape[1] / (Fs * sec))
    n_ch = data.shape[0]

    if seg_num == 0:
        output = run_spectral_analysis(data_whole, Fs, MONTAGE)
        combiners = ['mean', 'median', 'std', 'skewness', 'kurtosis']
        output_combiner = np.array([output for _ in combiners])
    else:
        # Process each segment
        data = data[:, :seg_num*sec*Fs]  # Clip the extra EEG segment on the right
        output_temp2 = []

        for tt in range(seg_num):
            segment = data[:, tt*sec*Fs:(tt+1)*sec*Fs]
            pow = np.zeros((n_ch, 5))
            
            for j in range(n_ch):
                a = segment[j, :]
                delta = lfilter(bl, al, a)
                p_delta = np.dot(delta, delta)

                g3 = lfilter(bg3, ag3, a)
                p_g3 = np.dot(g3, g3)

                theta = lfilter(bt, at, a)
                p_theta = np.dot(theta, theta)

                alpha = lfilter(ba, aa, a)
                p_alpha = np.dot(alpha, alpha)

                beta = lfilter(bb, ab, a)
                p_beta = np.dot(beta, beta)
                p_total = p_delta + p_g3 + p_theta + p_alpha + p_beta  # only up to 100Hz

                if p_total > 0:
                    p_delta = p_delta / p_total
                    p_beta = p_beta / p_total
                    p_alpha = p_alpha / p_total
                    p_theta = p_theta / p_total
                    p_g3 = p_g3 / p_total
                else:
                    p_delta = 0
                    p_beta = 0
                    p_alpha = 0
                    p_theta = 0
                    p_g3 = 0

                pow[j, 0] = p_delta
                pow[j, 1] = p_theta
                pow[j, 2] = p_alpha
                pow[j, 3] = p_beta
                pow[j, 4] = p_g3

            pow = pow.T.flatten()

            output_temp2.append(pow)

        output_temp2 = np.array(output_temp2)
        output_combiner = np.vstack([
            np.mean(output_temp2, axis=0),
            np.median(output_temp2, axis=0),
            np.std(output_temp2, ddof=1, axis=0),
            skew(output_temp2, bias=False, axis=0),
            kurtosis(output_temp2, fisher=False, axis=0)
            ])

        return output_combiner

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
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    data = np.load(file_path)

    feature_type = 'S'

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

            features = run_spectral_seg2(data, Fs, montage, sec)
            if features is None:
                continue

            for idx, combiner_feature in enumerate(features):
                np.save(output_filenames[idx], combiner_feature)

    # print('File processed:', file_path)

def process_file_wrapper(args):
    process_file(*args)

def main():
    output_dir = '/home/yash@mydre.org/Documents/data/ML_EEG_TUD/data_v1/features/S_npy'
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