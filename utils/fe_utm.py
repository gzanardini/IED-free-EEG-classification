import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import cmath
import warnings
#from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def safe_log(x, epsilon=1e-10):
    if x > 0:
        return np.log(x)
    elif x == 0:
        return float('-inf')
    else:
        return cmath.log(x)

def zero_crossings(signal):
    # Shift signal by one and compare with the original signal, looking for sign changes
    sign_changes = np.diff(np.sign(signal))
    # Zero-crossings are where sign_changes is non-zero
    crossings = np.where(sign_changes != 0)[0]
    return len(crossings)

def do_bandpass_filtering(x, Fs, HP_fc, LP_fc):
    # Design butterworth bandpass filter
    nyq = 0.5 * Fs
    low = HP_fc / nyq
    high = LP_fc / nyq
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, x)

def ma_filter(x, wlength_ma):
    # Moving average filter
    return np.convolve(x, np.ones(wlength_ma)/wlength_ma, mode='valid')

def general_nleo(x, l=1, p=2, q=0, s=3):
    # check parameters:
    if ((l + p) != (q + s) and any(np.sort((l, p)) != np.sort((q, s)))):
        warnings('Incorrect parameters for NLEO. May be zero!')

    N = len(x)
    x_nleo = np.zeros(N)

    iedges = abs(l) + abs(p) + abs(q) + abs(s)
    n = np.arange(iedges + 1, (N - iedges - 1))

    x_nleo[n] = x[n-l] * x[n-p] - x[n-q] * x[n-s]

    return(x_nleo)

def discrete_hilbert(x, DBplot=False):
    """Discrete Hilbert transform

    Parameters
    ----------
    x: ndarray
        input signal
    DBplot: bool, optional
        plot or not 

    Returns
    -------
    x_hilb : ndarray
        Hilbert transform of x

    """
    N = len(x)
    Nh = np.ceil(N / 2)
    k = np.arange(N)

    # build the Hilbert transform in the frequency domain:
    H = -1j * np.sign(Nh - k) * np.sign(k)
    x_hilb = np.fft.ifft(np.fft.fft(x) * H)
    x_hilb = np.real(x_hilb)

    if DBplot:
        plt.figure(10, clear=True)
        plt.plot(np.imag(H))

    return(x_hilb)

def gen_edo(x, DBplot=False):
    """Generate EDO Γ[x(n)] from simple formula in the time domain:

    Γ[x(n)] = y(n)² + H[y(n)]²

    where y(n) is the derivative of x(n) using the central-finite method and H[.] is the
    Hilbert transform.

    Parameters
    ----------
    x: ndarray
        input signal
    DBplot: bool, optional
        plot or not

    Returns
    -------
    x_edo : ndarray
        EDO of x

    """
    # 1. check if odd length and if so make even:
    N_start = len(x)
    if (N_start % 2) != 0:
        x = np.hstack((x, 0))

    N = len(x)
    nl = np.arange(1, N - 1)
    xx = np.zeros(N)

    # 2. calculate the Hilbert transform
    h = discrete_hilbert(x)

    # 3. implement with the central finite difference equation
    xx[nl] = ((x[nl+1] ** 2) + (x[nl-1] ** 2) +
              (h[nl+1] ** 2) + (h[nl-1] ** 2)) / 4 - ((x[nl+1] * x[nl-1] +
                                                       h[nl+1] * h[nl-1]) / 2)

    # trim and zero-pad and the ends:
    x_edo = np.pad(xx[2:(len(xx) - 2)], (2, 2),
                   'constant', constant_values=(0, 0))

    return(x_edo[0:N_start])

def cal_freqweighted_energy(x, Fs, method, wlength_ma=None, bandpass_filter_params=None):
    if bandpass_filter_params is not None:
        LP_fc, HP_fc = bandpass_filter_params
        x = do_bandpass_filtering(x, Fs, HP_fc, LP_fc)
    
    # Calculate frequency-weighted energy
    if method == 'teager':
        # Placeholder for teager method
        x_nleo = general_nleo(x, 0, 0, 1, -1)
    elif method == 'envelope_diff':
        x_nleo = gen_edo(x)
    elif method ==  'palmu':
        l=1
        p=2
        q=0
        s=3
        x_nleo=general_nleo(x,l,p,q,s)
        x_nleo=np.abs(x_nleo)
        
    elif method =='abs_teager':
        l=0
        p=0
        q=1
        s=-1
        x_nleo=general_nleo(x,l,p,q,s)    
        x_nleo=np.abs(x_nleo)
        
    elif method ==  'env_only':
        x_nleo=np.abs( np.sqare(x))
    else:
        print("Error")
        
    # Smooth with MA filter
    if wlength_ma is not None:
        x_nleo = ma_filter(x_nleo, wlength_ma)
        x_nleo = np.roll(x_nleo, -wlength_ma//2)
    
    return x_nleo

def wentropy(x,entType,addiontalParameter = None):

    if entType == 'shannon':

        x = np.power(x[ x != 0 ],2)

        return - np.sum(np.multiply(x,np.log(x)))

    elif entType == 'threshold':

        if addiontalParameter is None or isinstance(addiontalParameter,str):

            return None

        x = np.absolute(x)

        return np.sum((x > addiontalParameter))

    elif entType == 'norm':

        if addiontalParameter is None or isinstance(addiontalParameter,str) or addiontalParameter < 1:

            return None

        x = np.absolute(x)

        return np.sum(np.power(x,addiontalParameter))

    elif entType == 'sure':

        if addiontalParameter is None or isinstance(addiontalParameter,str):

            return None

        N = len(x)

        x2 = np.square(x)

        t2 = addiontalParameter**2

        xgt = np.sum((x2 > t2))

        xlt = N - xgt


        return N - (2*xlt) + (t2 *xgt) + np.sum(np.multiply(x2,(x2 <= t2)))

    elif entType == 'logenergy':

        x = np.square(x[x != 0])

        return np.sum(np.log(x))

    else:
        print("invalid entropy type")
        return None

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

def run_UTM_whole(data, Fs, MONTAGE):
    data_montage = apply_montage(data, MONTAGE)
    y = []  # To store features
    for j in range(data_montage.shape[0]):
        data_j = data_montage[j, :]
        signal_mean = np.mean(data_j)
        signal_median = np.median(data_j)
        signal_std = np.std(data_j)
        signal_skewness = skew(data_j)
        signal_kurtosis = kurtosis(data_j, fisher=False)
        signal_zcd = zero_crossings(data_j)
        # Implement cal_freqweighted_energy or find a Python alternative
        signal_nleo_env_diff = safe_log(np.sum(cal_freqweighted_energy(data_j, Fs, 'envelope_diff')))
        signal_nleo_teager = safe_log(np.sum(cal_freqweighted_energy(data_j, Fs, 'teager')))
        signal_energy_tdomain = safe_log(np.sum(signal.square(np.abs(data_j))))
        F = fft(data_j)
        pow = np.log(np.sum(F * np.conj(F)))
        signal_energy_fdomain = safe_log(np.sum(signal.square(np.abs(data_j))))
        # Implement or find Python equivalents for entropy functions
        signal_ShEn = wentropy(data_j, entType = 'shannon')
        Vpp = np.ptp(data_j)
        # Peak detection
        pks, locs = find_peaks(data_j, height=50, distance=round(Fs/50))
        signal_npks = len(pks)

        # Plot the original signal
        y_j = [signal_mean, signal_median, signal_std, signal_skewness, signal_kurtosis,
                signal_zcd, signal_nleo_env_diff, signal_nleo_teager,
                signal_energy_tdomain, signal_energy_fdomain,
                signal_ShEn, Vpp, signal_npks]
        y.extend(y_j)
    return y

def run_UTM_seg(data, Fs, MONTAGE, sec):
    data_whole = data.copy()

    data = apply_montage(data, MONTAGE)
    n_ch, n_samples = data.shape
    # Calculate number of segments
    seg_num = int(data.shape[1] / (Fs * sec))

    if seg_num == 0:
        output = run_UTM_whole(data_whole, Fs, MONTAGE)
        combiners = ['mean', 'median', 'std', 'skewness', 'kurtosis']
        output_combiner = np.array([output for _ in combiners])
    else:
        # Process each segment
        data = data[:, :seg_num*sec*Fs]  # Clip the extra EEG segment on the right
        output = []
        for tt in range(seg_num):
            segment = data[:, tt*sec*Fs:(tt+1)*sec*Fs]
            y = []
            for j in range(n_ch):
                data_j = segment[j, :]
                signal_mean = np.mean(data_j)
                signal_median = np.median(data_j)
                signal_std = np.std(data_j)
                signal_skewness = skew(data_j)
                signal_kurtosis = kurtosis(data_j, fisher=False)
                signal_zcd = zero_crossings(data_j)
                # Implement cal_freqweighted_energy or find a Python alternative
                signal_nleo_env_diff = safe_log(np.sum(cal_freqweighted_energy(data_j, Fs, 'envelope_diff')))
                signal_nleo_teager = safe_log(np.sum(cal_freqweighted_energy(data_j, Fs, 'teager')))
                signal_energy_tdomain = safe_log(np.sum(signal.square(np.abs(data_j))))
                F = fft(data_j)
                pow = np.log(np.sum(F * np.conj(F)))
                signal_energy_fdomain = safe_log(np.sum(signal.square(np.abs(data_j))))
                # Implement or find Python equivalents for entropy functions
                signal_ShEn = wentropy(data_j, entType = 'shannon')
                Vpp = np.ptp(data_j)
                # Peak detection
                pks, locs = find_peaks(data_j, height=50, distance=round(Fs/50))
                signal_npks = len(pks)

                # Plot the original signal
                y_j = [signal_mean, signal_median, signal_std, signal_skewness, signal_kurtosis,
                        signal_zcd, signal_nleo_env_diff, signal_nleo_teager,
                        signal_energy_tdomain, signal_energy_fdomain,
                        signal_ShEn, Vpp, signal_npks]
                y.extend(y_j)
            # if output is None:
            #     output = y  # First assignment directly
            # else:
            #     output = np.vstack([output, y])
            output.append(y)
        output_combiner = np.vstack([
            np.mean(output, axis=0),
            np.median(output, axis=0),
            np.std(output, ddof=1, axis=0),
            skew(output, bias=False, axis=0),
            kurtosis(output, fisher=False, axis=0)
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
        #print(f"File {file_path} not found, skipping.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    data = np.load(file_path)

    feature_type = 'UTM'

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

            features = run_UTM_seg(data, Fs, montage, sec)
            if features is None:
                continue

            for idx, combiner_feature in enumerate(features):
                np.save(output_filenames[idx], combiner_feature)

    # print('File processed:', file_path)

def process_file_wrapper(args):
    process_file(*args)

def main():
    output_dir = '/home/yash@mydre.org/Documents/data/ML_EEG_TUD/data_v1/features/UTM_npy'
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