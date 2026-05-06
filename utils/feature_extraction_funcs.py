import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import correlate, butter, lfilter, find_peaks, filtfilt, hilbert
from tqdm.contrib.concurrent import process_map
import pywt
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from scipy.stats import kurtosis, skew
from scipy import signal
from scipy.fft import fft
from stockwell import st
import cmath
import bct
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
    
def run_cc_whole(data, MONTAGE):
    """
    Input is preprocessed npy files
    Output is cross-correlation of each channel with every other channel saved in npy format too
    This is for the whole data segment.
    """
    n_ch, n_samples = data.shape
    data_montage = apply_montage(data, MONTAGE)

    out_mat = np.zeros((n_ch, n_ch))
    out_temp = []

    for j in range(data_montage.shape[0]):
        for k in range(data_montage.shape[0]):
            if j > k:
                corr = correlate(data_montage[j], data_montage[k], mode='full')
                norm_factor = np.sqrt(np.sum(data_montage[j]**2) * np.sum(data_montage[k]**2))
                corr /= norm_factor
                max_corr = np.max(corr)
                out_mat[k, j] = max_corr
                out_temp.append([j+1, k+1, max_corr])

    out_temp = np.array(out_temp)
    output = out_temp[:, 2]

    return output

def run_cc_seg(data, Fs, MONTAGE, sec):
    """
    Input is preprocessed npy files
    Output is cross-correlation of each channel with every other channel saved in npy format too
    This is for other data segment.
    """
    data_whole = data.copy()

    data = apply_montage(data, MONTAGE)
    n_ch, n_samples = data.shape
    # Calculate number of segments
    seg_num = int(data.shape[1] / (Fs * sec))

    if seg_num == 0:
        output = run_cc_whole(data_whole, MONTAGE)
        combiners = ['mean', 'median', 'std', 'skewness', 'kurtosis']
        output_combiner = np.array([output for _ in combiners])
    else:
        # Process each segment
        data = data[:, :seg_num*sec*Fs]
        output = []
        for tt in range(seg_num):
            segment = data[:, tt*sec*Fs:(tt+1)*sec*Fs]
            out_mat = np.zeros((n_ch, n_ch))

            out_temp = []
            for j in range(n_ch):
                for k in range(n_ch):
                    if j > k:
                        # normalized cross-correlation
                        corr = correlate(segment[j], segment[k], mode='full')
                        norm_factor = np.sqrt(np.sum(segment[j]**2) * np.sum(segment[k]**2))
                        corr /= norm_factor
                        
                        max_corr = np.max(corr)
                        out_mat[k, j] = max_corr
                        out_temp.append([j+1, k+1, max_corr])

            out_temp = np.array(out_temp)

            out = out_temp[:, 2]
            output.append(out)
        output_combiner = np.vstack([
            np.mean(output, axis=0),
            np.median(output, axis=0),
            np.std(output, ddof=1, axis=0),
            skew(output, axis=0),
            kurtosis(output, fisher=False, axis=0)
            ])
        return output_combiner
    
def run_cwt_whole(data, Fs, MONTAGE, WAVELET_TYPE):
    #print(f'[WAVELET] MONTAGE={MONTAGE} | WAVELET={WAVELET_TYPE} | Fs={Fs} Hz | {datetime.now().strftime("%H:%M:%S")}')
    n_ch, n_samples = data.shape
    data_montage = apply_montage(data, MONTAGE)
    dt = 1 / Fs
    s0 = 2.45
    ds = 0.4875
    Nbsc = int(np.log2(n_samples * dt / s0) / ds)
    scales = s0 * 2 ** (np.arange(Nbsc) * ds)
    c_RP = [] 

    while len(scales) < 13:
        s0 -= 0.05
        Nbsc = int(np.log2(n_samples * dt / s0) / ds)
        scales = s0 * 2 ** (np.arange(Nbsc) * ds)
        if s0 <= 0:
            raise ValueError("s0 has been decremented to zero or negative value, unable to get enough scales")

    for j in range(data_montage.shape[0]):
        data_j = data_montage[j, :]
        cwtmatr, freqs = pywt.cwt(data_j, scales, WAVELET_TYPE, sampling_period=dt, method = 'fft')

        n_scales_to_pick = min(13, len(cwtmatr))
        c_all = np.abs(cwtmatr[:n_scales_to_pick, :])

        for i in range(n_scales_to_pick):
            c_temp = c_all[i, :]
            c_RP.append(np.mean(c_temp ** 2))
            c_RP.append(np.std(c_temp ** 2))
    return np.array(c_RP)  # feature vector

def run_cwt_seg(data, Fs, MONTAGE, WAVELET_TYPE, sec):
    data_whole = data.copy()

    data = apply_montage(data, MONTAGE)
    n_ch, n_samples = data.shape
    # Calculate number of segments
    seg_num = int(data.shape[1] / (Fs * sec))
    #print("s:", seg_num)
    c_RP = []
    dt = 1 / Fs
    s0 = 2.45
    ds = 0.4875
    Nbsc = int(np.log2(n_samples * dt / s0) / ds)
    scales = s0 * 2 ** (np.arange(Nbsc) * ds)

    out = np.zeros((seg_num, 26 * n_ch))

    if seg_num == 0:
        out = run_cwt_whole(data_whole, Fs, MONTAGE, WAVELET_TYPE)
        combiners = ['mean', 'median', 'std', 'skewness', 'kurtosis']
        output_combiner = np.array([out for _ in combiners])
    else:
        while len(scales) < 13:
            s0 -= 0.05
            Nbsc = int(np.log2(n_samples * dt / s0) / ds)
            scales = s0 * 2 ** (np.arange(Nbsc) * ds)
            if s0 <= 0:
                raise ValueError("s0 has been decremented to zero or negative value, unable to get enough scales")
        # Process each segment
        data = data[:, :seg_num*sec*Fs]

        for tt in range(seg_num):
            segment = data[:, tt*sec*Fs:(tt+1)*sec*Fs]
            c_RP = [] 
            for j in range(n_ch):
                data_j = segment[j, :]
                cwtmatr, freqs = pywt.cwt(data_j, scales, WAVELET_TYPE, sampling_period=dt, method = 'fft')
                n_scales_to_pick = 13  # Number of scales to pick
                c_all = np.abs(cwtmatr[:n_scales_to_pick, :])
                for i in range(n_scales_to_pick):
                    c_temp = np.array(c_all[i, :])
                    c_RP.append(np.mean(c_temp ** 2))
                    c_RP.append(np.std(c_temp ** 2))

            out[tt, :] = c_RP
        output_combiner = np.vstack([
            np.mean(out, axis=0),
            np.median(out, axis=0),
            np.std(out, ddof=1, axis=0),
            skew(out, axis=0),
            kurtosis(out, fisher=False, axis=0)
            ])
    return output_combiner

def detcoef(c, l, level):
    return c[level]

def appcoef(coeffs, wavelet, level, **kwargs):
    max_level = len(coeffs) - 1
    if level == max_level:
        return coeffs[0]
    approx = pywt.waverec(coeffs[:-level], wavelet, **kwargs)

    if np.abs(approx[-1] - approx[-2]) < 0.00001:
        approx = approx[:-1]
    
    return approx

def run_dwt_whole(data, Fs, MONTAGE, WAVELET_TYPE):
    #print(f'[WAVELET] MONTAGE={MONTAGE} | WAVELET={WAVELET_TYPE} | Fs={Fs} Hz | {datetime.now().strftime("%H:%M:%S")}')

    data_montage = apply_montage(data, MONTAGE)
    c_RP = [] 
    for j in range(data_montage.shape[0]):
        signal = data_montage[j, :]
        coeffs = pywt.wavedec(signal, WAVELET_TYPE, level=6)
        lengths = [len(c) for c in coeffs]

        cD6 = detcoef(coeffs, lengths, 1)
        cD5 = detcoef(coeffs, lengths, 2)
        cD4 = detcoef(coeffs, lengths, 3)
        cD3 = detcoef(coeffs, lengths, 4)
        cD2 = detcoef(coeffs, lengths, 5)
        cD1 = detcoef(coeffs, lengths, 6)

        cA6 = appcoef(coeffs, WAVELET_TYPE, 6)
        cA5 = appcoef(coeffs, WAVELET_TYPE, 5)
        cA4 = appcoef(coeffs, WAVELET_TYPE, 4)
        cA3 = appcoef(coeffs, WAVELET_TYPE, 3)
        cA2 = appcoef(coeffs, WAVELET_TYPE, 2)
        cA1 = appcoef(coeffs, WAVELET_TYPE, 1)
        c_all = [cD1, cD2, cD3, cD4, cD5, cD6, cA1, cA2, cA3, cA4, cA5, cA6]

        for c in c_all:
            c_temp = np.array(c)
            c_RP.append(np.mean(c_temp ** 2))
            c_RP.append(np.std(c_temp ** 2))

    return np.array(c_RP)  # feature vector

def run_dwt_seg(data, Fs, MONTAGE, WAVELET, sec):
    data_whole = data.copy()
    data = apply_montage(data, MONTAGE)
    # Calculate number of segments
    seg_num = int(data.shape[1] / (Fs * sec))
    n_ch = data.shape[0]
    seg_len = sec * Fs
    out = np.zeros((seg_num, 24 * n_ch))
    if seg_num == 0:
        output = run_dwt_whole(data_whole, Fs, MONTAGE, WAVELET)
        combiners = ['mean', 'median', 'std', 'skewness', 'kurtosis']
        output_combiner = np.array([output for _ in combiners])
    else:
        # Process each segment
        data = data[:, :seg_num*sec*Fs]
        output = None

        for tt in range(seg_num):
            segment = data[:, tt * seg_len:(tt + 1) * seg_len]
            c_RP = []
            
            for j in range(n_ch):
                signal = segment[j, :]
                coeffs = pywt.wavedec(signal, WAVELET, level=6)
                lengths = [len(c) for c in coeffs]

                cD6 = detcoef(coeffs, lengths, 1)
                cD5 = detcoef(coeffs, lengths, 2)
                cD4 = detcoef(coeffs, lengths, 3)
                cD3 = detcoef(coeffs, lengths, 4)
                cD2 = detcoef(coeffs, lengths, 5)
                cD1 = detcoef(coeffs, lengths, 6)

                cA6 = appcoef(coeffs, WAVELET, 6)
                cA5 = appcoef(coeffs, WAVELET, 5)
                cA4 = appcoef(coeffs, WAVELET, 4)
                cA3 = appcoef(coeffs, WAVELET, 3)
                cA2 = appcoef(coeffs, WAVELET, 2)
                cA1 = appcoef(coeffs, WAVELET, 1)
                c_all = [cD1, cD2, cD3, cD4, cD5, cD6, cA1, cA2, cA3, cA4, cA5, cA6]

                for c in c_all:
                    c_temp = np.array(c)
                    c_RP.append(np.mean(c_temp ** 2))
                    c_RP.append(np.std(c_temp ** 2))

            out[tt, :] = c_RP

        output_combiner = np.vstack([
            np.mean(out, axis=0),
            np.median(out, axis=0),
            np.std(out, ddof=1, axis=0),
            skew(out, bias=False, axis=0),
            kurtosis(out, fisher=False, axis=0)
            ])
        return output_combiner

def normalize(W):
    W = W / np.max(np.abs(W))
    return W

def autofix(W):
    # Clear diagonal
    np.fill_diagonal(W, 0)
    
    # Remove Infs and NaNs
    W[np.isinf(W) | np.isnan(W)] = 0
    
    # Ensure exact binariness
    U = np.unique(W)
    if len(U) > 1:
        idx_0 = np.abs(W) < 1e-10
        idx_1 = np.abs(W - 1) < 1e-10
        if np.all(idx_0 | idx_1):
            W[idx_0] = 0
            W[idx_1] = 1
    
    # Ensure exact symmetry
    if not np.array_equal(W, W.T):
        if np.max(np.abs(W - W.T)) < 1e-10:
            W = (W + W.T) / 2
    
    return W

def eigenvector_centrality_und(network):
    '''
    Eigenvector centrality is a self-referential measure of centrality:
    nodes have high eigenvector centrality if they connect to other nodes
    that have high eigenvector centrality. The eigenvector centrality of
    node i is equivalent to the ith element in the eigenvector
    corresponding to the largest eigenvalue of the adjacency matrix.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary/weighted undirected adjacency matrix

    Returns
    -------
    v : Nx1 np.ndarray
        eigenvector associated with the largest eigenvalue of the matrix
    '''
    n = len(network)
    
    if n < 1000:
        vals, vecs = eig(network)
    else:
        vals, vecs = eigs(csr_matrix(network), k=1, which='LM')
    
    idx = np.argmax(np.real(vals))
    ec = np.abs(vecs[:, idx])
    
    # Reshape the eigenvector to ensure it is a column vector
    v = ec.reshape(len(ec), 1)

    # Debugging: Print eigenvalues and the selected eigenvector
    #print("Eigenvalues:", vals)
    #print("Selected eigenvector (pre-normalization):", vecs[:, idx])
    #print("Selected eigenvector (post-normalization):", ec)

    return v

def cuberoot(x):
    """
    Calculate the cubic root of x, handling negative values correctly.
    """
    return np.sign(x) * np.abs(x) ** (1 / 3)

def clustering_coef_wu(W):
    """
    The weighted clustering coefficient is the average "intensity" (geometric mean)
    of all triangles associated with each node.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix (all weights must be between 0 and 1)

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    """
    W=np.clip(W,0,1)
    # Ensure weights are between 0 and 1
    if np.max(W) > 1 or np.min(W) < 0:
        print(f"Max weight: {np.max(W)}")
        print(f"Min weight: {np.min(W)}")
        raise ValueError("All weights must be between 0 and 1")

    # Degrees calculation
    K = np.sum(W != 0, axis=1)
    #print(f"Degrees (K): {K}")

    # Number of 3-cycles
    W_cubed_root = np.power(W, 1/3)
    cyc3 = np.diag(np.linalg.matrix_power(W_cubed_root, 3))
    #print(f"Number of 3-cycles (cyc3): {cyc3}")

    # Handle no 3-cycles case
    K = K.astype(float)
    K[cyc3 == 0] = np.inf
    #print(f"Degrees after handling 3-cycles (K): {K}")

    # Clustering coefficient calculation
    C = cyc3 / (K * (K - 1))
    #print(f"Clustering Coefficient (C): {C}")

    return C

def custom_local_assortativity_wu_sign(W):
    '''
    Custom implementation of local assortativity for undirected weighted/signed networks.

    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    
    Returns
    -------
    loc_assort_pos : Nx1 np.ndarray
        local assortativity from positive weights
    loc_assort_neg : Nx1 np.ndarray
        local assortativity from negative weights
    '''
    n = len(W)
    np.fill_diagonal(W, 0)
    r_pos = bct.assortativity_wei(W * (W > 0))
    r_neg = bct.assortativity_wei(-W * (W < 0))

    str_pos, str_neg, _, _ = bct.strengths_und_sign(W)

    loc_assort_pos = np.full((n,), np.nan)
    loc_assort_neg = np.full((n,), np.nan)

    for curr_node in range(n):
        j_pos = np.where(W[curr_node, :] > 0)[0]
        if str_pos[curr_node] != 0:
            loc_assort_pos[curr_node] = np.sum(np.abs(str_pos[j_pos] - str_pos[curr_node])) / str_pos[curr_node]
        
        j_neg = np.where(W[curr_node, :] < 0)[0]
        if str_neg[curr_node] != 0:
            loc_assort_neg[curr_node] = np.sum(np.abs(str_neg[j_neg] - str_neg[curr_node])) / str_neg[curr_node]

    if not np.isnan(np.sum(loc_assort_pos)):
        loc_assort_pos = ((r_pos + 1) / n) - (loc_assort_pos / np.nansum(loc_assort_pos))
    if not np.isnan(np.sum(loc_assort_neg)):
        loc_assort_neg = ((r_neg + 1) / n) - (loc_assort_neg / np.nansum(loc_assort_neg))

    return loc_assort_pos, loc_assort_neg

def modularity_und(A, gamma=1):
    """
    Calculate the optimal community structure and modularity for a weighted undirected connection matrix A.
    
    Parameters
    ----------
    A : NxN np.ndarray
        Weighted undirected connection matrix.
    gamma : float, optional
        Resolution parameter. Default is 1.
    
    Returns
    -------
    ci : Nx1 np.ndarray
        Community structure indices.
    Q : float
        Modularity value.
    """
    N = len(A)                            # number of vertices
    K = np.sum(A, axis=1)                 # degree
    m = np.sum(K)                         # number of edges (each undirected edge is counted twice)
    B = A - gamma * np.outer(K, K) / m    # modularity matrix
    Ci = np.ones(N)                       # community indices
    cn = 1                                # number of communities
    U = [1, 0]                            # array of unexamined communities

    ind = np.arange(N)
    Bg = B
    Ng = N

    while U[0] != 0:                      # examine community U(1)
        vals, vecs = eig(Bg)
        i1 = np.argmax(vals.real)         # maximal positive (real part of) eigenvalue of Bg
        v1 = vecs[:, i1].real             # corresponding eigenvector

        S = np.ones(Ng)
        S[v1 < 0] = -1
        q = np.dot(S.T, np.dot(Bg, S))    # contribution to modularity

        if q > 1e-10:                     # contribution positive: U(1) is divisible
            qmax = q                      # maximal contribution to modularity
            np.fill_diagonal(Bg, 0)       # Bg is modified, to enable fine-tuning
            indg = np.ones(Ng)            # array of unmoved indices
            Sit = S
            while np.any(indg):           # iterative fine-tuning
                Qit = qmax - 4 * Sit * (Bg @ Sit)  # this line is equivalent to:
                qmax_candidate, imax = np.max(Qit * indg), np.argmax(Qit * indg)

                if np.isnan(qmax_candidate) or qmax_candidate <= qmax:
                    break

                qmax = qmax_candidate
                Sit[imax] = -Sit[imax]
                indg[imax] = np.nan

                if qmax > q:
                    q = qmax
                    S = Sit

            if abs(np.sum(S)) == Ng:      # unsuccessful splitting of U(1)
                U.pop(0)
            else:
                cn += 1
                Ci[ind[S == 1]] = U[0]    # split old U(1) into new U(1) and into cn
                Ci[ind[S == -1]] = cn
                U = [cn] + U

        else:                             # contribution nonpositive: U(1) is indivisible
            U.pop(0)

        if len(U) > 0:
            ind = np.where(Ci == U[0])[0]     # indices of unexamined community U(1)
            bg = B[ind][:, ind]
            Bg = bg - np.diag(np.sum(bg, axis=0))  # modularity matrix for U(1)
            Ng = len(ind)                     # number of vertices in U(1)

    s = np.tile(Ci, (N, 1))               # compute modularity
    Q = np.sum((s == s.T) * B / m)        # logical comparison to check community membership
    return Ci, Q

def compute_network_metrics(conn_3d):
    nNets = conn_3d.shape[2]
    network_measures = {
        'nodeDeg': [],
        'nodeStr': [],
        'nodeAssort': [],
        'nodePathL': [],
        'nodeLocEff': [],
        'nodeLocEffB': [],
        'nodeEcc': [],
        'nodeBetCent': [],
        'nodeBetCentB': [],
        'nodeEVCent': [],
        'nodeEVCentB': [],
        'nodeCC': [],
        'nodeCoreness': [],
        'nodePartCoef': [],
        'nodeDivCoef': [],
        'netAssorCoeff': [],
        'netGlobEff': [],
        'netRadius': [],
        'netDia': [],
        'netTrans': [],
        'netTransB': [],
        'netRichClub': [],
        'netEdgOvap': [],
        'netNodePairDeg': [],
        'edgeMatchIdx': []
    }

    for net in range(nNets):
        network = conn_3d[:, :, net]
        if network.shape[0] != network.shape[1]:
            raise ValueError(f"Network matrix at index {net} is not square: {network.shape}")
        nNodes = network.shape[0]
        
        #print(f"Processing network {net+1}/{nNets}")
        #print(f"Network shape: {network.shape}")
        
        if np.any(np.isnan(network)) or np.any(np.isinf(network)):
            #print(f"Network at index {net} contains NaN or Inf values. Skipping this network.")
            continue
        
        ci, q = modularity_und(network)
        ci = np.asarray(ci, dtype=int)

        network_measures['nodeDeg'].append(bct.degrees_und(network))
        network_measures['nodeStr'].append(bct.strengths_und(network))
        
        loc_assort_pos, loc_assort_neg = custom_local_assortativity_wu_sign(network)
        nodeAssort = np.nan_to_num(loc_assort_pos) + np.nan_to_num(loc_assort_neg)
        network_measures['nodeAssort'].append(nodeAssort)
        
        network_measures['netAssorCoeff'].append(bct.assortativity_wei(network, 0))

        distNet = bct.distance_wei_floyd(network, 'inv')[0]
        np.fill_diagonal(distNet, np.nan)
        distNet[np.isinf(distNet)] = np.nan
        nodePathL = np.nanmean(distNet, axis=1)
        network_measures['nodePathL'].append(nodePathL)

        network_measures['netGlobEff'].append(bct.efficiency_wei(network, 'global'))
        network_measures['nodeLocEff'].append(bct.efficiency_wei(network, 'local'))
        bin_network = bct.utils.binarize(network)
        network_measures['nodeLocEffB'].append(bct.efficiency_bin(bin_network, 1))

        distNet = bct.distance_wei_floyd(network, 'inv')[0]
        _, _, ecc, radius, diameter = bct.charpath(distNet)
        network_measures['nodeEcc'].append(ecc)
        network_measures['netRadius'].append(radius)
        network_measures['netDia'].append(diameter)

        nodeBetCent = bct.betweenness_wei(bct.utils.invert(network))
        nodeBetCent = (2 * nodeBetCent) / ((nNodes - 1) * (nNodes - 2))
        network_measures['nodeBetCent'].append(nodeBetCent)

        nodeBetCentB = bct.betweenness_bin(bin_network)
        nodeBetCentB = (2 * nodeBetCentB) / ((nNodes - 1) * (nNodes - 2))
        network_measures['nodeBetCentB'].append(nodeBetCentB)
        network_measures['nodeEVCent'].append(eigenvector_centrality_und(network))
        network_measures['nodeEVCentB'].append(eigenvector_centrality_und(bin_network))


        network_measures['nodeCC'].append(clustering_coef_wu(network))
        network_measures['netTrans'].append(bct.transitivity_wu(network))
        network_measures['netTransB'].append(bct.transitivity_bu(bin_network))
        network_measures['nodeCoreness'].append(bct.kcoreness_centrality_bu(bin_network))
        network_measures['netRichClub'].append(np.nanmean(bct.rich_club_wu(network)))

        _, ec, degij = bct.edge_nei_overlap_bu(bin_network)
        network_measures['netEdgOvap'].append(np.mean(ec))
        network_measures['netNodePairDeg'].append(np.mean(np.mean(degij)))

        network_measures['edgeMatchIdx'].append(bct.matching_ind_und(network))
        network_measures['nodePartCoef'].append(bct.participation_coef(network, ci))
        network_measures['nodeDivCoef'].append(bct.diversity_coef_sign(network, ci))

    metricsNode = {
        'nodeDeg': network_measures['nodeDeg'],
        'nodeStr': network_measures['nodeStr'],
        'nodeAssort': network_measures['nodeAssort'],
        'nodePathL': network_measures['nodePathL'],
        'nodeLocEff': network_measures['nodeLocEff'],
        'nodeLocEffB': network_measures['nodeLocEffB'],
        'nodeEcc': network_measures['nodeEcc'],
        'nodeBetCent': network_measures['nodeBetCent'],
        'nodeBetCentB': network_measures['nodeBetCentB'],
        'nodeEVCent': network_measures['nodeEVCent'],
        'nodeEVCentB': network_measures['nodeEVCentB'],
        'nodeCC': network_measures['nodeCC'],
        'nodeCoreness': network_measures['nodeCoreness'],
        'nodePartCoef': network_measures['nodePartCoef'],
        'nodeDivCoef': network_measures['nodeDivCoef'],
    }

    f1 = []
    for sub in range(nNets):
        f1_sub = []
        for metric in metricsNode:
            f1_sub.append(np.mean(metricsNode[metric][sub]))
        f1.append(f1_sub)

    metricsAggr = {
        'netAssorCoeff': network_measures['netAssorCoeff'],
        'netGlobEff': network_measures['netGlobEff'],
        'netRadius': network_measures['netRadius'],
        'netDia': network_measures['netDia'],
        'netTrans': network_measures['netTrans'],
        'netTransB': network_measures['netTransB'],
        'netRichClub': network_measures['netRichClub'],
        'netEdgOvap': network_measures['netEdgOvap'],
        'netNodePairDeg': network_measures['netNodePairDeg'],
    }
    f2 = np.array([metricsAggr[metric] for metric in metricsAggr]).T

    metricsEdge = {
        'edgeMatchIdx': network_measures['edgeMatchIdx']
    }

    f3 = []
    for sub in range(nNets):
        f3_sub = []
        for metric in metricsEdge:
            f3_sub.append(np.mean(metricsEdge[metric][sub]))
        f3.append(f3_sub)

    metricsNetwork = {
        'metricsNode': metricsNode,
        'metricsAggr': metricsAggr,
        'metricsEdge': metricsEdge
    }
    features = np.hstack([f1, f2, f3])
    features_trimmed = np.delete(features, [5, 8, 10, 20, 21], axis=1)  # remove specific features as mentioned
    features_trimmed = features_trimmed[0,:]
    return features_trimmed

def run_gcc_whole(data, MONTAGE):
    data_montage = apply_montage(data, MONTAGE)
    n_ch, n_samples = data_montage.shape
    out_mat = np.zeros((n_ch, n_ch))
    output = []
    for j in range(data_montage.shape[0]):
        for k in range(data_montage.shape[0]):
            if j > k:
                # Compute normalized cross-correlation
                corr = correlate(data_montage[j], data_montage[k], mode='full')
                norm_factor = np.sqrt(np.sum(data_montage[j]**2) * np.sum(data_montage[k]**2))
                corr /= norm_factor
                max_corr = np.max(corr)
                out_mat[k, j] = max_corr
            else:
                out_mat[k,j] = 1
    #print(out_mat.shape)
    conn = np.triu(out_mat) + np.triu(out_mat, 1).T
    conn_2d = np.zeros((n_ch, n_ch, 2))
    conn_2d[:, :, 0] = conn
    conn_2d[:, :, 1] = conn

    nNets = conn_2d.shape[2]
    n_ch = conn_2d.shape[0]
    conn_3d = np.zeros((n_ch, n_ch, 2))
    for net in range(nNets):
        fixed_conn = autofix(conn_2d[:, :, net])
        normalized_conn = normalize(fixed_conn)
        conn_3d[:, :, net] = normalized_conn
    metrics = compute_network_metrics(conn_3d)
    output.append(metrics)
    output = np.array(output)
    return output

def run_gcc_seg(data, Fs, MONTAGE, sec):
    data_whole = data.copy()

    data = apply_montage(data, MONTAGE)
    n_ch, n_samples = data.shape
    seg_num = int(data.shape[1] / (Fs * sec))

    if seg_num == 0:
        output = run_gcc_whole(data_whole, MONTAGE)
        combiners = ['mean', 'median', 'std', 'skewness', 'kurtosis']
        output_combiner = np.array([output for _ in combiners])
    else:

        data = data[:, :seg_num*sec*Fs]
        output = []
        for tt in range(seg_num):
            segment = data[:, tt*sec*Fs:(tt+1)*sec*Fs]
            out_mat = np.zeros((n_ch, n_ch))
            for j in range(n_ch):
                for k in range(n_ch):
                    if j > k:
                        # Compute normalized cross-correlation
                        corr = correlate(segment[j], segment[k], mode='full')
                        norm_factor = np.sqrt(np.sum(segment[j]**2) * np.sum(segment[k]**2))
                        corr /= norm_factor
                        max_corr = np.max(corr)
                        out_mat[k, j] = max_corr
                    else:
                        out_mat[k,j] = 1

            conn = np.triu(out_mat) + np.triu(out_mat, 1).T
            conn_2d = np.zeros((n_ch, n_ch, 2))
            conn_2d[:, :, 0] = conn
            conn_2d[:, :, 1] = conn

            nNets = conn_2d.shape[2]
            n_ch = conn_2d.shape[0]
            conn_3d = np.zeros((n_ch, n_ch, 2))
            for net in range(nNets):
                fixed_conn = autofix(conn_2d[:, :, net])
                normalized_conn = normalize(fixed_conn)
                conn_3d[:, :, net] = normalized_conn
            metrics = compute_network_metrics(conn_3d)
            output.append(metrics)

        output_combiner = np.vstack([
            np.mean(output, axis=0),
            np.median(output, axis=0),
            np.std(output, ddof=1, axis=0),
            skew(output, bias= False, axis=0),
            kurtosis(output, fisher=False, axis=0)
            ])
        return output_combiner
    
def PLV_John(phase_sig1, phase_sig2):
    # Vectorized computation of PLV
    phase_diff = phase_sig1 - phase_sig2
    e = np.exp(1j * phase_diff)
    plv = np.abs(np.sum(e)) / len(e)
    return plv

def run_gplv_whole(data, Fs, MONTAGE):
    data_montage = apply_montage(data, MONTAGE)
    n_ch, n_samples = data_montage.shape

    output = []

    out_mat = np.zeros((n_ch, n_ch))
    for j in range(n_ch):
        a1 = data_montage[j, :]
        for k in range(n_ch):
            a2 = data_montage[k, :]
            if j > k:
                out_mat[k, j] = PLV_John(a1, a2)
            else:
                out_mat[k,j] = 1

    conn = np.triu(out_mat) + np.triu(out_mat, 1).T
    conn_2d = np.zeros((n_ch, n_ch, 2))
    conn_2d[:, :, 0] = conn
    conn_2d[:, :, 1] = conn

    nNets = conn_2d.shape[2]
    n_ch = conn_2d.shape[0]
    conn_3d = np.zeros((n_ch, n_ch, 2))
    for net in range(nNets):
        fixed_conn = autofix(conn_2d[:, :, net])
        normalized_conn = normalize(fixed_conn)
        conn_3d[:, :, net] = normalized_conn
    metrics = compute_network_metrics(conn_3d)
    output.append(metrics)
    output = np.array(output)
    return output

def run_gplv_seg(data, Fs, MONTAGE, sec):
    data_whole = data.copy()

    data = apply_montage(data, MONTAGE)
    n_ch, n_samples = data.shape
    seg_num = int(data.shape[1] / (Fs * sec))

    if seg_num == 0:
        output = run_gplv_whole(data_whole, MONTAGE)
        combiners = ['mean', 'median', 'std', 'skewness', 'kurtosis']
        output_combiner = np.array([output for _ in combiners])
    else:

        data = data[:, :seg_num*sec*Fs]
        output = []
        for tt in range(seg_num):
            segment = data[:, tt*sec*Fs:(tt+1)*sec*Fs]
            out_mat = np.zeros((n_ch, n_ch))
            for j in range(n_ch):
                a1 = segment[j, :]
                for k in range(n_ch):
                    a2 = segment[k, :]
                    if j > k:
                        out_mat[k, j] = PLV_John(a1, a2)
                    else:
                        out_mat[k,j] = 1

            conn = np.triu(out_mat) + np.triu(out_mat, 1).T
            conn_2d = np.zeros((n_ch, n_ch, 2))
            conn_2d[:, :, 0] = conn
            conn_2d[:, :, 1] = conn

            nNets = conn_2d.shape[2]
            n_ch = conn_2d.shape[0]
            conn_3d = np.zeros((n_ch, n_ch, 2))
            for net in range(nNets):
                fixed_conn = autofix(conn_2d[:, :, net])
                normalized_conn = normalize(fixed_conn)
                conn_3d[:, :, net] = normalized_conn
            metrics = compute_network_metrics(conn_3d)
            output.append(metrics)

        output_combiner = np.vstack([
            np.mean(output, axis=0),
            np.median(output, axis=0),
            np.std(output, ddof=1, axis=0),
            skew(output, bias= False, axis=0),
            kurtosis(output, fisher=False, axis=0)
            ])
        return output_combiner
    
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

def run_mST_seg(data, Fs, epoch_width, MONTAGE):
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

            st_out_org = st.st(timeseries_segment, lo=minfreq, hi=maxfreq, gamma=0.9, win_type='gauss')

            sr_power_batches = []

            for fmin, fmax in bands:
                fmin_index, fmax_index = int(fmin), int(fmax)
                st_out_band = st_out_org[fmin_index:fmax_index, :]

                sr = np.sqrt(np.std(st_out_band, axis=0))
                sr_mean = np.mean(sr)

                sr_power_batches.extend([sr_mean])

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

def run_conn_phase(data, Fs, MONTAGE):
    eeg_montage = apply_montage(data, MONTAGE)
    
    filters = {
        'raw': (None, None),  # No filter for raw data
        'delta': butter(4, 4 / (Fs / 2), 'low'),
        'theta': butter(4, [f / (Fs / 2) for f in [4, 7.9]], 'band'),
        'alpha': butter(4, [f / (Fs / 2) for f in [8, 13]], 'band'),
        'beta': butter(4, [f / (Fs / 2) for f in [13, 30]], 'band'),
        'gamma': butter(4, [f / (Fs / 2) for f in [31, (Fs / 2) - 0.1]], 'band')
    }

    # Pre-filtering and Hilbert Transform
    filtered_signals = {}
    hilbert_transforms = {}
    for band, (b, a) in filters.items():
        if band == 'raw':
            filtered_signals[band] = [signal for signal in eeg_montage]
        else:
            filtered_signals[band] = [lfilter(b, a, signal) for signal in eeg_montage]
        hilbert_transforms[band] = [np.angle(hilbert(filtered_signal)) for filtered_signal in filtered_signals[band]]

    # Compute PLVs
    n_ch = len(eeg_montage)
    out_temp = []
    for j in range(n_ch):
        for k in range(j):  # Ensuring j > k to avoid redundant calculations
            plvs = [
                PLV_John(hilbert_transforms[band][j], hilbert_transforms[band][k])
                for band in filters.keys()  # Including raw and all bands
            ]
            out_temp.append(plvs)

    out_temp = np.array(out_temp)

    output = out_temp.T.flatten()

    return output

def run_plv_seg(data, Fs, MONTAGE, sec):
    data_whole = data.copy()
    eeg_montage = apply_montage(data_whole, MONTAGE)
    n_ch, n_samples = eeg_montage.shape
    seg_num = int(eeg_montage.shape[1] / (Fs * sec))

    segment_outputs = []

    if seg_num == 0:
        output = run_conn_phase(data_whole, Fs, MONTAGE)
        combiners = ['mean', 'median', 'std', 'skewness', 'kurtosis']
        output_combiner = np.array([output for _ in combiners])

    else:

        # Process each segment
        for tt in range(seg_num):
            segment = eeg_montage[:, tt*sec*Fs:(tt+1)*sec*Fs]
            filtered_signals = {}
            hilbert_transforms = {}

            filters = {
                'raw': (None, None),  # No filter for raw data
                'delta': butter(4, 4 / (Fs / 2), 'low'),
                'theta': butter(4, [f / (Fs / 2) for f in [4, 7.9]], 'band'),
                'alpha': butter(4, [f / (Fs / 2) for f in [8, 13]], 'band'),
                'beta': butter(4, [f / (Fs / 2) for f in [13, 30]], 'band'),
                'gamma': butter(4, [f / (Fs / 2) for f in [31, (Fs / 2) - 0.1]], 'band')
            }

            for band, (b, a) in filters.items():
                if band == 'raw':
                    filtered_signals[band] = segment
                else:
                    filtered_signals[band] = lfilter(b, a, segment, axis=1)
                hilbert_transforms[band] = [np.angle(hilbert(filtered_signal)) for filtered_signal in filtered_signals[band]]

            out_temp = []
            out_temp2 = []
            for j in range(n_ch):
                for k in range(j):
                    plvs = [PLV_John(hilbert_transforms[band][j], hilbert_transforms[band][k]) for band in filters.keys()]
                    out_temp.append(plvs)
                    out_temp2 = np.array(out_temp)
            segment_outputs.append(out_temp2.T.flatten())

        # Combine results from all segments
        segment_outputs = np.array(segment_outputs)  # Shape: (segments, combinations, bands)
        output_combiner = np.vstack([
            np.mean(segment_outputs, axis=0),
            np.median(segment_outputs, axis=0),
            np.std(segment_outputs, ddof=1, axis=0),
            skew(segment_outputs, bias=False, axis=0),
            kurtosis(segment_outputs, axis=0, fisher=False)
        ])

        return output_combiner

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
        j_pdf = np.histogram(data_j, bins=1024, density=True)[0]
        signal_ShEn = entropy(j_pdf, base=2)
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
                j_pdf = np.histogram(data_j, bins=1024, density=True)[0]
                signal_ShEn = entropy(j_pdf, base=2)
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