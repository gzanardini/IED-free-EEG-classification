import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import bct
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from multiprocessing import cpu_count
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
    # Ensure weights are between 0 and 1
    if np.max(W) > 1 or np.min(W) < 0:
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

    feature_type = 'GPLV'

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

            features = run_gplv_seg(data, Fs, montage, sec)
            if features is None:
                continue

            for idx, combiner_feature in enumerate(features):
                np.save(output_filenames[idx], combiner_feature)

    # print('File processed:', file_path)

def process_file_wrapper(args):
    process_file(*args)

def main():
    output_dir = '/home/yash@mydre.org/Documents/data/ML_EEG_TUD/data_v1/features/GPLV_npy'
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