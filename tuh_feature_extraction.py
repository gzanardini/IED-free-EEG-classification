import enum
import os
import sys
import numpy as np
from utils import yash_features as yf
import pickle as pkl
import mne
import pandas as pd

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

photostimulation_data_path = '/space/gzanardini/tuh_photostimulation_fulltrials'
feature_path= '/space/gzanardini/tuh_features_whole/'

stim_samples=pkl.load(open(os.path.join(photostimulation_data_path, 'stim_samples.pkl'), 'rb'))
description=pd.read_csv(os.path.join(photostimulation_data_path, 'stim_df.csv'))

FS=int(stim_samples[0].info['sfreq'])

combiner_names = ['mean', 'median', 'std', 'skewness', 'kurtosis']

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']

segment_lengths = [20,60]

# for montage in montages:
#     for segment_length in segment_lengths:
#         print(f'Processing montage: {montage}, segment length: {segment_length}')
#         spectral_features = []
#         for index,sample in enumerate(stim_samples):
#             print(f'Processing sample {index+1}/{len(stim_samples)}')
#             data= sample.get_data()

#             print(f"Data shape: {data.shape}")
#             spectral_features.append(yf.run_spectral_seg2(data,FS,MONTAGE=montage,sec=segment_length))
#         spectral_features = np.array(spectral_features)
#         print(f'Finished processing montage: {montage}, segment length: {segment_length}')
#         # Save the features
#         np.save(os.path.join(feature_path, f'spectral_{montage}_{segment_length}s.npy'), spectral_features)

for montage in montages:
    for segment_lenght in segment_lengths:
        print(f"Starting cwt - {montage} {segment_lenght}s")
        cwt_features = []     
        for index,sample in enumerate(stim_samples):
            print(f'Processing sample {index+1}/{len(stim_samples)}')
            data= sample.get_data()
            cwt_features.append(yf.run_cwt_seg(data, Fs=FS, MONTAGE=montage,WAVELET_TYPE='morl' ,sec=segment_lenght))
        cwt_features = np.array(cwt_features)
        print(f'Finished processing montage: {montage}, segment length: {segment_lenght}')
        # Save the features
        np.save(os.path.join(feature_path, f'cwt_{montage}_{segment_lenght}s.npy'), cwt_features)

for montage in montages:
    for segment_lenght in segment_lengths:
        print(f"Starting dwt - {montage} {segment_lenght}s")
        dwt_features = []
        for index,sample in enumerate(stim_samples):
            print(f'Processing sample {index+1}/{len(stim_samples)}')
            data= sample.get_data()
            dwt_features.append(yf.run_dwt_seg(data, Fs=FS, MONTAGE=montage, WAVELET='db4', sec=segment_lenght))
        dwt_features = np.array(dwt_features)
        print(f'Finished processing montage: {montage}, segment length: {segment_lenght}')
        # Save the features
        np.save(os.path.join(feature_path, f'dwt_{montage}_{segment_lenght}s.npy'), dwt_features)

## MST 

for montage in montages:
    for segment_lenght in segment_lengths:
        print(f"Starting mst - {montage} {segment_lenght}s")
        mst_features = []
        for index,sample in enumerate(stim_samples):
            print(f'Processing sample {index+1}/{len(stim_samples)}')
            data= sample.get_data()
            mst_features.append(yf.run_mST_seg(data, Fs=FS, MONTAGE=montage, epoch_width=segment_lenght))
        
        mst_features = np.array(mst_features)
        print(f'Finished {montage} {segment_lenght}s')
        print(f"Shape: {mst_features.shape}")
        np.save(os.path.join(feature_path, f"mst_{montage}_{segment_lenght}s.npy"), mst_features)
        

## SST

for montage in montages:
    for segment_lenght in segment_lengths:
        sst_features = []
        print(f"Starting sst - {montage} {segment_lenght}s")
        for index,sample in enumerate(stim_samples):
            print(f'Processing sample {index+1}/{len(stim_samples)}')
            data= sample.get_data()
            sst_features.append(yf.run_sST_seg(data, Fs=FS, MONTAGE=montage, epoch_width=segment_lenght))
        sst_features = np.array(sst_features)
        print(f'Finished {montage} {segment_lenght}s')
        print(f"Shape: {sst_features.shape}")
        np.save(os.path.join(feature_path, f"sst_{montage}_{segment_lenght}s.npy"), sst_features)


## UTM

for montage in montages:
    for segment_lenght in segment_lengths:
        print(f"Starting utm - {montage} {segment_lenght}s")
        utm_features = []
        for index,sample in enumerate(stim_samples):
            print(f'Processing sample {index+1}/{len(stim_samples)}')
            data= sample.get_data()
            utm_features.append(yf.run_UTM_seg(data, Fs=FS, MONTAGE=montage, sec=segment_lenght))

        utm_features = np.array(utm_features)
        print(f'Finished {montage} {segment_lenght}s')
        print(f"Shape: {utm_features.shape}")
        np.save(os.path.join(feature_path, f"utm_{montage}_{segment_lenght}s.npy"), utm_features)

## PLV 

for montage in montages:
    for segment_lenght in segment_lengths:
        print(f"Starting plv - {montage} {segment_lenght}s")
        plv_features = []   
        for index,sample in enumerate(stim_samples):
            print(f'Processing sample {index+1}/{len(stim_samples)}')
            data= sample.get_data()
            plv_features.append(yf.run_plv_seg(data, Fs=FS, MONTAGE=montage, sec=segment_lenght))
        
        plv_features = np.array(plv_features)
        print(f'Finished {montage} {segment_lenght}s')
        print(f"Shape: {plv_features.shape}")
        np.save(os.path.join(feature_path, f"plv_{montage}_{segment_lenght}s.npy"), plv_features)

## GPLV

for montage in montages:
    for segment_lenght in segment_lengths:
        print(f"Starting gplv - {montage} {segment_lenght}s")
        gplv_features = []
        for index,sample in enumerate(stim_samples):
            print(f'Processing sample {index+1}/{len(stim_samples)}')
            data= sample.get_data()
            with HiddenPrints():
                gplv_features.append(yf.run_gplv_seg(data, Fs=FS, MONTAGE=montage, sec=segment_lenght))

        gplv_features = np.array(gplv_features)
        print(f'Finished {montage} {segment_lenght}s')
        print(f"Shape: {gplv_features.shape}")
        np.save(os.path.join(feature_path, f"gplv_{montage}_{segment_lenght}s.npy"), gplv_features)

## CC 

for montage in montages:
    for segment_lenght in segment_lengths:
        print(f"Starting cc - {montage} {segment_lenght}s")
        cc_features = []         
        for index,sample in enumerate(stim_samples):
            print(f'Processing sample {index+1}/{len(stim_samples)}')
            data= sample.get_data()
            cc_features.append(yf.run_cc_seg(data, Fs=FS, MONTAGE=montage, sec=segment_lenght))
        
        cc_features = np.array(cc_features)
        print(f'Finished {montage} {segment_lenght}s')
        print(f"Shape: {cc_features.shape}")
        np.save(os.path.join(feature_path, f"cc_{montage}_{segment_lenght}s.npy"), cc_features)
        

## GCC
for montage in montages:
    for segment_lenght in segment_lengths:
        gcc_features = []
        for index,sample in enumerate(stim_samples):
            print(f'Processing sample {index+1}/{len(stim_samples)}')
            data= sample.get_data()
            with HiddenPrints():
                gcc_features.append(yf.run_gcc_seg(data, Fs=FS, MONTAGE=montage, sec=segment_lenght))
        
        gcc_features = np.array(gcc_features)
        print(f'Finished {montage} {segment_lenght}s')
        print(f"Shape: {gcc_features.shape}")
        np.save(os.path.join(feature_path, f"gcc_{montage}_{segment_lenght}s.npy"), gcc_features)

