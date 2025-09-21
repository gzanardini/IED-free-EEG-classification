import os
import sys
import numpy as np
from utils import yash_features as yf
import pickle as pkl
import mne
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
import datetime
import traceback
import threading
from collections import OrderedDict

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def log_message(message, level="INFO"):
    """Enhanced logging with timestamps"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()  # Ensure immediate output for nohup

def process_sample_spectral(args):
    """Process a single sample for spectral features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:  # Print progress every 50 samples to reduce log spam
        log_message(f'Spectral progress: {index+1}/{total_samples} samples processed')
    return yf.run_spectral_seg2(sample, FS, MONTAGE=montage, sec=segment_length)

def process_sample_cwt(args):
    """Process a single sample for CWT features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:
        log_message(f'CWT progress: {index+1}/{total_samples} samples processed')
    return yf.run_cwt_seg(sample, Fs=FS, MONTAGE=montage, WAVELET_TYPE='morl', sec=segment_length)

def process_sample_dwt(args):
    """Process a single sample for DWT features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:
        log_message(f'DWT progress: {index+1}/{total_samples} samples processed')
    return yf.run_dwt_seg(sample, Fs=FS, MONTAGE=montage, WAVELET='db4', sec=segment_length)

def process_sample_mst(args):
    """Process a single sample for MST features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:
        log_message(f'MST progress: {index+1}/{total_samples} samples processed')
    return yf.run_mST_seg(sample, Fs=FS, MONTAGE=montage, epoch_width=segment_length)

def process_sample_sst(args):
    """Process a single sample for SST features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:
        log_message(f'SST progress: {index+1}/{total_samples} samples processed')
    return yf.run_sST_seg(sample, Fs=FS, MONTAGE=montage, epoch_width=segment_length)

def process_sample_utm(args):
    """Process a single sample for UTM features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:
        log_message(f'UTM progress: {index+1}/{total_samples} samples processed')
    return yf.run_UTM_seg(sample, Fs=FS, MONTAGE=montage, sec=segment_length)

def process_sample_plv(args):
    """Process a single sample for PLV features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:
        log_message(f'PLV progress: {index+1}/{total_samples} samples processed')
    return yf.run_plv_seg(sample, Fs=FS, MONTAGE=montage, sec=segment_length)

def process_sample_gplv(args):
    """Process a single sample for GPLV features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:
        log_message(f'GPLV progress: {index+1}/{total_samples} samples processed')
    with HiddenPrints():
        return yf.run_gplv_seg(sample, Fs=FS, MONTAGE=montage, sec=segment_length)

def process_sample_cc(args):
    """Process a single sample for CC features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:
        log_message(f'CC progress: {index+1}/{total_samples} samples processed')
    return yf.run_cc_seg(sample, Fs=FS, MONTAGE=montage, sec=segment_length)

def process_sample_gcc(args):
    """Process a single sample for GCC features"""
    sample, index, total_samples, FS, montage, segment_length = args
    if index % 50 == 0:
        log_message(f'GCC progress: {index+1}/{total_samples} samples processed')
    with HiddenPrints():
        return yf.run_gcc_seg(sample, Fs=FS, MONTAGE=montage, sec=segment_length)

def parallel_feature_extraction_ordered(data, FS, montage, segment_length, feature_type, num_processes=None):
    """
    Extract features in parallel for a single feature type while maintaining sample order
    """
    if num_processes is None:
        num_processes = min(cpu_count(), len(data))
    
    log_message(f'Starting {feature_type} - {montage} {segment_length}s with {num_processes} processes (ordered)')
    
    # Prepare arguments for parallel processing
    args_list = [(sample, index, len(data), FS, montage, segment_length) 
                 for index, sample in enumerate(data)]
    
    # Select the appropriate processing function
    process_functions = {
        'spectral': process_sample_spectral,
        'cwt': process_sample_cwt,
        'dwt': process_sample_dwt,
        'mst': process_sample_mst,
        'sst': process_sample_sst,
        'utm': process_sample_utm,
        'plv': process_sample_plv,
        'gplv': process_sample_gplv,
        'cc': process_sample_cc,
        'gcc': process_sample_gcc
    }
    
    process_func = process_functions[feature_type]
    
    # Process in parallel while maintaining order
    start_time = time.time()
    log_message(f'Processing {len(data)} samples for {feature_type}...')
    
    try:
        with Pool(processes=num_processes) as pool:
            # Use pool.map which maintains order by default
            features = pool.map(process_func, args_list)
    except Exception as e:
        log_message(f'Error during parallel processing: {str(e)}', "ERROR")
        raise
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    features = np.array(features)
    log_message(f'Completed {feature_type} - {montage} {segment_length}s in {processing_time:.2f} seconds')
    log_message(f'Output shape: {features.shape}')
    log_message(f'Processing rate: {len(data)/processing_time:.2f} samples/second')
    
    return features

def process_single_combination(args):
    """
    Process a single feature type-montage-segment combination
    Returns (feature_type, montage, segment_length, features, success, error_msg)
    """
    data, FS, montage, segment_length, feature_type, num_processes, feature_path = args
    
    try:
        # Check if file already exists
        output_file = os.path.join(feature_path, f'{feature_type}_{montage}_{segment_length}s.npy')
        if os.path.exists(output_file):
            return (feature_type, montage, segment_length, None, 'skipped', 'File already exists')
        
        # Process features
        features = parallel_feature_extraction_ordered(
            data, FS, montage, segment_length, feature_type, num_processes
        )
        
        # Save features
        np.save(output_file, features)
        
        return (feature_type, montage, segment_length, features.shape, 'success', None)
        
    except Exception as e:
        error_msg = f'Error: {str(e)}\nTraceback: {traceback.format_exc()}'
        return (feature_type, montage, segment_length, None, 'failed', error_msg)

def parallel_combinations_processing(data, FS, feature_path, montages, segment_lengths, 
                                   feature_types, samples_processes, combinations_processes):
    """
    Process multiple combinations in parallel while maintaining order within each combination
    """
    log_message(f"Starting multi-level parallel processing:")
    log_message(f"  - {samples_processes} processes for sample processing")
    log_message(f"  - {combinations_processes} processes for combination processing")
    
    # Prepare all combination arguments
    combination_args = []
    for feature_type in feature_types:
        for montage in montages:
            for segment_length in segment_lengths:
                args = (data, FS, montage, segment_length, feature_type, 
                       samples_processes, feature_path)
                combination_args.append(args)
    
    total_combinations = len(combination_args)
    log_message(f"Total combinations to process: {total_combinations}")
    
    # Process combinations in parallel
    completed_combinations = 0
    skipped_combinations = 0
    failed_combinations = 0
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor for better control over parallel execution
    with ProcessPoolExecutor(max_workers=combinations_processes) as executor:
        # Submit all jobs
        future_to_args = {executor.submit(process_single_combination, args): args 
                         for args in combination_args}
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            _, _, montage, segment_length, feature_type, _, _ = args
            
            try:
                feature_type_result, montage_result, segment_length_result, shape, status, error_msg = future.result()
                
                if status == 'success':
                    completed_combinations += 1
                    log_message(f'✓ Completed {feature_type_result}-{montage_result}-{segment_length_result}s - Shape: {shape}')
                elif status == 'skipped':
                    skipped_combinations += 1
                    log_message(f'⊘ Skipped {feature_type_result}-{montage_result}-{segment_length_result}s - {error_msg}', "WARN")
                elif status == 'failed':
                    failed_combinations += 1
                    log_message(f'✗ Failed {feature_type_result}-{montage_result}-{segment_length_result}s', "ERROR")
                    log_message(f'Error details: {error_msg}', "ERROR")
                
                # Progress update
                total_processed = completed_combinations + skipped_combinations + failed_combinations
                progress_pct = (total_processed / total_combinations) * 100
                
                elapsed_time = time.time() - start_time
                if total_processed > 0:
                    avg_time_per_combo = elapsed_time / total_processed
                    remaining_combos = total_combinations - total_processed
                    estimated_remaining = avg_time_per_combo * remaining_combos
                    
                    log_message(f'Progress: {total_processed}/{total_combinations} ({progress_pct:.1f}%) - '
                              f'ETA: {estimated_remaining/3600:.2f}h')
                
            except Exception as e:
                failed_combinations += 1
                log_message(f'✗ Exception in {feature_type}-{montage}-{segment_length}s: {str(e)}', "ERROR")
    
    total_time = time.time() - start_time
    
    return {
        'total_combinations': total_combinations,
        'completed': completed_combinations,
        'skipped': skipped_combinations,
        'failed': failed_combinations,
        'total_time': total_time
    }

def main():
    # Start logging
    log_message("="*80)
    log_message("TUH EEG Feature Extraction - Multi-Level Parallel Processing Started")
    log_message("="*80)
    
    # Load data
    log_message("Loading data...")
    data_file = '/space/gzanardini/tuh_eeg/preprocessed/no_photostim_periods.pkl'
    
    try:
        start_load = time.time()
        data = pkl.load(open(data_file, 'rb'))
        load_time = time.time() - start_load
        log_message(f"Data loaded successfully in {load_time:.2f} seconds")
        log_message(f"Dataset contains {len(data)} samples")
    except Exception as e:
        log_message(f"Failed to load data: {str(e)}", "ERROR")
        return
    
    feature_path = '/space/gzanardini/tuh_background/'
    FS = int(250)

    # montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
    # segment_lengths = [1, 2, 5, 10, 20, 60, 120]
    # feature_types = ['spectral', 'cwt', 'dwt', 'mst', 'sst', 'utm', 'plv', 'gplv', 'cc', 'gcc']
    montages = ['Laplacian', 'CAR']
    segment_lengths = [1]
    feature_types = ['gcc']
    
    # Configure parallel processing
    total_cores = cpu_count()
    
    # Strategy: Use fewer processes for combinations to allow more processes for samples
    # This ensures better parallelization within each feature extraction
    combinations_processes = min(4, total_cores // 2)  # Max 4 combinations in parallel
    samples_processes = max(2, total_cores // combinations_processes)  # Remaining cores for samples
    
    log_message(f"System info: {total_cores} CPU cores available")
    log_message(f"Multi-level parallelization strategy:")
    log_message(f"  - Combinations processed in parallel: {combinations_processes}")
    log_message(f"  - Samples processed in parallel per combination: {samples_processes}")
    log_message(f"  - Total parallel workers: {combinations_processes * samples_processes}")
    log_message(f"Output directory: {feature_path}")
    
    total_combinations = len(montages) * len(segment_lengths) * len(feature_types)
    log_message(f"Total feature combinations to process: {total_combinations}")
    
    total_start_time = time.time()
    
    # Use multi-level parallel processing
    results = parallel_combinations_processing(
        data=data,
        FS=FS,
        feature_path=feature_path,
        montages=montages,
        segment_lengths=segment_lengths,
        feature_types=feature_types,
        samples_processes=samples_processes,
        combinations_processes=combinations_processes
    )

    # Final summary
    total_time = time.time() - total_start_time
    log_message("\n" + "="*80)
    log_message("MULTI-LEVEL PARALLEL PROCESSING COMPLETE - FINAL SUMMARY")
    log_message("="*80)
    log_message(f"Total processing time: {total_time/3600:.2f} hours")
    log_message(f"Total combinations: {results['total_combinations']}")
    log_message(f"Completed: {results['completed']}")
    log_message(f"Skipped (already existed): {results['skipped']}")
    log_message(f"Failed: {results['failed']}")
    
    if results['completed'] > 0:
        log_message(f"Average time per combination: {results['total_time']/results['completed']:.2f} seconds")
        success_rate = (results['completed'] / (results['completed'] + results['failed'])) * 100
        log_message(f"Success rate: {success_rate:.1f}%")
    
    log_message(f"Parallelization efficiency:")
    log_message(f"  - Combinations processes: {combinations_processes}")
    log_message(f"  - Samples processes per combination: {samples_processes}")
    log_message(f"  - CPU utilization: {(combinations_processes * samples_processes / total_cores)*100:.1f}%")
    log_message("="*80)

if __name__ == '__main__':
    main()
