"""
TUH Epilepsy Dataset Preprocessing Pipeline

This script provides a comprehensive preprocessing pipeline for the TUH Epilepsy dataset.
It includes:
- Dataset loading and filtering by duration and channels
- EEG signal preprocessing (filtering, resampling, notch filtering)
- Photostimulation signal extraction and processing
- Data persistence with CSV and pickle formats

The pipeline processes raw EEG recordings, extracts photostimulation periods,
and segments them for further analysis while maintaining metadata for each sample.

Example:
    python preprocess_tuh_epilepsy.py --config config.yaml

Author: EEG Analysis Team
Last Modified: 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from copy import deepcopy, copy

from utils.TUEP import TUHEpilepsy
from utils.photostim import extract_photostim_chunks
import mne


# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def setup_pandas_display():
    """Configure pandas display options for better readability of large datasets."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    np.set_printoptions(threshold=1000, edgeitems=20, linewidth=1000)


def load_dataset(path: str) -> TUHEpilepsy:
    """
    Load the TUH Epilepsy dataset from disk.
    
    Parameters
    ----------
    path : str
        Path to the TUH Epilepsy dataset root directory
        
    Returns
    -------
    TUHEpilepsy
        Loaded dataset object with lazy loading enabled (preload=False)
    """
    dataset = TUHEpilepsy(
        path=path,
        set_montage=False,
        rename_channels=True,
        target_name='epilepsy',
        n_jobs=1,
        preload=False
    )
    return dataset


# ============================================================================
# DATA FILTERING FUNCTIONS
# ============================================================================

def select_by_duration(dataset: TUHEpilepsy, tmin: float = 10) -> TUHEpilepsy:
    """
    Filter dataset to keep only recordings with duration >= tmin seconds.
    
    Parameters
    ----------
    dataset : TUHEpilepsy
        Input dataset
    tmin : float, optional
        Minimum duration in seconds (default: 10)
        
    Returns
    -------
    TUHEpilepsy
        Filtered dataset containing only long enough recordings
    """
    new_dataset = deepcopy(dataset)
    new_dataset.datasets = [
        sample for sample in dataset.datasets
        if sample.raw.n_times / sample.raw.info['sfreq'] >= tmin
    ]
    return new_dataset


def select_by_channel(dataset: TUHEpilepsy, channels: list) -> TUHEpilepsy:
    """
    Filter dataset to keep only recordings that have all specified channels.
    
    Parameters
    ----------
    dataset : TUHEpilepsy
        Input dataset
    channels : list
        List of required channel names (case-insensitive matching)
        
    Returns
    -------
    TUHEpilepsy
        Filtered dataset containing only recordings with all required channels
    """
    new_dataset = deepcopy(dataset)
    new_dataset.datasets = [
        sample for sample in dataset.datasets
        if set(channels).issubset(ch_name.upper() for ch_name in sample.raw.ch_names)
    ]
    return new_dataset


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess(
    in_dataset: TUHEpilepsy,
    target_freq: int = None,
    photic_ph: bool = True
) -> TUHEpilepsy:
    """
    Apply standard EEG preprocessing to the dataset.
    
    Preprocessing steps:
    1. Rename channels to uppercase
    2. Handle PHOTIC PH (photostimulation) channel - add if missing
    3. Pick and reorder channels to standard montage
    4. Load data into memory
    5. Apply bandpass filter (0.1 Hz to Nyquist frequency)
    6. Apply notch filter (60 Hz for powerline interference)
    7. Resample to target frequency if specified
    
    Parameters
    ----------
    in_dataset : TUHEpilepsy
        Input dataset to preprocess
    target_freq : int, optional
        Target sampling frequency in Hz. If None, no resampling (default: None)
    photic_ph : bool, optional
        Whether to include PHOTIC PH (photostimulation) channel (default: True)
        
    Returns
    -------
    TUHEpilepsy
        Preprocessed dataset with modified raw data
    """
    dataset = deepcopy(in_dataset)
    
    if photic_ph:
        channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
                    'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'PHOTIC PH']
    else:
        channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
                    'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
    
    for i, sample in enumerate(dataset.datasets):
        # Check if required channels are present (excluding PHOTIC PH)
        if not set(channels[:-1]).issubset(ch_name.upper() for ch_name in sample.raw.ch_names):
            print(f"Channels missing in sample {i}")
            print(f"Missing: {set(channels[:-1]) - set(ch.upper() for ch in sample.raw.ch_names)}")
            continue
        
        # Standardize channel names to uppercase
        sample.raw.rename_channels(lambda x: x.upper(), verbose=False)
        
        # Handle PHOTIC PH channel
        if 'PHOTIC PH' in sample.raw.ch_names and photic_ph:
            sample.raw.set_channel_types({'PHOTIC PH': 'stim'}, on_unit_change="ignore")
        elif 'PHOTIC PH' not in sample.raw.ch_names and photic_ph:
            sample.raw.load_data()
            sample.raw.add_channels(
                [mne.io.RawArray(
                    np.zeros((1, sample.raw.n_times)),
                    mne.create_info(['PHOTIC PH'], sample.raw.info['sfreq'], ['stim']),
                    verbose=False
                )],
                force_update_info=True
            )
            print(f"Added PHOTIC PH channel to sample {i}")
        
        # Select and order channels
        sample.raw.pick(channels)
        sample.raw.reorder_channels(channels)
        sample.raw.load_data()
        
        # Filtering
        sample.raw.filter(
            0.1, (sample.raw.info['sfreq'] / 2) - 0.1,
            verbose=False, n_jobs=8, l_trans_bandwidth=0.1
        )
        sample.raw.notch_filter(60, verbose=False, n_jobs=8)
        
        # Resampling
        if sample.raw.info['sfreq'] != target_freq and target_freq is not None:
            sample.raw.resample(target_freq, n_jobs=8)
            print(f"Resampled sample {i} to {target_freq} Hz")
    
    return dataset


def remove_calibration_signal(eeg: np.ndarray, amplitude: float, threshold: float = 1) -> np.ndarray:
    """
    Remove calibration signal from the beginning of EEG recording.
    
    Scans the anchor channel (typically channel 19 - PHOTIC PH) and trims
    the recording to start after the calibration signal ends.
    
    Parameters
    ----------
    eeg : np.ndarray
        EEG data array of shape (n_channels, n_samples)
    amplitude : float
        Expected calibration signal amplitude
    threshold : float, optional
        Tolerance threshold around amplitude (default: 1)
        
    Returns
    -------
    np.ndarray
        EEG data with calibration signal removed from the beginning
    """
    for t in range(eeg.shape[1]):
        if np.abs(eeg[19, t]) < amplitude - threshold:
            start = t
            break
    else:
        start = 0
    
    return eeg[:, start:]


# ============================================================================
# PHOTOSTIMULATION PROCESSING
# ============================================================================

# Manually identified samples with problematic photostimulation signals
# Despite having PHOTIC PH channel, these samples lack clear photostimulation patterns
SAMPLES_TO_DISCARD = {
    "aaaaabju": {"session": 1, "segment": 0},
    "aaaaaflb": {"session": 1, "segment": 0},
    "aaaaagxr": {"session": 1, "segment": 1},
    "aaaaajgn": {"session": 1, "segment": 0},
    "aaaaajrh": {"session": 2, "segment": 0}
}


def preproc_photostim(dataset: TUHEpilepsy, fs_target: int = 250):
    """
    Extract and preprocess photostimulation periods from dataset.
    
    This function:
    1. Filters for recordings with valid photostimulation signal
    2. Removes calibration artifacts
    3. Applies filtering and resampling
    4. Validates signal presence and duration
    5. Returns processed samples and metadata
    
    Parameters
    ----------
    dataset : TUHEpilepsy
        Input dataset containing raw recordings
    fs_target : int, optional
        Target sampling frequency in Hz (default: 250)
        
    Returns
    -------
    tuple
        - pd.DataFrame: Metadata for samples with photostimulation
        - list: Processed MNE Raw objects with photostimulation signals
    """
    channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
                'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'PHOTIC PH']
    
    channels_dict = {ch: 'eeg' for ch in channels[:-1]}
    channels_dict['PHOTIC PH'] = 'stim'
    
    new_df = pd.DataFrame(index=range(len(dataset.datasets)), columns=dataset.description.columns)
    samples_with_stim = []
    ctr = 0
    
    for i, recording in enumerate(dataset.datasets):
        # Skip manually identified problematic samples
        if recording.description['subject'] in SAMPLES_TO_DISCARD:
            sample_meta = SAMPLES_TO_DISCARD[recording.description['subject']]
            if (recording.description['session'] == sample_meta['session'] and
                recording.description['segment'] == sample_meta['segment']):
                print(f"Discarding {recording.description['subject']} - "
                      f"Session {recording.description['session']} - "
                      f"Segment {recording.description['segment']}")
                continue
        
        # Process only if photostimulation channel exists
        if 'PHOTIC PH' in recording.raw.ch_names:
            recording.raw.pick(channels)
            recording.raw.reorder_channels(channels)
            recording.raw.set_channel_types(channels_dict, on_unit_change="warn")
            
            # Remove calibration signal and convert to microvolts
            temp = remove_calibration_signal(recording.raw.get_data(), 100)
            temp[19] = temp[19] * 1e-6
            
            # Validate: non-zero photostim signal and sufficient duration
            if not np.all(temp[19] == 0) and temp.shape[1] / recording.raw.info['sfreq'] >= 30:
                recording.raw = mne.io.RawArray(temp, recording.raw.info, verbose=False)
                
                # Apply filters
                recording.raw.notch_filter(60, verbose=False, n_jobs=8)
                
                # Resample if needed
                if recording.raw.info['sfreq'] != fs_target and fs_target is not None:
                    recording.raw.resample(fs_target, n_jobs=8)
                    print(f"Resampled sample {i} to {fs_target} Hz")
                
                samples_with_stim.append(recording.raw)
                new_df.loc[ctr] = dataset.description.loc[i]
                ctr += 1
    
    new_df = new_df.dropna()
    
    return new_df, samples_with_stim


def extract_photostim_periods_and_segments(
    stim_samples: list,
    stim_df: pd.DataFrame
) -> tuple:
    """
    Extract photostimulation periods and non-photostimulation segments.
    
    For each recording:
    - Identifies start/end of photostimulation signal (channel 19)
    - Extracts 1-second buffer before/after signal
    - Creates corresponding non-photostimulation segments
    - Handles edge case of recording with multiple stimulation periods
    
    Parameters
    ----------
    stim_samples : list
        List of MNE Raw objects with photostimulation signals
    stim_df : pd.DataFrame
        Metadata DataFrame for stimulation samples
        
    Returns
    -------
    tuple
        - list: Photostimulation periods (numpy arrays)
        - list: Non-photostimulation periods (numpy arrays)
        - pd.DataFrame: Metadata for photostimulation periods
        - pd.DataFrame: Metadata for non-photostimulation periods
    """
    photostim_periods = []
    no_photostim_periods = []
    intervals = []
    
    wins_df = pd.DataFrame(index=range(len(stim_samples) + 1), columns=stim_df.columns)
    no_ph_df = pd.DataFrame(index=range(len(stim_samples) + 1), columns=stim_df.columns)
    
    for i, edf in enumerate(stim_samples):
        sample = edf.get_data()
        print(f'Processing sample {i}: {sample.shape[1] / 250:.2f} seconds')
        print(f'Channel order: {edf.ch_names}')
        
        # Find start and end of photostimulation signal
        start = None
        end = None
        
        for t in range(sample.shape[1]):
            if np.abs(sample[19, t]) > 0:
                start = t
                break
        
        for t in range(sample.shape[1] - 1, 0, -1):
            if np.abs(sample[19, t]) > 0:
                end = t
                break
        
        if start is None or end is None:
            continue
        
        # Add 1-second buffer
        start -= 250  # 1 second at 250 Hz
        end += 250
        
        temp = sample[:, start:end]
        print(f'Extracted segment - Start: {start}, End: {end}, '
              f'Length: {temp.shape[1] / 250:.2f} seconds')
        intervals.append((start, end))
        
        photostim_periods.append(temp)
        wins_df.loc[i] = stim_df.loc[i]
        del temp
        
        # Extract non-stimulation segments
        temp1 = sample[:, :start]
        temp2 = sample[:, end:]
        temp = np.concatenate((temp1, temp2), axis=1)
        
        no_photostim_periods.append(temp)
        no_ph_df.loc[i] = stim_df.loc[i]
        
        del temp1, temp2, temp
    
    # Handle edge case: sample 10 has two different stimulation periods
    if len(photostim_periods) > 10:
        temp_df = wins_df.copy().iloc[:11]
        temp_df2 = wins_df.copy().iloc[10:]
        wins_df = pd.concat([temp_df, temp_df2], ignore_index=True, axis=0)
        wins_df.dropna(inplace=True)
        
        # Split the long sample
        long_sample = copy(photostim_periods[10])
        
        start = None
        end = None
        for t in range(long_sample.shape[1]):
            if np.abs(long_sample[19, t]) > 0:
                start = t
                break
        
        for t in range(long_sample.shape[1] - 1, 0, -1):
            if np.abs(long_sample[19, t]) > 0:
                end = t
                break
        
        midpoint = (start + end) // 2
        first_part = copy(long_sample[:, start:midpoint + 250])
        second_part = copy(long_sample[:, midpoint - 250:end + 250])
        
        print(f'Splitting long sample - Full: [{start}, {end}], Midpoint: {midpoint}')
        
        # Process first part
        for t in range(first_part.shape[1]):
            if np.abs(first_part[19, t]) > 0:
                start_first = t
                break
        
        for t in range(first_part.shape[1] - 1, 0, -1):
            if np.abs(first_part[19, t]) > 0:
                end_first = t
                break
        
        end_first += 250
        first_part = copy(first_part[:, start_first:end_first])
        print(f'First part: Start {start_first}, End {end_first}, '
              f'Length: {first_part.shape[1]} samples')
        
        # Process second part
        for t in range(second_part.shape[1]):
            if np.abs(second_part[19, t]) > 0:
                start_second = t
                break
        
        for t in range(second_part.shape[1] - 1, 0, -1):
            if np.abs(second_part[19, t]) > 0:
                end_second = t
                break
        
        second_part = copy(second_part[:, start_second - 250:end_second + 250])
        
        photostim_periods[10] = first_part
        photostim_periods.insert(11, second_part)
    
    no_ph_df.dropna(inplace=True)
    
    print(f'Photostimulation periods: {len(photostim_periods)}')
    print(f'Non-photostimulation periods: {len(no_photostim_periods)}')
    print(f'Metadata records (stimulation): {len(wins_df)}')
    print(f'Metadata records (non-stimulation): {len(no_ph_df)}')
    
    return photostim_periods, no_photostim_periods, wins_df, no_ph_df


# ============================================================================
# DATA PERSISTENCE
# ============================================================================

def save_processed_data(
    stim_samples: list,
    stim_df: pd.DataFrame,
    output_dir: str = 'photostimulation/temp_data/'
) -> None:
    """
    Save processed stimulation samples and metadata.
    
    Parameters
    ----------
    stim_samples : list
        List of processed Raw objects with stimulation signals
    stim_df : pd.DataFrame
        Metadata DataFrame for stimulation samples
    output_dir : str, optional
        Directory to save outputs (default: 'photostimulation/temp_data/')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'stim_samples.pkl'), 'wb') as f:
        pickle.dump(stim_samples, f)
    
    stim_df.to_csv(os.path.join(output_dir, 'stim_df.csv'), index=False)
    
    print(f"Saved {len(stim_samples)} stimulation samples to {output_dir}")


def save_photostim_periods(
    photostim_periods: list,
    no_photostim_periods: list,
    wins_df: pd.DataFrame,
    no_ph_df: pd.DataFrame,
    output_dir: str = 'photostimulation/data/'
) -> None:
    """
    Save extracted photostimulation and non-photostimulation periods.
    
    Parameters
    ----------
    photostim_periods : list
        List of photostimulation period arrays
    no_photostim_periods : list
        List of non-photostimulation period arrays
    wins_df : pd.DataFrame
        Metadata for photostimulation periods
    no_ph_df : pd.DataFrame
        Metadata for non-photostimulation periods
    output_dir : str, optional
        Directory to save outputs (default: 'photostimulation/data/')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'photostim_periods.pkl'), 'wb') as f:
        pickle.dump(photostim_periods, f)
    
    with open(os.path.join(output_dir, 'no_photostim_periods.pkl'), 'wb') as f:
        pickle.dump(no_photostim_periods, f)
    
    wins_df.to_csv(os.path.join(output_dir, 'photostim.csv'), index=False)
    no_ph_df.to_csv(os.path.join(output_dir, 'no_photostim.csv'), index=False)
    
    print(f"Saved {len(photostim_periods)} photostimulation periods to {output_dir}")
    print(f"Saved {len(no_photostim_periods)} non-photostimulation periods to {output_dir}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(
    tuep_path: str = '/space/gzanardini/tuh_eeg/tuh_eeg_epilepsy',
    save_intermediate: bool = True,
    save_final: bool = True
) -> tuple:
    """
    Execute the complete preprocessing pipeline.
    
    Pipeline steps:
    1. Load TUH Epilepsy dataset
    2. Filter by channels and duration
    3. Apply general preprocessing
    4. Extract photostimulation periods
    5. Segment data into stimulation and non-stimulation periods
    6. Save results
    
    Parameters
    ----------
    tuep_path : str, optional
        Path to TUH Epilepsy dataset (default: '/space/gzanardini/tuh_eeg/tuh_eeg_epilepsy')
    save_intermediate : bool, optional
        Save intermediate stimulation samples (default: True)
    save_final : bool, optional
        Save final segmented data (default: True)
        
    Returns
    -------
    tuple
        - list: Photostimulation periods
        - list: Non-photostimulation periods
        - pd.DataFrame: Stimulation metadata
        - pd.DataFrame: Non-stimulation metadata
    """
    print("=" * 70)
    print("TUH EPILEPSY PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Setup
    setup_pandas_display()
    
    # Step 1: Load dataset
    print("\n[Step 1/5] Loading TUH Epilepsy dataset...")
    tuep = load_dataset(tuep_path)
    print(f"Loaded {len(tuep.datasets)} recordings")
    
    # Step 2: Filter by channels
    channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ',
                'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'PHOTIC PH']
    print(f"\n[Step 2/5] Filtering by required channels ({len(channels)} channels)...")
    tuep_filtered = select_by_channel(tuep, channels)
    print(f"Kept {len(tuep_filtered.datasets)} recordings with all required channels")
    
    # Step 3: Apply preprocessing
    print("\n[Step 3/5] Applying EEG preprocessing...")
    tuep_preprocessed = preprocess(tuep_filtered, target_freq=250, photic_ph=True)
    print("Preprocessing complete")
    
    # Step 4: Filter by duration
    print(f"\n[Step 4/5] Filtering by duration (minimum 10 seconds)...")
    tuep_final = select_by_duration(tuep_preprocessed, tmin=10)
    print(f"Kept {len(tuep_final.datasets)} recordings with sufficient duration")
    
    # Step 5a: Extract photostimulation periods
    print("\n[Step 5a/5] Extracting photostimulation periods...")
    stim_df, stim_samples = preproc_photostim(tuep_final, fs_target=250)
    print(f"Found {len(stim_samples)} recordings with valid photostimulation signals")
    
    if save_intermediate:
        save_processed_data(stim_samples, stim_df)
    
    # Step 5b: Segment into stimulation and non-stimulation periods
    print("\n[Step 5b/5] Segmenting into stimulation/non-stimulation periods...")
    photostim_periods, no_photostim_periods, wins_df, no_ph_df = \
        extract_photostim_periods_and_segments(stim_samples, stim_df)
    
    if save_final:
        save_photostim_periods(photostim_periods, no_photostim_periods, wins_df, no_ph_df)
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    
    return photostim_periods, no_photostim_periods, wins_df, no_ph_df


if __name__ == '__main__':
    photostim_periods, no_photostim_periods, wins_df, no_ph_df = main()
