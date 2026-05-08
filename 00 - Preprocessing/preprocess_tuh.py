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
    python 000 - preprocess_tuh_epilepsy.py
"""

from utils.preprocessing import (
    load_dataset,
    select_by_channel,
    preprocess,
    select_by_duration,
    preproc_photostim,
    extract_photostim_periods_and_segments,
    save_processed_data,
    save_photostim_periods
)

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

