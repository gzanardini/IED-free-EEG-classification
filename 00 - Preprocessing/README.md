# Preprocessing Directory

This directory contains the TUH Epilepsy EEG preprocessing pipeline.

## Files

- **`preprocess_tuh_epilepsy.py`** - Main preprocessing script
  - Complete pipeline for loading, filtering, and processing TUH Epilepsy dataset
  - Can be run directly or imported for individual functions
  - Run: `python preprocess_tuh_epilepsy.py`

- **`PREPROCESSING_GUIDE.md`** - Comprehensive documentation
  - Full API documentation for all functions
  - Pipeline architecture and workflow
  - Usage examples and parameter descriptions
  - Troubleshooting guide

- **`feature_extraction_tuh.py`** - Unified TUH feature extraction script
    - Replaces the separate background and photostimulation extractors
    - Can process `background`, `ips`, or `both`
    - Writes source-specific outputs to separate subdirectories

## Quick Start

```bash
# Run the complete preprocessing pipeline
python preprocess_tuh_epilepsy.py
```

This will:
1. Load the TUH Epilepsy dataset
2. Filter by required EEG channels
3. Apply standard preprocessing (filtering, resampling)
4. Extract photostimulation periods
5. Segment data and save results to `../photostimulation/data/`

### Feature Extraction

Run the unified extractor from this folder:

```bash
python feature_extraction_tuh.py --source both
```

This loads the precomputed background and photostimulation samples and saves feature arrays under separate `background/` and `ips/` subdirectories inside the configured output root.

## Key Functions

### High-level
- `main()` - Execute complete pipeline

### Dataset Operations
- `load_dataset()` - Load TUH dataset
- `select_by_channel()` - Filter by required channels
- `select_by_duration()` - Filter by minimum duration

### Preprocessing
- `preprocess()` - Apply EEG preprocessing
- `remove_calibration_signal()` - Remove initial calibration artifact

### Photostimulation Processing
- `preproc_photostim()` - Extract photostimulation periods
- `extract_photostim_periods_and_segments()` - Segment stimulation/non-stimulation periods

### Data Persistence
- `save_processed_data()` - Save intermediate results
- `save_photostim_periods()` - Save final segmented periods

## Output

**Intermediate** (`photostimulation/temp_data/`):
- `stim_samples.pkl` - Preprocessed stimulation samples
- `stim_df.csv` - Stimulation sample metadata

**Final** (`photostimulation/data/`):
- `photostim_periods.pkl` - Stimulation period arrays
- `no_photostim_periods.pkl` - Non-stimulation period arrays
- `photostim.csv` - Stimulation period metadata
- `no_photostim.csv` - Non-stimulation period metadata

## Data Flow

```
TUH Raw Data
    ↓
[load_dataset] → All recordings
    ↓
[select_by_channel] → Recordings with required EEG channels
    ↓
[preprocess] → Filtered, resampled EEG
    ↓
[select_by_duration] → Recordings ≥ 10 seconds
    ↓
[preproc_photostim] → Valid photostimulation recordings
    ↓
[extract_photostim_periods_and_segments] → Stimulation & non-stimulation periods
    ↓
[save_photostim_periods] → Final output files
```

## Configuration

Key parameters (modify in script):
- `TUEP_PATH` - Path to TUH dataset (default: `/space/gzanardini/tuh_eeg/tuh_eeg_epilepsy`)
- `target_freq` - Resampling target (default: 250 Hz)
- `tmin` - Minimum recording duration (default: 10 seconds)
- Channel montage - Defined in function parameters

## For More Information

See `PREPROCESSING_GUIDE.md` for:
- Detailed API documentation
- Parameter descriptions
- Pipeline architecture
- Usage examples
- Troubleshooting guide

See `feature_extraction_tuh.py` for:
- Unified feature extraction for background and photostimulation data
- CLI arguments for source, montages, segment lengths, and output location
- Source-specific output layout

---

*Converted from `preprocess.ipynb` notebook*
