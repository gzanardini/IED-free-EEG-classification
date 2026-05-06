# TUH Epilepsy Preprocessing Pipeline

## Overview

`preprocess_tuh_epilepsy.py` is a comprehensive preprocessing pipeline for the Temple University Hospital (TUH) Epilepsy EEG dataset. It handles raw EEG data from the TUH database, applies standard signal processing techniques, extracts photostimulation periods, and segments the data into meaningful windows for downstream analysis.

The directory also includes `feature_extraction_tuh.py`, a unified feature extraction script that combines the previous background and photostimulation extractors into a single CLI.

## Features

- **Dataset Loading**: Loads TUH Epilepsy data with flexible channel selection
- **Quality Filtering**: Filters recordings by channel availability and duration
- **EEG Preprocessing**: 
  - Channel standardization and reordering
  - Bandpass filtering (0.1 Hz - Nyquist)
  - Notch filtering at 60 Hz (powerline interference)
  - Resampling to target frequency
- **Photostimulation Processing**: 
  - Detects and extracts photostimulation periods
  - Removes calibration artifacts
  - Validates signal quality
- **Data Segmentation**: Splits recordings into photostimulation and non-photostimulation periods with temporal buffers
- **Persistent Storage**: Saves processed data as pickle and CSV files

## Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `mne-python`
- `scipy`
- Custom utilities: `utils.TUEP`, `utils.photostim`

## Usage

### Basic Usage (Command Line)

```bash
python preprocess_tuh_epilepsy.py
```

This will run the complete pipeline with default settings.

### Programmatic Usage

```python
from preprocess_tuh_epilepsy import main, preprocess, preproc_photostim

# Run complete pipeline
photostim_periods, no_photostim_periods, wins_df, no_ph_df = main(
    tuep_path='/space/gzanardini/tuh_eeg/tuh_eeg_epilepsy'
)

# Or use individual functions
from preprocess_tuh_epilepsy import load_dataset, select_by_channel, select_by_duration

dataset = load_dataset('/path/to/tuh_data')
dataset = select_by_channel(dataset, ['FP1', 'CZ', 'O2', 'PHOTIC PH'])
dataset = select_by_duration(dataset, tmin=10)
```

## Pipeline Architecture

### 1. Data Loading (`load_dataset`)

Loads the TUH Epilepsy dataset with lazy loading disabled for preprocessing efficiency.

**Parameters:**
- `path` (str): Path to TUH Epilepsy dataset root directory

**Returns:**
- `TUHEpilepsy` object with all recordings loaded

### 2. Filtering

#### Channel-based Filtering (`select_by_channel`)

Keeps only recordings that have all required EEG channels. The standard montage includes:
- 19 EEG channels: FP1, F3, C3, P3, F7, T3, T5, O1, FZ, CZ, PZ, FP2, F4, C4, P4, F8, T4, T6, O2
- 1 Stimulation marker channel: PHOTIC PH

**Parameters:**
- `dataset` (TUHEpilepsy): Input dataset
- `channels` (list): Required channel names (case-insensitive)

**Returns:**
- `TUHEpilepsy` object with filtered recordings

#### Duration-based Filtering (`select_by_duration`)

Keeps only recordings longer than a minimum duration threshold (default: 10 seconds).

**Parameters:**
- `dataset` (TUHEpilepsy): Input dataset
- `tmin` (float): Minimum duration in seconds (default: 10)

**Returns:**
- `TUHEpilepsy` object with duration-filtered recordings

### 3. EEG Preprocessing (`preprocess`)

Applies standard EEG signal processing in the following order:

**Steps:**
1. Rename channels to uppercase for consistency
2. Handle missing PHOTIC PH channel (adds zero-signal channel if absent)
3. Pick and reorder channels to standard montage
4. Load raw data into memory
5. **Bandpass Filter**: 0.1 Hz (high-pass) to Nyquist frequency (low-pass)
   - Prevents DC drift and very high-frequency noise
   - Uses linear-phase filter with controlled bandwidth
6. **Notch Filter**: 60 Hz (powerline interference)
   - Removes electrical noise from AC mains
7. **Resampling**: Optional resampling to target frequency (typically 250 Hz)
   - Reduces data size while preserving relevant information

**Parameters:**
- `in_dataset` (TUHEpilepsy): Input dataset
- `target_freq` (int, optional): Target sampling frequency in Hz (default: None)
- `photic_ph` (bool, optional): Include PHOTIC PH channel (default: True)

**Returns:**
- `TUHEpilepsy` object with preprocessed data

**Important Notes:**
- Data is loaded into memory during processing
- All modifications are applied in-place
- Missing channels trigger a warning and skip that recording

### 4. Photostimulation Processing (`preproc_photostim`)

Extracts recordings with valid photostimulation signals and applies specialized preprocessing.

**Quality Criteria:**
- Non-zero photostimulation signal in PHOTIC PH channel
- Recording duration ≥ 30 seconds after calibration removal
- Not in the manually identified problematic samples list

**Preprocessing Steps:**
1. Detect and remove calibration signal from beginning
2. Validate photostimulation signal presence
3. Apply notch filtering (60 Hz)
4. Resample to target frequency

**Manually Excluded Samples:**
```python
SAMPLES_TO_DISCARD = {
    "aaaaabju": {"session": 1, "segment": 0},
    "aaaaaflb": {"session": 1, "segment": 0},
    "aaaaagxr": {"session": 1, "segment": 1},
    "aaaaajgn": {"session": 1, "segment": 0},
    "aaaaajrh": {"session": 2, "segment": 0}
}
```

These samples have PHOTIC PH channels but lack clear stimulation patterns.

**Parameters:**
- `dataset` (TUHEpilepsy): Input dataset
- `fs_target` (int, optional): Target sampling frequency in Hz (default: 250)

**Returns:**
- `pd.DataFrame`: Metadata for valid photostimulation samples
- `list`: Processed MNE Raw objects with photostimulation signals

### 5. Photostimulation Period Extraction (`extract_photostim_periods_and_segments`)

For each valid stimulation sample:
1. Identifies start and end of photostimulation signal (non-zero samples in channel 19)
2. Adds 1-second temporal buffer before and after
3. Extracts corresponding non-stimulation segments
4. Handles edge case: Sample 10 contains two distinct stimulation periods (splits automatically)

**Parameters:**
- `stim_samples` (list): Raw objects with stimulation signals
- `stim_df` (pd.DataFrame): Stimulation metadata

**Returns:**
- `list`: Photostimulation periods (numpy arrays)
- `list`: Non-photostimulation periods (numpy arrays)
- `pd.DataFrame`: Metadata for photostimulation periods
- `pd.DataFrame`: Metadata for non-photostimulation periods

### 6. Data Persistence

#### Save Intermediate Results (`save_processed_data`)

Saves extracted stimulation samples before segmentation.

**Output:**
- `stim_samples.pkl`: Pickled list of Raw objects
- `stim_df.csv`: Metadata CSV

#### Save Final Results (`save_photostim_periods`)

Saves final segmented periods.

**Output:**
- `photostim_periods.pkl`: Stimulation period numpy arrays
- `no_photostim_periods.pkl`: Non-stimulation period numpy arrays
- `photostim.csv`: Stimulation metadata
- `no_photostim.csv`: Non-stimulation metadata

## Feature Extraction

`feature_extraction_tuh.py` is the consolidated feature extraction entry point for the TUH preprocessing workspace.

### Supported Sources

- `background` - loads `no_photostim_periods.pkl`
- `ips` - loads `stim_samples.pkl`
- `both` - processes both sources sequentially

### Extraction Families

The script keeps the existing feature families:

- `spectral`
- `cwt`
- `dwt`
- `mst`
- `sst`
- `utm`
- `plv`
- `gplv`
- `cc`
- `gcc`

### Example Usage

```bash
python feature_extraction_tuh.py --source both
python feature_extraction_tuh.py --source background --skip-existing
python feature_extraction_tuh.py --source ips --output-root /space/gzanardini/tuh_background
```

### Output Layout

Feature arrays are written to source-specific subdirectories so background and photostimulation runs do not overwrite each other:

- `<output-root>/background/`
- `<output-root>/ips/`

File names stay consistent with the previous scripts, for example `spectral_CAR_10s.npy`.

## Main Pipeline Execution (`main`)

The `main()` function orchestrates the complete pipeline:

```python
def main(
    tuep_path: str = '/space/gzanardini/tuh_eeg/tuh_eeg_epilepsy',
    save_intermediate: bool = True,
    save_final: bool = True
) -> tuple
```

**Step-by-step execution:**
1. Load dataset
2. Filter by required channels
3. Apply EEG preprocessing
4. Filter by minimum duration
5. Extract photostimulation periods
6. Segment into stimulation/non-stimulation periods
7. Save results

**Parameters:**
- `tuep_path` (str): Path to TUH Epilepsy dataset root
- `save_intermediate` (bool): Save intermediate stimulation samples
- `save_final` (bool): Save final segmented periods

**Returns:**
- `photostim_periods` (list): Segmented stimulation periods
- `no_photostim_periods` (list): Segmented non-stimulation periods
- `wins_df` (pd.DataFrame): Stimulation period metadata
- `no_ph_df` (pd.DataFrame): Non-stimulation period metadata

## Output Files

### Intermediate Output (`photostimulation/temp_data/`)
- `stim_samples.pkl`: Preprocessed Raw objects with photostimulation signal
- `stim_df.csv`: Sample metadata

### Final Output (`photostimulation/data/`)
- `photostim_periods.pkl`: Numpy arrays of stimulation periods
- `no_photostim_periods.pkl`: Numpy arrays of non-stimulation periods
- `photostim.csv`: Stimulation period metadata
- `no_photostim.csv`: Non-stimulation period metadata

## Metadata Format

Both `photostim.csv` and `no_photostim.csv` contain the following columns (from TUH dataset):
- `subject`: Subject ID
- `session`: Session number
- `segment`: Recording segment number
- `epilepsy`: Epilepsy status (label)
- Additional TUH metadata columns

## Calibration Signal Removal

The `remove_calibration_signal()` function removes an initial calibration artifact that appears at the beginning of some recordings. The calibration signal is identified by high amplitude values in the PHOTIC PH channel.

**Parameters:**
- `eeg` (np.ndarray): EEG data (n_channels × n_samples)
- `amplitude` (float): Expected calibration signal amplitude
- `threshold` (float): Tolerance around expected amplitude

## Edge Cases

### Sample 10 - Multiple Stimulation Periods

Sample 10 in the dataset contains two distinct photostimulation periods within a single recording. The pipeline automatically:
1. Detects the long recording
2. Identifies the midpoint between the two stimulation periods
3. Creates separate segments with 250-sample (1-second) temporal overlap
4. Updates the periods list and metadata accordingly

## Performance Considerations

- **Memory**: Data is loaded into RAM during preprocessing. Large datasets may require significant memory.
- **Parallel Processing**: Uses 8 parallel jobs for filtering and resampling operations
- **Processing Time**: Depends on dataset size and computing resources

## Troubleshooting

### Missing Channels
If warnings appear about missing channels, verify:
1. Channel names in raw data match expected montage
2. Data was recorded with standard electrode placement
3. MNE channel renaming is applied correctly

### Photostimulation Signal Validation
If few samples pass photostimulation processing:
1. Check `SAMPLES_TO_DISCARD` for excluded problematic samples
2. Verify calibration signal removal isn't removing valid signal
3. Review minimum duration requirement (30 seconds)

### File I/O Errors
Ensure output directories exist and have write permissions:
```bash
mkdir -p photostimulation/data
mkdir -p photostimulation/temp_data
```

## Future Improvements

- Add command-line argument parsing for configuration
- Implement parallel processing for multiple datasets
- Add visualization functions for signal inspection
- Create unit tests for preprocessing functions
- Support for additional photostimulation protocols

## References

- MNE-Python documentation: https://mne.tools/
- TUH EEG Dataset: https://isip.piconepress.com/projects/tuh_eeg/html/overview.html
- EEG Signal Processing standards: https://doi.org/10.1016/j.neuroimage.2015.02.033

## Author Notes

This pipeline was developed iteratively based on exploration in `preprocess.ipynb`. Key design decisions:
- 250 Hz target frequency chosen to balance information preservation with file size
- 0.1 Hz high-pass filter ensures signal stability over long recordings
- 1-second temporal buffer around stimulation periods provides context
- Manual sample exclusion list prevents poor-quality photostimulation signals from contaminating analysis
