# Classification of IED-free EEG responses for epilepsy diagnosis

This repository contains the code for the EMBC2026 paper "Classification of IED-free EEG Responses for
Assisted Epilepsy Diagnosis" available [here](https://arxiv.org/abs/2605.22858).

It includes preprocessing, feature extraction, training, evaluation, and explainability workflows for TUH and EMC datasets.

TUH data is available under the [TUH EEG Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) and EMC data cannot be made available to the public due to privacy restrictions. For this reason the corresponding EMC preprocessing scripts are not included. The respective training scripts are provided, as their structure is similar to the TUH one but do not contain any sensitive information. 

The code is structured to allow running the full pipeline on either dataset (potentially extendable to other datasets), with configurable paths and parameters.

## Project Structure

- `00 - Preprocessing/`: TUH preprocessing and feature extraction entry points.
- `01 - Training/`: EMC and TUH training scripts (single-set and ensemble).
- `02 - Evaluation, visualization and explainability/`: result aggregation, plotting, and interpretability notebooks.
- `utils/`: shared utilities for preprocessing, feature extraction, training, and plotting.

## Dependencies

Core packages used by the scripts and notebooks:

- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- `mne`
- `xgboost`, `wandb`, `joblib`, `tqdm`
- `statsmodels`, `shap`
- `PyWavelets` (imported as `pywt`), `stockwell`, `bctpy` (imported as `bct`)
- `torch`
- `cupy` (install a CUDA-specific build, for example `cupy-cuda11x`)

Install the pinned list from [requirements.txt](requirements.txt) and update the CuPy entry to match your CUDA version if needed.

```bash
python -m pip install -r requirements.txt
```

## Quickstart - Usage examples

1. Preprocess TUH raw data and extract photostimulation periods:

```bash
python "00 - Preprocessing/preprocess_tuh.py"
```

2. Extract features for background + IPS samples:

```bash
python "00 - Preprocessing/extract_features_tuh.py" --source both --skip-existing
```

1. Train an ensemble model:

```bash
python "01 - Training/tuh/tuh_ensemble_background.py"
```

4. Run evaluation notebooks (example):

```bash
jupyter notebook "02 - Evaluation, visualization and explainability/Ensemble results plotting.ipynb"
```

To avoid processes interruptions, it's reccomended to run them as background jobs using `nohup` or similar tools, for example:

```bash
nohup python "01 - Training/tuh/tuh_ensemble_background.py" > tuh_ensemble_background.log 2>&1 &
```	


## Preprocessing

- `00 - Preprocessing/preprocess_tuh.py`
	- End-to-end TUH preprocessing pipeline.
	- Filters by required channels/duration, preprocesses EEG, extracts photostimulation periods, and saves segmented outputs.

- `00 - Preprocessing/extract_features_tuh.py`
	- Unified feature extraction runner for `background`, `ips`, or `both` sources.
	- Configurable montages, segment lengths, and feature families; can skip existing outputs.

## Training (EMC and TUH)

Naming conventions:

- `ss` = single-set training (`run_single_set_experiment`)
- `ensemble` = ensemble feature-combination training (`run_ensemble_experiment`)
- `ips` = intermittent photic stimulation samples
- `bg` = background/resting-state samples
- `hv` = hyperventilation samples
- `noieds` = only for TUH, excludes specified subjects with IEDs from training/evaluation

### EMC scripts

- `01 - Training/emc/emc_ss_background.py`: single-set sweep on EMC background dataset.
- `01 - Training/emc/emc_ss_hypervent.py`: single-set sweep on EMC hyperventilation dataset.
- `01 - Training/emc/emc_ss_ips.py`: single-set sweep on EMC IPS dataset.
- `01 - Training/emc/emc_ensemble_background.py`: ensemble training on EMC background dataset.
- `01 - Training/emc/emc_ensemble_ips.py`: ensemble training on EMC IPS dataset.
- `01 - Training/emc/emc_ensemble_hv_all.py`: ensemble training on all EMC hyperventilation subjects.
- `01 - Training/emc/emc_ensemble_hv_responders.py`: ensemble training on HV responder subjects.
- `01 - Training/emc/emc_ensemble_hv_nonresponders.py`: ensemble training on HV non-responder subjects.
- `01 - Training/emc/emc_ensemble_ips_bg.py`: dual-source ensemble combining IPS + background feature sources.
- `01 - Training/emc/emc_ensemble_ips_bg_hv.py`: triple-source ensemble combining IPS + background + HV sources.

### TUH scripts

- `01 - Training/tuh/tuh_ss_bg.py`: single-set sweep on TUH background dataset.
- `01 - Training/tuh/tuh_ss_bg_noieds.py`: TUH background single-set sweep excluding specified subjects.
- `01 - Training/tuh/tuh_ss_ips.py`: single-set sweep on TUH IPS/whole dataset.
- `01 - Training/tuh/tuh_ss_ips_noieds.py`: TUH IPS single-set sweep excluding specified subjects.
- `01 - Training/tuh/tuh_ensemble_background.py`: ensemble training on TUH background split.
- `01 - Training/tuh/tuh_ensemble_background_noieds.py`: TUH background ensemble excluding specified subjects.
- `01 - Training/tuh/tuh_ensemble_ips.py`: ensemble training on TUH IPS/whole split.
- `01 - Training/tuh/tuh_ensemble_ips_noieds.py`: TUH IPS ensemble excluding specified subjects.
- `01 - Training/tuh/tuh_ensemble_ips_bg.py`: dual-source ensemble combining TUH IPS + background sources.
- `01 - Training/tuh/tuh_ensemble_ips_bg_noieds.py`: dual-source TUH IPS + background ensemble excluding specified subjects.

## Evaluation and Explainability

- `02 - Evaluation, visualization and explainability/Ensemble results plotting.ipynb`
	- Aggregates ensemble results, computes metrics (BAC, F1, AUC, AUPRC), and plots best combinations.
- `02 - Evaluation, visualization and explainability/Single-sets results plotting.ipynb`
	- Aggregates single-set results across feature families and produces ranking plots.
- `02 - Evaluation, visualization and explainability/Explainability/ale_emc.ipynb`
	- ALE analysis with LOSO folds for EMC data.
- `02 - Evaluation, visualization and explainability/Explainability/shap_emc.ipynb`
	- SHAP summaries for EMC features (configured for PLV in the notebook).
- `02 - Evaluation, visualization and explainability/Explainability/shap_tuh.ipynb`
	- SHAP summaries for TUH features (configured for UTM in the notebook).
- `02 - Evaluation, visualization and explainability/Explainability/shap_tuh_iedfree.ipynb`
	- SHAP summaries for TUH IED-free analysis (includes subject filtering).

## Notes

- Most scripts use absolute data/log paths (for example under `/space/gzanardini/...`). Update paths in each entry point before running in a new environment.
- Training scripts are thin configuration wrappers around `utils.model_training`.
- Evaluation notebooks depend on helpers in `utils.plotting`.

## References
For the full list of references, see the [paper](https://arxiv.org/abs/2605.22858).
For an additional reference on a more extensive version of the work, see the author's MSc theses [here (G.Z.)](https://repository.tudelft.nl/record/uuid:e1e4aba1-87d0-477c-8264-bfb1509aa3ea) and [here (P.v.d.K](https://resolver.tudelft.nl/uuid:e89c0857-496b-40a4-9361-c5a94680b908)

