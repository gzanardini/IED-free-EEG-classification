# Classification of IED-free EEG responses for epilepsy diagnosis

This repository contains code and helpers for preprocessing, feature extraction, and training used in the TUH epilepsy experiments. The README below focuses on the three most relevant areas for development and running experiments: preprocessing, training, and utilities.

**Relevant Folders**

- [00 - Preprocessing](00%20-%20Preprocessing/README.md): Preprocessing and feature-extraction tools for TUH data. See the folder for scripts such as `preprocess_tuh_epilepsy.py`, `feature_extraction_tuh.py`, and the `PREPROCESSING_GUIDE.md` walkthrough.
- [01 - Training](01%20-%20Training/): Training and ensemble scripts. Contains two main groups:
  - `emc/` — scripts for the EMC experiments (ensemble and single-set training/evaluation).
  - `tuh/` — scripts for TUH-based training and ensembles.
- [utils](utils/): Shared helper modules used across preprocessing and training. Notable modules include `base.py`, `ensemble_common.py`, `ensemble_pipeline.py`, `ensemble_runner.py`, `feature_extraction_funcs.py`, `myutils.py`, `single_set_common.py`, and `TUEP.py`. See [utils/__init__.py](utils/__init__.py) for package entry points.

**Quickstart (high-level)**

1. Prepare raw TUH data and run preprocessing/feature extraction.

	Example:

	```bash
	python "00 - Preprocessing/preprocess_tuh_epilepsy.py"
	```

	For detailed steps and parameters, refer to [00 - Preprocessing/README.md](00%20-%20Preprocessing/README.md) and `PREPROCESSING_GUIDE.md` in that folder.

2. Run training or ensemble scripts from `01 - Training/` after features are prepared.

	Example (run a training script):

	```bash
	python "01 - Training/tuh/tuh_ensemble_background.py"
	```

	See the `emc/` and `tuh/` subfolders for available entry points and configuration patterns.

3. Use helpers from `utils/` when writing or modifying preprocessing or training code — common functionality (data loading, feature helpers, ensemble utilities) lives here.

**Where to look next**

- Start with [00 - Preprocessing/README.md](00%20-%20Preprocessing/README.md) to prepare data and features.
- Inspect `01 - Training/emc/` and `01 - Training/tuh/` for training workflows and example scripts.
- Browse `utils/` for reusable utilities and pipeline scaffolding.

If you'd like, I can also:
- Add runnable examples for a full preprocessing → feature extraction → training pipeline.
- Create a short quickstart script that chains the common steps.

