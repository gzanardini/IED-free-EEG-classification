import cupy as cp
import os
import sys
import pandas as pd

# Ensure the workspace root is on sys.path so local packages (like `utils`) are importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.model_training import EnsembleExperimentConfig, run_ensemble_experiment


N_RUNS = 5
N_CUDA = 0
DEVICE = "cuda"
PROJECT_NAME = "emc_ensemble_hv_responders"
DATA_FOLDER = "/space/gzanardini/emc_hv/"
LOG_FOLDER = "/space/gzanardini/emc/"
N_JOBS_XGB = 4
NUM_WORKERS = 10
SIMPLEX_ALPHA = 1.05
RESPONDER_CSV = "/users/gzanardini/eeg_thesis/emc/hv_responders.csv"

FEATURE_NAMES = ["cc", "cwt", "dwt", "plv", "mst", "sst", "sr", "utm", "gcc", "gplv"]

BEST_PARAMETERS = {
    "sr": ("BipolarDB", 60, "skew"),
    "cwt": ("CAR", 2, "median"),
    "dwt": ("Laplacian", 60, "std"),
    "mst": ("BipolarDB", 60, "kurtosis"),
    "sst": ("BipolarDB", 60, "kurtosis"),
    "cc": ("Laplacian", 60, "std"),
    "plv": ("CAR", 60, "skew"),
    "gcc": ("BipolarDB", 10, "skew"),
    "gplv": ("BipolarDB", 2, "kurt"),
    "utm": ("Cz", 10, "kurt"),
}


def load_included_subjects():
    hv_responders = pd.read_csv(RESPONDER_CSV)
    return hv_responders.loc[hv_responders["hv_success"] == 1, "subject"].tolist()


def build_config():
    return EnsembleExperimentConfig(
        dataset_name="emc_hv_responders",
        project_name=PROJECT_NAME,
        log_folder=LOG_FOLDER,
        n_runs=N_RUNS,
        run_name_template="{feature_set}_run_{run_n}",
        device=DEVICE,
        cuda_idx=N_CUDA,
        data_folder=DATA_FOLDER,
        source_type="single_source",
        feature_names=FEATURE_NAMES,
        best_parameters=BEST_PARAMETERS,
        cache_array_converter=cp.array,
        n_jobs_xgb=N_JOBS_XGB,
        n_parallel_features=NUM_WORKERS,
        simplex_alpha=SIMPLEX_ALPHA,
        include_subjects=load_included_subjects(),
        xgb_params={
            "n_estimators": 100,
            "max_depth": 6,
            "subsample": 0.9,
            "gamma": 0.1,
            "learning_rate": 0.01,
        },
    )


def main():
    run_ensemble_experiment(build_config())


if __name__ == "__main__":
    main()
