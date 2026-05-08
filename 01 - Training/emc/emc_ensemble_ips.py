import cupy as cp
import os
import sys
# Ensure the workspace root is on sys.path so local packages (like `utils`) are importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.model_training import EnsembleExperimentConfig, run_ensemble_experiment


N_RUNS = 5
N_CUDA = 0
DEVICE = "cuda"
PROJECT_NAME = "emc_ensemble_retrain"
DATA_FOLDER = "/space/gzanardini/emc_whole/split"
LOG_FOLDER = "/space/gzanardini/emc/"
N_JOBS_XGB = 4
NUM_WORKERS = 10
SIMPLEX_ALPHA = 1.05

FEATURE_NAMES = ["cc", "cwt", "dwt", "plv", "mst", "sst", "spectral", "utm", "gcc", "gplv"]

BEST_PARAMETERS = {
    "spectral": ("Cz", 10, "std"),
    "cwt": ("BipolarDB", 2, "median"),
    "dwt": ("Laplacian", 10, "median"),
    "mst": ("BipolarDB", 60, "median"),
    "sst": ("CAR", 10, "median"),
    "cc": ("CAR", 1, "std"),
    "plv": ("Laplacian", 60, "kurt"),
    "gcc": ("CAR", 60, "median"),
    "gplv": ("Laplacian", 2, "std"),
    "utm": ("Laplacian", 20, "std"),
}


def build_config():
    return EnsembleExperimentConfig(
        dataset_name="emc_ips",
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
        skip_existing_outputs=True,
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
