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
DEVICE = "cpu"
PROJECT_NAME = "tuh_ensemble_retrain_noieds"
DATA_FOLDER = "/space/gzanardini/tuh_whole/split"
LOG_FOLDER = "/space/gzanardini/tuh/"
N_JOBS_XGB = 4
NUM_WORKERS = 10
SIMPLEX_ALPHA = 1.05
SUBJECTS_TO_SKIP = ["aaaaajgj", "aaaaakcd"]

FEATURE_NAMES = ["cc", "cwt", "dwt", "plv", "mst", "sst", "spectral", "utm", "gcc", "gplv"]

BEST_PARAMETERS = {
    "spectral": ("CAR", 1, "skew"),
    "cwt": ("BipolarDB", 60, "std"),
    "dwt": ("Cz", 10, "skew"),
    "mst": ("BipolarDB", 10, "skew"),
    "sst": ("Laplacian", 20, "skew"),
    "cc": ("Cz", 10, "skew"),
    "plv": ("Laplacian", 2, "std"),
    "gcc": ("CAR", 1, "std"),
    "gplv": ("BipolarDB", 1, "mean"),
    "utm": ("Laplacian", 60, "mean"),
}


def build_config():
    return EnsembleExperimentConfig(
        dataset_name="tuh_ips_noieds",
        project_name=PROJECT_NAME,
        log_folder=LOG_FOLDER,
        n_runs=N_RUNS,
        run_name_template="{feature_set}_run_{run_n}",
        device=DEVICE,
        cuda_idx=N_CUDA,
        wandb_reinit=True,
        wandb_check_existing=True,
        data_folder=DATA_FOLDER,
        source_type="single_source",
        feature_names=FEATURE_NAMES,
        best_parameters=BEST_PARAMETERS,
        cache_array_converter=cp.array,
        n_jobs_xgb=N_JOBS_XGB,
        n_parallel_features=NUM_WORKERS,
        simplex_alpha=SIMPLEX_ALPHA,
        subjects_to_skip=SUBJECTS_TO_SKIP,
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
