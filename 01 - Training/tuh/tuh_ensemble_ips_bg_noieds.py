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
DEVICE = f"cuda:{N_CUDA}"
PROJECT_NAME = "tuh_ips+bg_noieds"
DATA_FOLDER_IPS = "/space/gzanardini/tuh_whole/split"
DATA_FOLDER_BG = "/space/gzanardini/tuh_background/split"
LOG_FOLDER = "/space/gzanardini/tuh/"
N_JOBS_XGB = 1
NUM_WORKERS = 10
SIMPLEX_ALPHA = 1.05
SUBJECTS_TO_SKIP = ["aaaaajgj", "aaaaakcd"]

FEATURE_NAMES = ["cc", "cwt", "dwt", "plv", "mst", "sst", "spectral", "utm", "gcc", "gplv"]

BEST_PARAMETERS_IPS = {
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

BEST_PARAMETERS_BACKGROUND = {
    "spectral": ("BipolarDB", 2, "kurt"),
    "cwt": ("Cz", 1, "skew"),
    "dwt": ("Cz", 10, "skew"),
    "mst": ("BipolarDB", 1, "median"),
    "sst": ("Laplacian", 20, "std"),
    "cc": ("CAR", 120, "mean"),
    "plv": ("Cz", 60, "std"),
    "gcc": ("Cz", 20, "kurt"),
    "gplv": ("Laplacian", 10, "mean"),
    "utm": ("Laplacian", 60, "median"),
}


def build_config():
    return EnsembleExperimentConfig(
        dataset_name="tuh_ips_background_noieds",
        project_name=PROJECT_NAME,
        log_folder=LOG_FOLDER,
        n_runs=N_RUNS,
        run_name_template="{feature_set}_run_{run_n}",
        device=DEVICE,
        cuda_idx=N_CUDA,
        data_folders={"ips": DATA_FOLDER_IPS, "background": DATA_FOLDER_BG},
        source_type="dual_source",
        feature_names=FEATURE_NAMES,
        best_parameters_by_source={
            "ips": BEST_PARAMETERS_IPS,
            "background": BEST_PARAMETERS_BACKGROUND,
        },
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
