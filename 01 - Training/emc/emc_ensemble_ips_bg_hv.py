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
PROJECT_NAME = "emc_ips+bg+hv"
DATA_FOLDER_IPS = "/space/gzanardini/emc_whole/split"
DATA_FOLDER_BG = "/space/gzanardini/emc_background/split"
DATA_FOLDER_HV = "/space/gzanardini/emc_hv/"
LOG_FOLDER = "/space/gzanardini/emc/"
N_JOBS_XGB = 4
NUM_WORKERS = 10
SIMPLEX_ALPHA = 1.05

FEATURE_NAMES = ["cc", "cwt", "dwt", "plv", "mst", "sst", "spectral", "utm", "gcc", "gplv"]

BEST_PARAMETERS_IPS = {
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

BEST_PARAMETERS_BACKGROUND = {
    "spectral": ("Cz", 5, "skew"),
    "cwt": ("Cz", 2, "kurt"),
    "dwt": ("Laplacian", 10, "median"),
    "mst": ("Cz", 60, "mean"),
    "sst": ("Cz", 1, "kurt"),
    "cc": ("BipolarDB", 2, "median"),
    "plv": ("CAR", 2, "mean"),
    "gcc": ("CAR", 2, "mean"),
    "gplv": ("Cz", 20, "median"),
    "utm": ("Laplacian", 20, "median"),
}

BEST_PARAMETERS_HV = {
    "spectral": ("BipolarDB", 60, "skew"),
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

def build_config():
    return EnsembleExperimentConfig(
        dataset_name="emc_ips_background_hv",
        project_name=PROJECT_NAME,
        log_folder=LOG_FOLDER,
        n_runs=N_RUNS,
        run_name_template="{feature_set}_run_{run_n}",
        device=DEVICE,
        cuda_idx=N_CUDA,
        data_folders={"ips": DATA_FOLDER_IPS, "background": DATA_FOLDER_BG, "hv": DATA_FOLDER_HV},
        source_type="triple_source",
        feature_names=FEATURE_NAMES,
        best_parameters_by_source={
            "ips": BEST_PARAMETERS_IPS,
            "background": BEST_PARAMETERS_BACKGROUND,
            "hv": BEST_PARAMETERS_HV,
        },
        cache_array_converter=cp.array,
        n_jobs_xgb=N_JOBS_XGB,
        n_parallel_features=NUM_WORKERS,
        simplex_alpha=SIMPLEX_ALPHA,
        wandb_check_existing=True,
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
