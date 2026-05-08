import cupy as cp

from utils.model_training import EnsembleExperimentConfig, run_ensemble_experiment


N_RUNS = 5
N_CUDA = 0
DEVICE = "cuda"
PROJECT_NAME = "emc_ensemble_background" # Name of the project for logging purposes, e.g., in Weights & Biases
DATA_FOLDER = "/space/gzanardini/emc_background/split" # Path to the folder containing the preprocessed features
LOG_FOLDER = "/space/gzanardini/emc/" # Path to the folder where logs and model checkpoints will be saved. Make sure this folder exists and is writable.
N_JOBS_XGB = 4
NUM_WORKERS = 10
SIMPLEX_ALPHA = 1.05

FEATURE_NAMES = ["cc", "cwt", "dwt", "plv", "mst", "sst", "spectral", "utm", "gcc", "gplv"]

BEST_PARAMETERS = {
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

def build_config():
    return EnsembleExperimentConfig(
        dataset_name="emc_background",
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
