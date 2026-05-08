import cupy as cp
import os
import sys

# Ensure the workspace root is on sys.path so local packages (like `utils`) are importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.model_training import SingleSetExperimentConfig, run_single_set_experiment


N_RUNS = 5
N_CUDA = 1
DATA_FOLDER = "/space/gzanardini/emc_whole/split/"
PROJECT_NAME = "emc_singleset_final"
LOG_FOLDER = "/space/gzanardini/emc/"

MONTAGES = ["CAR", "Cz", "BipolarDB", "Laplacian"]
SEGMENT_LENGTHS = [1, 2, 5, 10, 20, 60]
FEATURE_NAMES = ["cc", "cwt", "dwt", "gcc", "gplv", "plv", "mst", "sst", "spectral", "utm"]
COMBINERS = ["mean", "median", "std", "skew", "kurt"]


def build_config():
    return SingleSetExperimentConfig(
        dataset_name="emc_ips",
        project_name=PROJECT_NAME,
        log_folder=LOG_FOLDER,
        n_runs=N_RUNS,
        run_name_template="{feature_set}_run_{run_n}",
        device=f"cuda:{N_CUDA}",
        cuda_idx=N_CUDA,
        wandb_reinit=True,
        data_folder=DATA_FOLDER,
        montages=MONTAGES,
        segment_lengths=SEGMENT_LENGTHS,
        feature_names=FEATURE_NAMES,
        combiners=COMBINERS,
        xgb_params={
            "n_estimators": 100,
            "max_depth": 6,
            "subsample": 0.9,
            "n_jobs": 4,
            "gamma": 0.1,
            "learning_rate": 0.1,
        },
        as_device_array=cp.array,
        metadata={"epochs": False},
    )


def main():
    run_single_set_experiment(build_config())

if __name__ == "__main__":
    main()
