import cupy as cp
from utils.model_training import SingleSetExperimentConfig, run_single_set_experiment


N_RUNS = 5
N_CUDA = 0
DATA_FOLDER = "/space/gzanardini/emc_hv/"
PROJECT_NAME = "emc_hypervent"
LOG_FOLDER = "/space/gzanardini/emc/"

MONTAGES = ["BipolarDB"]
SEGMENT_LENGTHS = [2, 5, 10, 20, 60, 120]
FEATURE_NAMES = ["cc", "cwt", "dwt", "gcc", "plv", "gplv", "utm", "mst", "sst", "sa", "sr"]
COMBINERS = ["mean", "median", "std", "skew", "kurt"]


def build_config():
    return SingleSetExperimentConfig(
        dataset_name="emc_hypervent",
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
