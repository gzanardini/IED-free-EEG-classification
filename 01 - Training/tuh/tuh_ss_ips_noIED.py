import cupy as cp
from utils.model_training import SingleSetExperimentConfig, run_single_set_experiment


N_RUNS = 5
N_CUDA = 1
PROJECT_NAME = "tuh_singleset_noied_final"
FEAT_FOLDER = "/space/gzanardini/tuh_whole/split/"
LOG_FOLDER = "/space/gzanardini/tuh/"

MONTAGES = ["CAR", "Cz", "BipolarDB", "Laplacian"]
SEGMENT_LENGTHS = [1, 2, 5, 10, 20, 60]
FEATURE_NAMES = ["cc", "cwt", "dwt", "gcc", "gplv", "plv", "mst", "sst", "spectral", "utm"]
COMBINERS = ["mean", "median", "std", "skew", "kurt"]
SUBJECTS_TO_SKIP = ["aaaaajgj", "aaaaakcd"]


def build_config():
    return SingleSetExperimentConfig(
        dataset_name="tuh_ips_noieds",
        project_name=PROJECT_NAME,
        log_folder=LOG_FOLDER,
        n_runs=N_RUNS,
        run_name_template="{feature_set}_run_{run_n}",
        device=f"cuda:{N_CUDA}",
        cuda_idx=N_CUDA,
        wandb_reinit=True,
        data_folder=FEAT_FOLDER,
        montages=MONTAGES,
        segment_lengths=SEGMENT_LENGTHS,
        feature_names=FEATURE_NAMES,
        combiners=COMBINERS,
        subjects_to_skip=SUBJECTS_TO_SKIP,
        xgb_params={
            "n_estimators": 100,
            "max_depth": 7,
            "subsample": 0.8,
            "n_jobs": 4,
            "gamma": 0.1,
            "learning_rate": 0.05,
        },
        as_device_array=cp.array,
        metadata={"epochs": False},
    )


def main():
    run_single_set_experiment(build_config())

if __name__ == "__main__":
    main()
