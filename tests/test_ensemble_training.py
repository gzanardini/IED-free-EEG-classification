import importlib.util
import pathlib
import sys
import types

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "utils" / "model_training.py"

wandb_stub = types.ModuleType("wandb")
wandb_stub.login = lambda key=None: None
wandb_stub.init = lambda **kwargs: None
wandb_stub.finish = lambda: None
wandb_stub.log = lambda data=None, **kwargs: None
wandb_stub.Table = lambda data=None, columns=None: {"data": data, "columns": columns}
wandb_stub.Histogram = lambda values=None: values
wandb_stub.plot = types.SimpleNamespace(
    confusion_matrix=lambda **kwargs: None,
    line=lambda *args, **kwargs: None,
)
wandb_stub.config = types.SimpleNamespace(update=lambda payload=None: None)
wandb_stub.Api = lambda timeout=None: types.SimpleNamespace(runs=lambda path=None: [])
sys.modules.setdefault("wandb", wandb_stub)

xgboost_stub = types.ModuleType("xgboost")
xgboost_stub.XGBClassifier = object
sys.modules.setdefault("xgboost", xgboost_stub)

SPEC = importlib.util.spec_from_file_location("model_training", MODULE_PATH)
mt = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules["model_training"] = mt
SPEC.loader.exec_module(mt)


def write_description_csv(folder: pathlib.Path, subjects: list[str], labels: list[int]) -> None:
    pd.DataFrame({"subject": subjects, "epilepsy": labels}).to_csv(folder / "description.csv", index=False)


def test_load_data_both_aligned_reorders_to_background_subjects(tmp_path):
    bg = tmp_path / "bg"
    ips = tmp_path / "ips"
    bg.mkdir()
    ips.mkdir()

    write_description_csv(bg, ["s1", "s2", "s3"], [0, 1, 0])
    write_description_csv(ips, ["s3", "s1", "s2"], [0, 0, 1])

    description, labels, subjects, unique_subjects, subject_labels, ips_reorder_indices = mt.load_data_both_aligned(
        str(bg),
        str(ips),
    )

    assert list(description["subject"]) == ["s1", "s2", "s3"]
    assert np.array_equal(labels, np.array([0, 1, 0]))
    assert np.array_equal(subjects, np.array(["s1", "s2", "s3"]))
    assert np.array_equal(unique_subjects, np.array(["s1", "s2", "s3"]))
    assert ips_reorder_indices == [1, 2, 0]
    assert np.array_equal(subject_labels[:, 0], np.array(["s1", "s2", "s3"]))


def test_preload_combined_feature_cache_reorders_and_concatenates(tmp_path):
    bg = tmp_path / "bg"
    ips = tmp_path / "ips"
    bg.mkdir()
    ips.mkdir()

    np.save(bg / "feat_CAR_1s_mean.npy", np.array([[1, 10], [2, 20], [3, 30]]))
    np.save(ips / "feat_CAR_1s_mean.npy", np.array([[300, 3000], [100, 1000], [200, 2000]]))

    cache = mt.preload_combined_feature_cache(
        feature_names=["feat"],
        best_parameters_background={"feat": ("CAR", 1, "mean")},
        best_parameters_ips={"feat": ("CAR", 1, "mean")},
        data_folder_bg=str(bg),
        data_folder_ips=str(ips),
        ips_reorder_indices=[1, 2, 0],
        array_converter=lambda x: x,
    )

    combined = cache["feat"]
    expected = np.array(
        [
            [1, 10, 100, 1000],
            [2, 20, 200, 2000],
            [3, 30, 300, 3000],
        ]
    )
    assert np.array_equal(combined, expected)


def test_run_ensemble_experiment_writes_standardized_outputs(tmp_path, monkeypatch):
    description = pd.DataFrame({"subject": ["s1", "s2", "s3", "s4", "s5", "s6"]})
    labels = np.array([0, 0, 0, 1, 1, 1])
    subjects = description["subject"].to_numpy()
    unique_subjects = np.array(["s1", "s2", "s3", "s4", "s5", "s6"])
    cache = {"f1": np.ones((6, 1)), "f2": np.ones((6, 1))}

    monkeypatch.setattr(mt, "setup_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mt,
        "prepare_ensemble_dataset",
        lambda config: (description, labels, subjects, unique_subjects, cache),
    )

    def fake_train_ensemble_models(
        feature_combination,
        train_idxs,
        val_idxs,
        test_idxs,
        y_train,
        y_val,
        seed,
        **kwargs,
    ):
        test_labels = labels[test_idxs]
        test_probs = np.where(test_labels == 1, 0.8, 0.2)
        return {
            "feature_models": [],
            "meta_model": None,
            "calibrators": [],
            "weights": np.array([0.5, 0.5]),
            "lr_weights": np.array([0.5, 0.5]),
            "val_probs": np.where(y_val == 1, 0.8, 0.2),
            "test_probs": test_probs,
            "test_preds": test_labels.copy(),
            "opt_threshold": 0.5,
            "auc": 1.0,
            "bac": 1.0,
            "bac80": 1.0,
        }

    monkeypatch.setattr(mt, "train_ensemble_models", fake_train_ensemble_models)

    config = mt.EnsembleExperimentConfig(
        dataset_name="ensemble_demo",
        project_name="proj",
        log_folder=str(tmp_path),
        n_runs=1,
        run_name_template="{feature_set}_run_{run_n}",
        device="cpu",
        feature_names=["f1", "f2"],
        source_type="single_source",
        data_folder="unused",
        best_parameters={"f1": ("CAR", 1, "mean"), "f2": ("CAR", 1, "mean")},
        combination_min_len=2,
        combination_max_len=3,
        log_to_wandb=False,
    )

    paths = mt.run_ensemble_experiment(config)
    assert len(paths) == 1

    summary_path, predictions_path = paths[0]
    summary_df = pd.read_csv(summary_path)
    predictions_df = pd.read_csv(predictions_path)

    assert list(summary_df.columns) == [
        "project_name",
        "dataset_name",
        "mode",
        "run",
        "seed",
        "run_name",
        "feature_set",
        "auc",
        "bac",
        "bac80",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auprc",
        "ap",
        "combination_length",
    ]
    assert list(predictions_df.columns) == [
        "project_name",
        "dataset_name",
        "mode",
        "run",
        "seed",
        "run_name",
        "feature_set",
        "fold_id",
        "subject_id",
        "y_true",
        "y_pred",
        "y_prob",
    ]
    assert len(predictions_df) == 6
    assert set(predictions_df["subject_id"]) == {"s1", "s2", "s3", "s4", "s5", "s6"}
    assert summary_df["feature_set"].iloc[0] == "f1+f2"


def test_run_ensemble_experiment_respects_include_subjects(tmp_path, monkeypatch):
    description = pd.DataFrame({"subject": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]})
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    subjects = description["subject"].to_numpy()
    unique_subjects = np.array(["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"])
    cache = {"f1": np.ones((8, 1)), "f2": np.ones((8, 1))}

    monkeypatch.setattr(mt, "setup_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mt,
        "prepare_ensemble_dataset",
        lambda config: (description, labels, subjects, unique_subjects, cache),
    )

    def fake_train_ensemble_models(
        feature_combination,
        train_idxs,
        val_idxs,
        test_idxs,
        y_train,
        y_val,
        seed,
        **kwargs,
    ):
        test_labels = labels[test_idxs]
        return {
            "feature_models": [],
            "meta_model": None,
            "calibrators": [],
            "weights": np.array([0.5, 0.5]),
            "lr_weights": np.array([0.5, 0.5]),
            "val_probs": np.where(y_val == 1, 0.8, 0.2),
            "test_probs": np.where(test_labels == 1, 0.8, 0.2),
            "test_preds": test_labels.copy(),
            "opt_threshold": 0.5,
            "auc": 1.0,
            "bac": 1.0,
            "bac80": 1.0,
        }

    monkeypatch.setattr(mt, "train_ensemble_models", fake_train_ensemble_models)

    config = mt.EnsembleExperimentConfig(
        dataset_name="ensemble_subset",
        project_name="proj",
        log_folder=str(tmp_path),
        n_runs=1,
        run_name_template="{feature_set}_run_{run_n}",
        device="cpu",
        feature_names=["f1", "f2"],
        source_type="single_source",
        data_folder="unused",
        best_parameters={"f1": ("CAR", 1, "mean"), "f2": ("CAR", 1, "mean")},
        combination_min_len=2,
        combination_max_len=3,
        include_subjects=["s1", "s2", "s3", "s5", "s6", "s7"],
        log_to_wandb=False,
    )

    paths = mt.run_ensemble_experiment(config)
    predictions_df = pd.read_csv(paths[0][1])
    assert set(predictions_df["subject_id"]) == {"s1", "s2", "s3", "s5", "s6", "s7"}
