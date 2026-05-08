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


class DummyXGB:
    fit_calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, x, y):
        DummyXGB.fit_calls += 1
        return self

    def predict(self, x):
        probs = self.predict_proba(x)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, x):
        x = np.asarray(x)
        base = x[:, 0].astype(float)
        scaled = 1 / (1 + np.exp(-base))
        scaled = np.clip(scaled, 0.05, 0.95)
        return np.column_stack([1 - scaled, scaled])


def test_calculate_bac_basic_properties():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    bac, fpr, tpr, thresholds = mt.calculate_bac(labels, scores, sens_thresh=0.8)

    assert 0 <= bac <= 1
    assert len(fpr) == len(tpr)
    assert len(thresholds) == len(tpr)


def test_handle_complex_numbers_ndarray_sanitizes_values():
    features = np.array([[1 + 2j, np.inf], [3 - 4j, -np.inf]])
    processed = mt.handle_complex_numbers(features)

    assert np.isrealobj(processed)
    assert np.isnan(processed[0, 1])
    assert np.isnan(processed[1, 1])
    assert processed[0, 0] == np.abs(1 + 2j)


def test_train_xgb_folds_output_shapes_with_mocked_model(monkeypatch):
    monkeypatch.setattr(mt, "XGBClassifier", DummyXGB)
    DummyXGB.fit_calls = 0

    features = np.arange(24).reshape(12, 2)
    labels = np.array([0, 1] * 6)
    splits = mt.build_loocv_splits(len(labels))

    y_pred, y_score, y_true = mt.train_xgb_folds(
        features,
        labels,
        splits,
        seed=13,
        model_params={
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "as_device_array": lambda x: x,
            "append_mode": "append",
        },
    )

    assert y_pred.shape == (len(labels),)
    assert y_score.shape == (len(labels),)
    assert y_true.shape == (len(labels),)
    assert DummyXGB.fit_calls == len(labels)


def test_subject_loso_splits_group_by_subject():
    subjects = np.array(["s1", "s1", "s2", "s3", "s3"])
    unique_subjects = np.array(["s1", "s2", "s3"])

    splits = mt.build_subject_loso_splits(subjects, unique_subjects)

    expected_tests = [
        np.array([0, 1]),
        np.array([2]),
        np.array([3, 4]),
    ]
    for (_, test_idx), expected in zip(splits, expected_tests):
        assert np.array_equal(test_idx, expected)


def test_build_train_val_test_indices_hold_outs_subject_and_splits_rest():
    description = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4", "s5", "s5", "s6", "s6"],
        }
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])

    train_idx, val_idx, test_idx = mt.build_train_val_test_indices(
        description,
        labels,
        subject="s1",
        split_ratio=0.4,
        seed=7,
    )

    train_subjects = set(description.iloc[train_idx]["subject"])
    val_subjects = set(description.iloc[val_idx]["subject"])
    test_subjects = set(description.iloc[test_idx]["subject"])

    assert test_subjects == {"s1"}
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.union(val_subjects) == {"s2", "s3", "s4", "s5", "s6"}


def test_apply_subject_filters_supports_include_and_skip():
    unique_subjects = np.array(["s1", "s2", "s3", "s4"])
    filtered = mt.apply_subject_filters(unique_subjects, include_subjects=["s2", "s3"], subjects_to_skip=["s3"])
    assert np.array_equal(filtered, np.array(["s2"]))


def test_save_run_outputs_uses_standardized_schema_and_paths(tmp_path):
    summary_df = pd.DataFrame(
        [
            {
                "project_name": "proj",
                "dataset_name": "dataset",
                "mode": "single_set",
                "run": 0,
                "seed": 11,
                "run_name": "feature|CAR|10|mean_run_0",
                "feature_set": "feature|CAR|10|mean",
                "auc": 0.7,
                "bac": 0.6,
                "bac80": 0.5,
                "accuracy": 0.75,
                "precision": 0.8,
                "recall": 0.7,
                "f1_score": 0.74,
                "auprc": 0.72,
                "ap": 0.71,
            }
        ]
    )
    predictions_df = pd.DataFrame(
        [
            {
                "project_name": "proj",
                "dataset_name": "dataset",
                "mode": "single_set",
                "run": 0,
                "seed": 11,
                "run_name": "feature|CAR|10|mean_run_0",
                "feature_set": "feature|CAR|10|mean",
                "fold_id": 0,
                "subject_id": "s1",
                "y_true": 1,
                "y_pred": 1,
                "y_prob": 0.8,
            }
        ]
    )

    summary_path, predictions_path = mt.save_run_outputs(
        summary_df,
        predictions_df,
        str(tmp_path),
        "proj",
        "feature|CAR|10|mean_run_0",
        11,
    )

    assert pathlib.Path(summary_path).name == "summary__feature|CAR|10|mean_run_0__seed_11.csv"
    assert pathlib.Path(predictions_path).name == "predictions__feature|CAR|10|mean_run_0__seed_11.csv"

    saved_summary = pd.read_csv(summary_path)
    saved_predictions = pd.read_csv(predictions_path)

    assert list(saved_summary.columns) == list(summary_df.columns)
    assert list(saved_predictions.columns) == list(predictions_df.columns)


def test_save_predictions_csv_uses_standardized_prediction_columns(tmp_path):
    path = mt.save_predictions_csv(
        y_pred=np.array([0, 1]),
        y_score=np.array([0.2, 0.8]),
        y_true=np.array([0, 1]),
        output_dir=str(tmp_path / "proj"),
        name_parts=("CAR", "spectral", 10, "mean"),
        run_n=2,
        seed=99,
    )

    saved = pd.read_csv(path)
    assert list(saved.columns) == [
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
    assert saved["run_name"].iloc[0] == "spectral_CAR_10s_mean_run_2"


def test_train_ensemble_models_returns_weights_in_feature_order(monkeypatch):
    monkeypatch.setattr(mt, "XGBClassifier", DummyXGB)

    cache = {
        "f2": np.array([[0.2], [0.8], [0.1], [0.9], [0.3], [0.7], [0.4], [0.6]]),
        "f1": np.array([[0.9], [0.1], [0.8], [0.2], [0.7], [0.3], [0.6], [0.4]]),
    }
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    result = mt.train_ensemble_models(
        ["f2", "f1"],
        train_idxs=np.array([0, 1, 2, 3]),
        val_idxs=np.array([4, 5]),
        test_idxs=np.array([6, 7]),
        y_train=y[[0, 1, 2, 3]],
        y_val=y[[4, 5]],
        seed=5,
        cache=cache,
        xgb_params={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1},
        n_jobs_xgb=1,
        device="cpu",
        n_parallel_features=2,
        simplex_alpha=1.05,
    )

    assert len(result["weights"]) == 2
    assert len(result["calibrators"]) == 2
    assert result["test_probs"].shape == (2,)
    assert result["test_preds"].shape == (2,)
    assert np.isclose(np.sum(result["weights"]), 1.0)


def test_converted_scripts_import_shared_runner():
    checks = {
        "01 - Training/tuh/tuh_ss_bg.py": "from utils.model_training import SingleSetExperimentConfig, run_single_set_experiment",
        "01 - Training/tuh/tuh_ensemble_background.py": "from utils.model_training import EnsembleExperimentConfig, run_ensemble_experiment",
        "01 - Training/emc/emc_ss_background.py": "from utils.model_training import SingleSetExperimentConfig, run_single_set_experiment",
        "01 - Training/emc/emc_ensemble_background.py": "from utils.model_training import EnsembleExperimentConfig, run_ensemble_experiment",
    }

    for rel_path, snippet in checks.items():
        content = (ROOT / rel_path).read_text()
        assert snippet in content
