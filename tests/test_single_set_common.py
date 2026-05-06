import importlib.util
import pathlib
import sys
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "utils" / "single_set_common.py"

wandb_stub = types.ModuleType("wandb")
wandb_stub.login = lambda key=None: None
wandb_stub.log = lambda data=None: None
wandb_stub.Table = lambda data=None, columns=None: {"data": data, "columns": columns}
wandb_stub.plot = types.SimpleNamespace(
    confusion_matrix=lambda **kwargs: None,
    line=lambda *args, **kwargs: None,
)
sys.modules.setdefault("wandb", wandb_stub)

xgboost_stub = types.ModuleType("xgboost")
xgboost_stub.XGBClassifier = object
sys.modules.setdefault("xgboost", xgboost_stub)

SPEC = importlib.util.spec_from_file_location("single_set_common", MODULE_PATH)
ssc = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(ssc)


class DummyXGB:
    fit_calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, x, y):
        DummyXGB.fit_calls += 1
        self._n = len(x)
        return self

    def predict(self, x):
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x):
        probs = np.full((len(x), 2), 0.5)
        probs[:, 1] = 0.25
        probs[:, 0] = 0.75
        return probs


def test_calculate_bac_basic_properties():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    bac, fpr, tpr, thresholds = ssc.calculate_bac(labels, scores, sens_thresh=0.8)

    assert 0 <= bac <= 1
    assert len(fpr) == len(tpr)
    assert len(thresholds) == len(tpr)


def test_handle_complex_numbers_ndarray_sanitizes_values():
    features = np.array([[1 + 2j, np.inf], [3 - 4j, -np.inf]])
    processed = ssc.handle_complex_numbers(features)

    assert np.isrealobj(processed)
    assert np.isnan(processed[0, 1])
    assert np.isnan(processed[1, 1])
    assert processed[0, 0] == np.abs(1 + 2j)


def test_train_xgb_folds_output_shapes_with_mocked_model(monkeypatch):
    monkeypatch.setattr(ssc, "XGBClassifier", DummyXGB)
    DummyXGB.fit_calls = 0

    features = np.arange(24).reshape(12, 2)
    labels = np.array([0, 1] * 6)
    splits = ssc.build_loocv_splits(len(labels))

    y_pred, y_score, y_true = ssc.train_xgb_folds(
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

    splits = ssc.build_subject_loso_splits(subjects, unique_subjects)

    expected_tests = [
        np.array([0, 1]),
        np.array([2]),
        np.array([3, 4]),
    ]
    for (_, test_idx), expected in zip(splits, expected_tests):
        assert np.array_equal(test_idx, expected)


def test_loocv_splits_hold_out_single_sample():
    splits = ssc.build_loocv_splits(5)

    assert len(splits) == 5
    held_out = []
    for train_idx, test_idx in splits:
        assert len(test_idx) == 1
        assert len(train_idx) == 4
        held_out.append(test_idx[0])

    assert sorted(held_out) == [0, 1, 2, 3, 4]


def test_run_name_formats_preserved_in_scripts():
    root = ROOT

    checks = {
        "tuh/tuh_ss_ips.py": "name=f'{feature_name}_{montage}_{segment_length}s_{combiner}_run_{run_n}'",
        "tuh/tuh_ss_bg.py": "name=f'{feature_name}_{montage}_{segment_length}s_{combiner}_run_{run_n}'",
        "tuh/tuh_ss_ips_noIED.py": "name=f'{feature_name}_{montage}_{segment_length}s_{combiner}_run_{run_n}'",
        "tuh/tuh_ss_bg_noieds.py": "name=f'{feature_name}_{montage}_{segment_length}s_{combiner}_run_{run_n}'",
        "emc/emc_ss_ips.py": "name=f'{feature_name}_{montage}_{segment_length}s_{combiner}run_{run_n}'",
        "emc/emc_ss_background.py": "name=f'{feature_name}_{montage}_{segment_length}s_{combiner}run_{run_n}'",
        "emc/emc_ss_hypervent.py": "name=RUN_NAME",
    }

    for rel_path, snippet in checks.items():
        content = (root / rel_path).read_text()
        assert snippet in content
