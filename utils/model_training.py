from __future__ import annotations

import itertools
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import wandb
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def identity(value):
    return value


@dataclass
class BaseExperimentConfig:
    dataset_name: str
    project_name: str
    log_folder: str
    n_runs: int
    run_name_template: str
    device: str = "cpu"
    cuda_idx: int = 0
    wandb_key: Optional[str] = None
    log_to_wandb: bool = True
    wandb_reinit: bool = False
    wandb_timeout: int = 60
    wandb_check_existing: bool = False
    wandb_check_path: Optional[str] = None
    checkpoint_run_name: Optional[str] = None
    random_seed_upper: int = 5000
    subjects_to_skip: Sequence[str] = field(default_factory=tuple)
    include_subjects: Sequence[str] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SingleSetExperimentConfig(BaseExperimentConfig):
    data_folder: str = ""
    montages: Sequence[str] = field(default_factory=tuple)
    segment_lengths: Sequence[int] = field(default_factory=tuple)
    feature_names: Sequence[str] = field(default_factory=tuple)
    combiners: Sequence[str] = field(default_factory=tuple)
    source_type: str = "single_source"
    xgb_params: dict[str, Any] = field(default_factory=dict)
    as_device_array: Callable[[np.ndarray], Any] = identity
    skip_feature_config: Optional[Callable[[dict[str, Any]], bool]] = None


@dataclass
class EnsembleExperimentConfig(BaseExperimentConfig):
    feature_names: Sequence[str] = field(default_factory=tuple)
    source_type: str = "single_source"
    split_ratio: float = 0.3
    data_folder: Optional[str] = None
    data_folders: dict[str, str] = field(default_factory=dict)
    best_parameters: dict[str, tuple[str, int, str]] = field(default_factory=dict)
    best_parameters_by_source: dict[str, dict[str, tuple[str, int, str]]] = field(default_factory=dict)
    cache_array_converter: Callable[[np.ndarray], Any] = identity
    n_jobs_xgb: int = 1
    n_parallel_features: int = 1
    simplex_alpha: float = 1.05
    combination_min_len: int = 2
    combination_max_len: int = 10
    xgb_params: dict[str, Any] = field(default_factory=dict)
    skip_existing_outputs: bool = False


class SimplexLogistic:
    """Simple wrapper around simplex-constrained logistic weights."""

    def __init__(self, weights: np.ndarray):
        self.coef_ = weights[None, :]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = expit(X @ self.coef_.ravel())
        return np.column_stack([1 - probabilities, probabilities])


def setup_environment(device: str, cuda_idx: int, wandb_key: Optional[str] = None) -> None:
    """Initialize CUDA device selection and wandb login."""
    if device != "cpu":
        from cupy.cuda import Device

        Device(cuda_idx).use()

    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        wandb.login(key=os.getenv("WANDB_API_KEY"))


def handle_complex_numbers(features: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """Convert complex values and replace infinities with NaNs."""
    if isinstance(features, pd.DataFrame):
        for column in features.columns:
            if np.iscomplexobj(features[column]):
                features[column] = features[column].apply(np.abs)
            features[column].replace([np.inf, -np.inf], np.nan, inplace=True)
        return features

    if np.iscomplexobj(features):
        features = np.abs(features)
    elif not np.issubdtype(features.dtype, np.floating):
        features = features.astype(float)
    features[~np.isfinite(features)] = np.nan
    return features


def load_subject_data(path: str):
    """Load subject-level description metadata from a feature folder."""
    description = pd.read_csv(os.path.join(path, "description.csv"))
    labels = description["epilepsy"].to_numpy()
    subjects = description["subject"].to_numpy()
    unique_subjects = np.unique(subjects)

    subject_labels = []
    for subject in unique_subjects:
        label = labels[subjects == subject][0]
        subject_labels.append([subject, label])

    return description, labels, subjects, unique_subjects, np.array(subject_labels)


def load_feature_array(
    base_folder: str,
    feature_name: str,
    montage: str,
    segment_length: int,
    combiner: str,
) -> np.ndarray:
    """Load one feature matrix and sanitize it."""
    data_path = os.path.join(base_folder, f"{feature_name}_{montage}_{segment_length}s_{combiner}.npy")
    features = np.load(data_path)
    features = handle_complex_numbers(features)
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    return features


def load_feature_from_best_params(
    data_folder: str,
    best_parameters: dict[str, tuple[str, int, str]],
    feature_name: str,
    array_converter: Callable[[np.ndarray], Any],
):
    """Load one feature matrix using a best-parameter table."""
    montage, segment_length, combiner = best_parameters[feature_name]
    features = load_feature_array(data_folder, feature_name, montage, segment_length, combiner)
    return array_converter(features), montage, segment_length, combiner


def preload_feature_cache(
    feature_names: Sequence[str],
    best_parameters: dict[str, tuple[str, int, str]],
    data_folder: str,
    array_converter: Callable[[np.ndarray], Any],
) -> dict[str, Any]:
    """Preload all feature arrays from a single source."""
    cache: dict[str, Any] = {}
    for feature_name in feature_names:
        features, _, _, _ = load_feature_from_best_params(
            data_folder,
            best_parameters,
            feature_name,
            array_converter,
        )
        cache[feature_name] = features
    return cache


def load_data_both_aligned(data_folder_bg: str, data_folder_ips: str):
    """Load and align background and IPS datasets on the common subject set."""
    bg_description = pd.read_csv(os.path.join(data_folder_bg, "description.csv"))
    bg_labels = bg_description["epilepsy"].to_numpy()
    bg_subjects = bg_description["subject"].to_numpy()
    bg_unique_subjects = np.unique(bg_subjects)

    ips_description = pd.read_csv(os.path.join(data_folder_ips, "description.csv"))
    ips_labels = ips_description["epilepsy"].to_numpy()
    ips_subjects = ips_description["subject"].to_numpy()

    common_subjects = np.intersect1d(bg_unique_subjects, np.unique(ips_subjects))

    bg_mask = np.isin(bg_subjects, common_subjects)
    filtered_bg_description = bg_description[bg_mask].reset_index(drop=True)
    filtered_bg_labels = bg_labels[bg_mask]
    filtered_bg_subjects = bg_subjects[bg_mask]

    ips_mask = np.isin(ips_subjects, common_subjects)
    filtered_ips_labels = ips_labels[ips_mask]
    filtered_ips_subjects = ips_subjects[ips_mask]

    assert np.array_equal(np.sort(filtered_bg_subjects), np.sort(filtered_ips_subjects))
    assert np.array_equal(
        filtered_bg_labels[np.argsort(filtered_bg_subjects)],
        filtered_ips_labels[np.argsort(filtered_ips_subjects)],
    )

    ips_reorder_indices = [np.where(filtered_ips_subjects == subject)[0][0] for subject in filtered_bg_subjects]

    subject_labels = []
    for subject in common_subjects:
        label = filtered_bg_labels[filtered_bg_subjects == subject][0]
        subject_labels.append([subject, label])

    return (
        filtered_bg_description,
        filtered_bg_labels,
        filtered_bg_subjects,
        common_subjects,
        np.array(subject_labels),
        ips_reorder_indices,
    )

def load_data_triple_aligned(data_folder_bg: str, data_folder_ips: str, data_folder_hv: str):
    """Load and align background, IPS and HV datasets on the common subject set."""
    bg_description = pd.read_csv(os.path.join(data_folder_bg, "description.csv"))
    bg_labels = bg_description["epilepsy"].to_numpy()
    bg_subjects = bg_description["subject"].to_numpy()
    bg_unique_subjects = np.unique(bg_subjects)

    ips_description = pd.read_csv(os.path.join(data_folder_ips, "description.csv"))
    ips_labels = ips_description["epilepsy"].to_numpy()
    ips_subjects = ips_description["subject"].to_numpy()

    hv_description = pd.read_csv(os.path.join(data_folder_hv, "description.csv"))
    hv_labels = hv_description["epilepsy"].to_numpy()
    hv_subjects = hv_description["subject"].to_numpy()

    common_subjects = np.intersect1d(bg_unique_subjects, np.intersect1d(np.unique(ips_subjects), np.unique(hv_subjects)))

    bg_mask = np.isin(bg_subjects, common_subjects)
    filtered_bg_description = bg_description[bg_mask].reset_index(drop=True)
    filtered_bg_labels = bg_labels[bg_mask]
    filtered_bg_subjects = bg_subjects[bg_mask]

    ips_mask = np.isin(ips_subjects, common_subjects)
    filtered_ips_labels = ips_labels[ips_mask]
    filtered_ips_subjects = ips_subjects[ips_mask]

    hv_mask = np.isin(hv_subjects, common_subjects)
    filtered_hv_labels = hv_labels[hv_mask]
    filtered_hv_subjects = hv_subjects[hv_mask]

    # Ensure labels align across sources
    assert np.array_equal(np.sort(filtered_bg_subjects), np.sort(filtered_ips_subjects))
    assert np.array_equal(np.sort(filtered_bg_subjects), np.sort(filtered_hv_subjects))
    assert np.array_equal(
        filtered_bg_labels[np.argsort(filtered_bg_subjects)],
        filtered_ips_labels[np.argsort(filtered_ips_subjects)],
    )
    assert np.array_equal(
        filtered_bg_labels[np.argsort(filtered_bg_subjects)],
        filtered_hv_labels[np.argsort(filtered_hv_subjects)],
    )

    ips_reorder_indices = [np.where(filtered_ips_subjects == subject)[0][0] for subject in filtered_bg_subjects]
    hv_reorder_indices = [np.where(filtered_hv_subjects == subject)[0][0] for subject in filtered_bg_subjects]

    subject_labels = []
    for subject in common_subjects:
        label = filtered_bg_labels[filtered_bg_subjects == subject][0]
        subject_labels.append([subject, label])

    return (
        filtered_bg_description,
        filtered_bg_labels,
        filtered_bg_subjects,
        common_subjects,
        np.array(subject_labels),
        ips_reorder_indices,
        hv_reorder_indices,
    )

def preload_combined_feature_cache(
    feature_names: Sequence[str],
    best_parameters_background: dict[str, tuple[str, int, str]],
    best_parameters_ips: dict[str, tuple[str, int, str]],
    data_folder_bg: str,
    data_folder_ips: str,
    ips_reorder_indices: Sequence[int],
    array_converter: Callable[[np.ndarray], Any],
) -> dict[str, Any]:
    """Preload and concatenate background and IPS features after subject alignment."""
    cache: dict[str, Any] = {}
    for feature_name in feature_names:
        bg_montage, bg_segment_length, bg_combiner = best_parameters_background[feature_name]
        ips_montage, ips_segment_length, ips_combiner = best_parameters_ips[feature_name]

        bg_data = load_feature_array(data_folder_bg, feature_name, bg_montage, bg_segment_length, bg_combiner)
        ips_data = load_feature_array(data_folder_ips, feature_name, ips_montage, ips_segment_length, ips_combiner)
        ips_data_reordered = ips_data[list(ips_reorder_indices)]
        combined_data = np.concatenate([bg_data, ips_data_reordered], axis=1)
        cache[feature_name] = array_converter(combined_data)

    return cache

def preload_triple_combined_feature_cache(
    feature_names: Sequence[str],
    best_parameters_background: dict[str, tuple[str, int, str]],
    best_parameters_ips: dict[str, tuple[str, int, str]],
    best_parameters_hv: dict[str, tuple[str, int, str]],
    data_folder_bg: str,
    data_folder_ips: str,
    data_folder_hv: str,
    ips_reorder_indices: Sequence[int],
    hv_reorder_indices: Sequence[int],
    array_converter: Callable[[np.ndarray], Any],
) -> dict[str, Any]:
    """Preload and concatenate background, IPS and HV features after subject alignment."""
    cache: dict[str, Any] = {}
    for feature_name in feature_names:
        bg_montage, bg_segment_length, bg_combiner = best_parameters_background[feature_name]
        ips_montage, ips_segment_length, ips_combiner = best_parameters_ips[feature_name]
        hv_montage, hv_segment_length, hv_combiner = best_parameters_hv[feature_name]

        bg_data = load_feature_array(data_folder_bg, feature_name, bg_montage, bg_segment_length, bg_combiner)
        ips_data = load_feature_array(data_folder_ips, feature_name, ips_montage, ips_segment_length, ips_combiner)
        hv_data = load_feature_array(data_folder_hv, feature_name, hv_montage, hv_segment_length, hv_combiner)

        ips_data_reordered = ips_data[list(ips_reorder_indices)]
        hv_data_reordered = hv_data[list(hv_reorder_indices)]

        combined_data = np.concatenate([bg_data, ips_data_reordered, hv_data_reordered], axis=1)
        cache[feature_name] = array_converter(combined_data)

    return cache

def build_subject_loso_splits(
    description_subjects: Sequence[Any],
    unique_subjects: Sequence[Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build subject-level LOSO train/test splits."""
    description_subjects = np.asarray(description_subjects)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for subject in unique_subjects:
        test_idxs = np.where(description_subjects == subject)[0]
        train_idxs = np.where(description_subjects != subject)[0]
        splits.append((train_idxs, test_idxs))
    return splits


def build_loocv_splits(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Legacy helper retained for compatibility."""
    all_indices = np.arange(n_samples)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(n_samples):
        test_idx = np.array([idx])
        train_idx = np.delete(all_indices, test_idx)
        splits.append((train_idx, test_idx))
    return splits


def build_train_val_test_indices(
    description: pd.DataFrame,
    labels: np.ndarray,
    subject: Any,
    split_ratio: float,
    seed: int,
    unique_subjects: Optional[Iterable[Any]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train/val/test indices for subject-level LOSO with stratified val split."""
    test_idxs = np.where(description["subject"] == subject)[0]
    subjects = description["subject"].to_numpy()

    if unique_subjects is None:
        unique_subjects = np.unique(subjects)

    other_subjects = [candidate for candidate in unique_subjects if candidate != subject]
    other_subjects_labels = np.array([[candidate, labels[subjects == candidate][0]] for candidate in other_subjects])

    train_subjects, val_subjects = train_test_split(
        other_subjects,
        test_size=split_ratio,
        stratify=other_subjects_labels[:, 1],
        random_state=seed,
    )

    train_idxs = np.where(np.isin(subjects, train_subjects))[0]
    val_idxs = np.where(np.isin(subjects, val_subjects))[0]
    return train_idxs, val_idxs, test_idxs


def generate_feature_combinations(
    feature_names: Sequence[str],
    min_len: int = 2,
    max_len: int = 10,
) -> list[list[str]]:
    """Generate feature combinations using the historical bounds behavior."""
    combinations: list[list[str]] = []
    for length in range(min_len, max_len):
        for combination in itertools.combinations(feature_names, length):
            combinations.append(list(combination))
    return combinations


def get_cached_feature_data(cache: dict[str, Any], feature_name: str):
    """Fetch one feature array from a preloaded cache."""
    return cache[feature_name]


def train_simplex_logistic(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 2500,
    alpha: float = 1.05,
) -> np.ndarray:
    """Fit simplex-constrained logistic regression weights."""
    dimensions = X.shape[1]
    init_w = np.full(dimensions, 1.0 / dimensions)

    def nll(weights):
        logits = X @ weights
        ce = -np.sum(y * np.log(expit(logits)) + (1 - y) * np.log(1 - expit(logits)))
        dirichlet_penalty = (alpha - 1) * -np.sum(np.log(weights + 1e-12))
        return ce + dirichlet_penalty

    bounds = [(0.0, None)] * dimensions
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    result = minimize(
        nll,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter},
    )

    if not result.success:
        raise RuntimeError(f"Simplex LR did not converge: {result.message}")
    return result.x


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the threshold that maximizes the geometric mean of TPR and TNR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmeans = np.sqrt(tpr * (1 - fpr))
    opt_index = np.argmax(gmeans)
    return thresholds[opt_index]


def calculate_bac(
    labels: np.ndarray,
    scores: np.ndarray,
    sens_thresh: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute BAC at the first threshold satisfying a target sensitivity."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid_idxs = np.where(tpr >= sens_thresh)[0]
    if len(valid_idxs) == 0:
        threshold_sensitivity = thresholds[-1] if len(thresholds) > 0 else 0.5
    else:
        threshold_sensitivity = thresholds[valid_idxs[0]]

    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    bac = (sensitivity + specificity) / 2
    return bac, fpr, tpr, thresholds


def safe_roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return ROC AUC when possible, otherwise NaN."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return average precision when possible, otherwise NaN."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(np.trapz(precision, recall))


def sanitize_component(component: str) -> str:
    """Make a filename-safe component while keeping experiment names readable."""
    return str(component).replace(os.sep, "_").replace(" ", "_")


def feature_set_to_string(feature_set: str | Sequence[str]) -> str:
    """Normalize a feature set identifier to one string."""
    if isinstance(feature_set, str):
        return feature_set
    return "+".join(feature_set)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Compute the common metrics used by both single-set and ensemble experiments."""
    bac = balanced_accuracy_score(y_true, y_pred)
    bac80, fpr, tpr, thresholds = calculate_bac(y_true, y_prob, 0.8)
    auc = safe_roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    auprc = float(np.trapz(precision_curve, recall_curve))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ap = safe_average_precision(y_true, y_prob)
    return {
        "auc": auc,
        "bac": float(bac),
        "bac80": float(bac80),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auprc": auprc,
        "ap": ap,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def build_run_dataframes(
    *,
    run_n: int,
    seed: int,
    run_name: str,
    experiment_mode: str,
    dataset_name: str,
    project_name: str,
    feature_set: str,
    prediction_rows: list[dict[str, Any]],
    summary_metrics: dict[str, float | np.ndarray],
    extra_summary: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build standardized predictions and summary DataFrames."""
    predictions_df = pd.DataFrame(prediction_rows)
    if predictions_df.empty:
        predictions_df = pd.DataFrame(
            columns=[
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
        )

    summary_row = {
        "project_name": project_name,
        "dataset_name": dataset_name,
        "mode": experiment_mode,
        "run": run_n,
        "seed": seed,
        "run_name": run_name,
        "feature_set": feature_set,
        "auc": summary_metrics["auc"],
        "bac": summary_metrics["bac"],
        "bac80": summary_metrics["bac80"],
        "accuracy": summary_metrics["accuracy"],
        "precision": summary_metrics["precision"],
        "recall": summary_metrics["recall"],
        "f1_score": summary_metrics["f1_score"],
        "auprc": summary_metrics["auprc"],
        "ap": summary_metrics["ap"],
    }
    if extra_summary:
        summary_row.update(extra_summary)

    return pd.DataFrame([summary_row]), predictions_df


def build_output_paths(log_folder: str, project_name: str, run_name: str, seed: int) -> tuple[str, str]:
    """Return the standardized summary and prediction file paths for one run."""
    output_dir = os.path.join(log_folder, project_name)
    safe_run_name = sanitize_component(run_name)
    summary_path = os.path.join(output_dir, f"summary__{safe_run_name}__seed_{seed}.csv")
    predictions_path = os.path.join(output_dir, f"predictions__{safe_run_name}__seed_{seed}.csv")
    return summary_path, predictions_path


def save_run_outputs(
    summary_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    log_folder: str,
    project_name: str,
    run_name: str,
    seed: int,
) -> tuple[str, str]:
    """Save standardized outputs for either a single-set or ensemble experiment."""
    summary_path, predictions_path = build_output_paths(log_folder, project_name, run_name, seed)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    return summary_path, predictions_path


def save_predictions_csv(
    y_pred: np.ndarray,
    y_score: np.ndarray,
    y_true: np.ndarray,
    output_dir: str,
    name_parts: tuple[str, str, int, str],
    run_n: int,
    seed: int,
) -> str:
    """Compatibility wrapper that preserves the standardized prediction saver."""
    montage, feature_name, segment_length, combiner = name_parts
    feature_set = f"{feature_name}|{montage}|{segment_length}|{combiner}"
    run_name = f"{feature_name}_{montage}_{segment_length}s_{combiner}_run_{run_n}"
    predictions_df = pd.DataFrame(
        {
            "project_name": os.path.basename(os.path.normpath(output_dir)),
            "dataset_name": "",
            "mode": "single_set",
            "run": run_n,
            "seed": seed,
            "run_name": run_name,
            "feature_set": feature_set,
            "fold_id": np.arange(len(y_true)),
            "subject_id": "",
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_score,
        }
    )
    output_dir = output_dir.rstrip("/\\")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"predictions__{sanitize_component(run_name)}__seed_{seed}.csv")
    predictions_df.to_csv(path, index=False)
    return path


def save_ensemble_results(
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    log_folder: str,
    project_name: str,
    run_name: str,
    run_n: int,
    seed: int,
) -> None:
    """Compatibility wrapper that uses the standardized saver paths."""
    if "run" not in results_df.columns:
        results_df = results_df.assign(run=run_n)
    if "seed" not in results_df.columns:
        results_df = results_df.assign(seed=seed)
    if "run_name" not in results_df.columns:
        results_df = results_df.assign(run_name=run_name)

    if "run" not in predictions_df.columns:
        predictions_df = predictions_df.assign(run=run_n)
    if "seed" not in predictions_df.columns:
        predictions_df = predictions_df.assign(seed=seed)
    if "run_name" not in predictions_df.columns:
        predictions_df = predictions_df.assign(run_name=run_name)

    save_run_outputs(results_df, predictions_df, log_folder, project_name, run_name, seed)


def should_skip_wandb_run(config: BaseExperimentConfig, run_name: str) -> bool:
    """Return True when a run with the same name already exists in wandb."""
    if not config.wandb_check_existing:
        return False

    api_path = config.wandb_check_path or config.project_name
    try:
        for existing_run in wandb.Api(timeout=config.wandb_timeout).runs(path=api_path):
            if existing_run.name == run_name:
                return True
    except Exception:
        return False
    return False


def maybe_log_run_metrics(
    summary_metrics: dict[str, float | np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """Log shared aggregate plots and metrics to wandb."""
    c_m = wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=["healthy", "epileptic"])
    roc_table = wandb.Table(
        data=[[fpr, tpr] for fpr, tpr in zip(summary_metrics["fpr"], summary_metrics["tpr"])],
        columns=["fpr", "tpr"],
    )
    roc_line = wandb.plot.line(roc_table, "fpr", "tpr", title="ROC Curve")

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_table = wandb.Table(
        data=[[precision, recall] for precision, recall in zip(precision_curve, recall_curve)],
        columns=["precision", "recall"],
    )
    pr_line = wandb.plot.line(pr_table, "precision", "recall", title="Precision-Recall Curve")

    wandb.log(
        {
            "auc": summary_metrics["auc"],
            "bac": summary_metrics["bac"],
            "bac80": summary_metrics["bac80"],
            "accuracy": summary_metrics["accuracy"],
            "precision": summary_metrics["precision"],
            "recall": summary_metrics["recall"],
            "f1_score": summary_metrics["f1_score"],
            "auprc": summary_metrics["auprc"],
            "ap": summary_metrics["ap"],
            "confusion_matrix": c_m,
            "roc_curve": roc_line,
            "precision_recall_curve": pr_line,
        }
    )


def log_metrics_to_wandb(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compatibility helper used by older scripts/tests."""
    summary_metrics = compute_classification_metrics(y_true, y_pred, y_score)
    maybe_log_run_metrics(summary_metrics, y_true, y_pred, y_score)
    score = float(summary_metrics["auc"]) + float(summary_metrics["bac80"])
    return (
        float(summary_metrics["bac"]),
        float(summary_metrics["bac80"]),
        float(summary_metrics["auc"]),
        score,
        float(summary_metrics["recall"]),
        float(summary_metrics["precision"]),
        float(summary_metrics["f1_score"]),
        float(summary_metrics["accuracy"]),
    )


def build_xgb_params(
    base_params: dict[str, Any],
    seed: int,
    y_train: np.ndarray,
    device: str,
    n_jobs_xgb: Optional[int] = None,
) -> dict[str, Any]:
    """Build one fold's XGBoost parameter dictionary."""
    params = dict(base_params)
    positives = int(np.sum(y_train))
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0
    params.update({"seed": seed, "scale_pos_weight": scale_pos_weight})

    if "device" not in params:
        params["device"] = device
    if n_jobs_xgb is not None and "n_jobs" not in params:
        params["n_jobs"] = n_jobs_xgb
    return params


def train_feature_model_parallel(
    args,
    *,
    cache: dict[str, Any],
    xgb_params: dict[str, Any],
    n_jobs_xgb: int,
    device: str,
):
    """Train one per-feature XGBoost model for an ensemble stage."""
    (
        feature_name,
        train_idxs,
        val_idxs,
        test_idxs,
        y_train,
        y_val,
        seed,
        retrain_on_trainval,
    ) = args

    np.random.seed(seed)
    data = cache[feature_name]

    if retrain_on_trainval:
        train_val_idxs = np.concatenate([train_idxs, val_idxs])
        y_train_val = np.concatenate([y_train, y_val])
        model = XGBClassifier(**build_xgb_params(xgb_params, seed, y_train_val, device, n_jobs_xgb))
        model.fit(data[train_val_idxs], y_train_val)
        test_probs = model.predict_proba(data[test_idxs])[:, 1]
        return {"feature_name": feature_name, "test_probs": test_probs, "model": model}

    model = XGBClassifier(**build_xgb_params(xgb_params, seed, y_train, device, n_jobs_xgb))
    model.fit(data[train_idxs], y_train)

    train_probs = model.predict_proba(data[train_idxs])[:, 1]
    val_probs = model.predict_proba(data[val_idxs])[:, 1]
    test_probs = model.predict_proba(data[test_idxs])[:, 1]
    auc = safe_roc_auc_score(y_val, val_probs)
    bac = balanced_accuracy_score(y_val, val_probs >= 0.5)
    bac80, _, _, _ = calculate_bac(y_val, val_probs, 0.8)

    return {
        "feature_name": feature_name,
        "train_probs": train_probs,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "auc": auc,
        "bac": float(bac),
        "bac80": float(bac80),
        "score": float(auc) + float(bac),
    }


def train_ensemble_models(
    feature_combination: Sequence[str],
    train_idxs: np.ndarray,
    val_idxs: np.ndarray,
    test_idxs: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    *,
    cache: dict[str, Any],
    xgb_params: dict[str, Any],
    n_jobs_xgb: int,
    device: str,
    n_parallel_features: int,
    simplex_alpha: float,
    log_stage1: Optional[Callable[[dict[str, Any]], None]] = None,
    log_stage2: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Train the shared two-stage ensemble pipeline for one LOSO fold."""
    args_list_stage1 = [
        (feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed + index, False)
        for index, feature_name in enumerate(feature_combination)
    ]
    feature_results_stage1 = Parallel(
        n_jobs=min(n_parallel_features, len(feature_combination)),
        backend="threading",
    )(
        delayed(train_feature_model_parallel)(
            args,
            cache=cache,
            xgb_params=xgb_params,
            n_jobs_xgb=n_jobs_xgb,
            device=device,
        )
        for args in args_list_stage1
    )

    ordered_stage1 = sorted(feature_results_stage1, key=lambda item: feature_combination.index(item["feature_name"]))
    val_probs_list = [item["val_probs"] for item in ordered_stage1]

    calibrated_probs = []
    calibrators = []
    for probs in val_probs_list:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrated = calibrator.fit_transform(probs, y_val)
        calibrated_probs.append(calibrated)
        calibrators.append(calibrator)

    meta_val_logits = np.column_stack([logit(np.clip(probs, 0.001, 0.999)) for probs in calibrated_probs])
    weights = train_simplex_logistic(meta_val_logits, y_val, alpha=simplex_alpha)
    meta_model = SimplexLogistic(weights)

    meta_val_probs = meta_model.predict_proba(meta_val_logits)[:, 1]
    stage1_auc = safe_roc_auc_score(y_val, meta_val_probs)
    stage1_bac = balanced_accuracy_score(y_val, meta_val_probs >= 0.5)
    stage1_bac80, _, _, _ = calculate_bac(y_val, meta_val_probs, 0.8)

    if log_stage1:
        log_stage1(
            {
                "auc": stage1_auc,
                "bac": float(stage1_bac),
                "bac80": float(stage1_bac80),
                "weights": weights,
                "meta_probs": meta_val_probs,
                "raw_probs": np.column_stack(val_probs_list),
                "calibrated_probs": np.column_stack(calibrated_probs),
                "logits": meta_val_logits,
            }
        )

    args_list_stage2 = [
        (feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed + index, True)
        for index, feature_name in enumerate(feature_combination)
    ]
    feature_results_stage2 = Parallel(
        n_jobs=min(n_parallel_features, len(feature_combination)),
        backend="threading",
    )(
        delayed(train_feature_model_parallel)(
            args,
            cache=cache,
            xgb_params=xgb_params,
            n_jobs_xgb=n_jobs_xgb,
            device=device,
        )
        for args in args_list_stage2
    )

    ordered_stage2 = sorted(feature_results_stage2, key=lambda item: feature_combination.index(item["feature_name"]))
    test_probs_list = [item["test_probs"] for item in ordered_stage2]
    calibrated_test_probs = [calibrator.transform(probs) for calibrator, probs in zip(calibrators, test_probs_list)]
    meta_test_logits = np.column_stack([logit(np.clip(probs, 0.001, 0.999)) for probs in calibrated_test_probs])
    meta_test_probs = meta_model.predict_proba(meta_test_logits)[:, 1]
    opt_threshold = find_optimal_threshold(y_val, meta_val_probs)
    meta_test_preds = (meta_test_probs >= opt_threshold).astype(int)

    if log_stage2:
        log_stage2(
            {
                "meta_probs": meta_test_probs,
                "meta_preds": meta_test_preds,
                "opt_threshold": opt_threshold,
                "raw_probs": np.column_stack(test_probs_list),
                "calibrated_probs": np.column_stack(calibrated_test_probs),
                "logits": meta_test_logits,
            }
        )

    return {
        "feature_models": ordered_stage2,
        "meta_model": meta_model,
        "calibrators": calibrators,
        "weights": weights,
        "lr_weights": weights,
        "val_probs": meta_val_probs,
        "test_probs": meta_test_probs,
        "test_preds": meta_test_preds,
        "opt_threshold": opt_threshold,
        "auc": stage1_auc,
        "bac": float(stage1_bac),
        "bac80": float(stage1_bac80),
    }


def filter_unique_subjects(unique_subjects: Sequence[Any], subjects_to_skip: Sequence[Any]) -> np.ndarray:
    """Filter excluded subjects from the unique subject list."""
    unique_subjects = np.asarray(unique_subjects)
    if not subjects_to_skip:
        return unique_subjects
    keep_mask = ~np.isin(unique_subjects, list(subjects_to_skip))
    return unique_subjects[keep_mask]


def apply_subject_filters(
    unique_subjects: Sequence[Any],
    include_subjects: Sequence[Any],
    subjects_to_skip: Sequence[Any],
) -> np.ndarray:
    """Apply inclusion and exclusion filters to the available subject list."""
    filtered = np.asarray(unique_subjects)
    if include_subjects:
        filtered = filtered[np.isin(filtered, list(include_subjects))]
    return filter_unique_subjects(filtered, subjects_to_skip)


def run_name_for_feature(config: BaseExperimentConfig, feature_set: str, run_n: int) -> str:
    """Format the configured run name."""
    return config.run_name_template.format(feature_set=feature_set, run_n=run_n)


def maybe_start_wandb_run(config: BaseExperimentConfig, run_name: str, payload: dict[str, Any]) -> None:
    """Start a wandb run when enabled."""
    if not config.log_to_wandb:
        return
    wandb.init(
        project=config.project_name,
        name=run_name,
        dir=config.log_folder,
        reinit=config.wandb_reinit,
    )
    wandb.config.update(payload)


def finish_wandb_run(config: BaseExperimentConfig) -> None:
    """Finish a wandb run when enabled."""
    if config.log_to_wandb:
        wandb.finish()


def run_single_set_experiment(config: SingleSetExperimentConfig) -> list[tuple[str, str]]:
    """Run a single-set experiment sweep with subject-level LOSO."""
    setup_environment(config.device, config.cuda_idx, config.wandb_key)
    description, labels, subjects, unique_subjects, _ = load_subject_data(config.data_folder)
    unique_subjects = apply_subject_filters(
        unique_subjects,
        config.include_subjects,
        config.subjects_to_skip,
    )

    saved_paths: list[tuple[str, str]] = []
    checkpoint_reached = config.checkpoint_run_name is None

    for montage, feature_name, segment_length, combiner in itertools.product(
        config.montages,
        config.feature_names,
        config.segment_lengths,
        config.combiners,
    ):
        feature_meta = {
            "montage": montage,
            "feature_name": feature_name,
            "segment_length": segment_length,
            "combiner": combiner,
        }
        if config.skip_feature_config and config.skip_feature_config(feature_meta):
            continue

        features = load_feature_array(config.data_folder, feature_name, montage, segment_length, combiner)
        feature_set = f"{feature_name}|{montage}|{segment_length}|{combiner}"

        for run_n in range(config.n_runs):
            run_name = run_name_for_feature(config, feature_set, run_n)
            if not checkpoint_reached:
                if run_name == config.checkpoint_run_name:
                    checkpoint_reached = True
                else:
                    continue

            seed = secrets.randbelow(config.random_seed_upper)
            np.random.seed(seed)

            if should_skip_wandb_run(config, run_name):
                continue

            maybe_start_wandb_run(
                config,
                run_name,
                {
                    "seed": seed,
                    "dataset_name": config.dataset_name,
                    "mode": "single_set",
                    "feature_name": feature_name,
                    "montage": montage,
                    "segment_length": segment_length,
                    "combiner": combiner,
                    **config.metadata,
                },
            )

            prediction_rows: list[dict[str, Any]] = []
            for fold_id, subject in enumerate(unique_subjects):
                train_idxs = np.where(subjects != subject)[0]
                test_idxs = np.where(subjects == subject)[0]
                y_train = labels[train_idxs].astype(int)
                y_test = labels[test_idxs].astype(int)

                model = XGBClassifier(**build_xgb_params(config.xgb_params, seed, y_train, config.device))
                model.fit(config.as_device_array(features[train_idxs]), config.as_device_array(y_train))

                y_pred = model.predict(config.as_device_array(features[test_idxs]))
                y_prob = model.predict_proba(config.as_device_array(features[test_idxs]))[:, 1]

                for index in range(len(y_test)):
                    prediction_rows.append(
                        {
                            "project_name": config.project_name,
                            "dataset_name": config.dataset_name,
                            "mode": "single_set",
                            "run": run_n,
                            "seed": seed,
                            "run_name": run_name,
                            "feature_set": feature_set,
                            "fold_id": fold_id,
                            "subject_id": subject,
                            "y_true": int(y_test[index]),
                            "y_pred": int(y_pred[index]),
                            "y_prob": float(y_prob[index]),
                            "montage": montage,
                            "feature_name": feature_name,
                            "segment_length": segment_length,
                            "combiner": combiner,
                        }
                    )

            predictions_df = pd.DataFrame(prediction_rows)
            summary_metrics = compute_classification_metrics(
                predictions_df["y_true"].to_numpy(),
                predictions_df["y_pred"].to_numpy(),
                predictions_df["y_prob"].to_numpy(),
            )
            summary_df, predictions_df = build_run_dataframes(
                run_n=run_n,
                seed=seed,
                run_name=run_name,
                experiment_mode="single_set",
                dataset_name=config.dataset_name,
                project_name=config.project_name,
                feature_set=feature_set,
                prediction_rows=prediction_rows,
                summary_metrics=summary_metrics,
                extra_summary={
                    "montage": montage,
                    "feature_name": feature_name,
                    "segment_length": segment_length,
                    "combiner": combiner,
                },
            )

            if config.log_to_wandb:
                maybe_log_run_metrics(
                    summary_metrics,
                    predictions_df["y_true"].to_numpy(),
                    predictions_df["y_pred"].to_numpy(),
                    predictions_df["y_prob"].to_numpy(),
                )

            saved_paths.append(
                save_run_outputs(summary_df, predictions_df, config.log_folder, config.project_name, run_name, seed)
            )
            finish_wandb_run(config)

    return saved_paths


def prepare_ensemble_dataset(
    config: EnsembleExperimentConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load subject metadata and a preloaded feature cache for an ensemble config."""
    if config.source_type == "single_source":
        description, labels, subjects, unique_subjects, _ = load_subject_data(config.data_folder or "")
        cache = preload_feature_cache(
            config.feature_names,
            config.best_parameters,
            config.data_folder or "",
            config.cache_array_converter,
        )
        return description, labels, subjects, unique_subjects, cache

    if config.source_type == "dual_source":
        data_folder_bg = config.data_folders["background"]
        data_folder_ips = config.data_folders["ips"]
        description, labels, subjects, unique_subjects, _, ips_reorder_indices = load_data_both_aligned(
            data_folder_bg,
            data_folder_ips,
        )
        cache = preload_combined_feature_cache(
            config.feature_names,
            config.best_parameters_by_source["background"],
            config.best_parameters_by_source["ips"],
            data_folder_bg,
            data_folder_ips,
            ips_reorder_indices,
            config.cache_array_converter,
        )
        return description, labels, subjects, unique_subjects, cache
    
    if config.source_type == "triple_source":
        data_folder_bg = config.data_folders["background"]
        data_folder_ips = config.data_folders["ips"]
        data_folder_hv = config.data_folders["hv"]
        description, labels, subjects, unique_subjects, _, ips_reorder_indices, hv_reorder_indices = load_data_triple_aligned(
            data_folder_bg,
            data_folder_ips,
            data_folder_hv,
        )
        cache = preload_triple_combined_feature_cache(
            config.feature_names,
            config.best_parameters_by_source["background"],
            config.best_parameters_by_source["ips"],
            config.best_parameters_by_source["hv"],
            data_folder_bg,
            data_folder_ips,
            data_folder_hv,
            ips_reorder_indices,
            hv_reorder_indices,
            config.cache_array_converter,
        )
        return description, labels, subjects, unique_subjects, cache

    raise ValueError(f"Unsupported source_type: {config.source_type}")


def run_ensemble_experiment(config: EnsembleExperimentConfig) -> list[tuple[str, str]]:
    """Run the shared two-stage ensemble experiment sweep."""
    setup_environment(config.device, config.cuda_idx, config.wandb_key)
    description, labels, subjects, unique_subjects, cache = prepare_ensemble_dataset(config)
    unique_subjects = apply_subject_filters(
        unique_subjects,
        config.include_subjects,
        config.subjects_to_skip,
    )

    all_combinations = generate_feature_combinations(
        config.feature_names,
        min_len=config.combination_min_len,
        max_len=config.combination_max_len,
    )

    saved_paths: list[tuple[str, str]] = []
    checkpoint_reached = config.checkpoint_run_name is None

    for combination in all_combinations:
        feature_set = feature_set_to_string(combination)
        for run_n in range(config.n_runs):
            run_name = run_name_for_feature(config, feature_set, run_n)
            if not checkpoint_reached:
                if run_name == config.checkpoint_run_name:
                    checkpoint_reached = True
                else:
                    continue

            seed = secrets.randbelow(config.random_seed_upper)
            np.random.seed(seed)

            if config.skip_existing_outputs:
                summary_path, _ = build_output_paths(config.log_folder, config.project_name, run_name, seed)
                if os.path.exists(summary_path):
                    continue

            if should_skip_wandb_run(config, run_name):
                continue

            maybe_start_wandb_run(
                config,
                run_name,
                {
                    "seed": seed,
                    "dataset_name": config.dataset_name,
                    "mode": "ensemble",
                    "feature_set": feature_set,
                    "combination_length": len(combination),
                    **config.metadata,
                },
            )

            prediction_rows: list[dict[str, Any]] = []
            for fold_id, subject in enumerate(unique_subjects):
                train_idxs, val_idxs, test_idxs = build_train_val_test_indices(
                    description,
                    labels,
                    subject,
                    config.split_ratio,
                    seed,
                    unique_subjects=unique_subjects,
                )
                y_train = labels[train_idxs].astype(int)
                y_val = labels[val_idxs].astype(int)
                y_test = labels[test_idxs].astype(int)

                def log_stage1(stage1: dict[str, Any]) -> None:
                    if not config.log_to_wandb:
                        return
                    wandb.log(
                        {
                            "stage1/auc": stage1["auc"],
                            "stage1/bac": stage1["bac"],
                            "stage1/bac80": stage1["bac80"],
                            "stage1/weights": wandb.Histogram(stage1["weights"]),
                            "stage1/meta_probs": wandb.Histogram(stage1["meta_probs"]),
                        }
                    )

                def log_stage2(stage2: dict[str, Any]) -> None:
                    if not config.log_to_wandb:
                        return
                    wandb.log(
                        {
                            "stage2/meta_probs": wandb.Histogram(stage2["meta_probs"]),
                            "stage2/meta_preds": wandb.Histogram(stage2["meta_preds"]),
                            "stage2/opt_threshold": stage2["opt_threshold"],
                        }
                    )

                ensemble_result = train_ensemble_models(
                    combination,
                    train_idxs,
                    val_idxs,
                    test_idxs,
                    y_train,
                    y_val,
                    seed,
                    cache=cache,
                    xgb_params=config.xgb_params,
                    n_jobs_xgb=config.n_jobs_xgb,
                    device=config.device,
                    n_parallel_features=config.n_parallel_features,
                    simplex_alpha=config.simplex_alpha,
                    log_stage1=log_stage1,
                    log_stage2=log_stage2,
                )

                if config.log_to_wandb:
                    wandb.log(
                        {
                            "validation/auc": ensemble_result["auc"],
                            "validation/bac": ensemble_result["bac"],
                            "validation/bac80": ensemble_result["bac80"],
                            "validation/opt_threshold": ensemble_result["opt_threshold"],
                            "validation/weights": wandb.Histogram(ensemble_result["weights"]),
                        }
                    )

                for index in range(len(y_test)):
                    prediction_rows.append(
                        {
                            "project_name": config.project_name,
                            "dataset_name": config.dataset_name,
                            "mode": "ensemble",
                            "run": run_n,
                            "seed": seed,
                            "run_name": run_name,
                            "feature_set": feature_set,
                            "fold_id": fold_id,
                            "subject_id": subject,
                            "y_true": int(y_test[index]),
                            "y_pred": int(ensemble_result["test_preds"][index]),
                            "y_prob": float(ensemble_result["test_probs"][index]),
                        }
                    )

            predictions_df = pd.DataFrame(prediction_rows)
            summary_metrics = compute_classification_metrics(
                predictions_df["y_true"].to_numpy(),
                predictions_df["y_pred"].to_numpy(),
                predictions_df["y_prob"].to_numpy(),
            )
            summary_df, predictions_df = build_run_dataframes(
                run_n=run_n,
                seed=seed,
                run_name=run_name,
                experiment_mode="ensemble",
                dataset_name=config.dataset_name,
                project_name=config.project_name,
                feature_set=feature_set,
                prediction_rows=prediction_rows,
                summary_metrics=summary_metrics,
                extra_summary={"combination_length": len(combination)},
            )

            if config.log_to_wandb:
                maybe_log_run_metrics(
                    summary_metrics,
                    predictions_df["y_true"].to_numpy(),
                    predictions_df["y_pred"].to_numpy(),
                    predictions_df["y_prob"].to_numpy(),
                )

            saved_paths.append(
                save_run_outputs(summary_df, predictions_df, config.log_folder, config.project_name, run_name, seed)
            )
            finish_wandb_run(config)

    return saved_paths


def train_xgb_folds(
    features: np.ndarray,
    labels: np.ndarray,
    split_iterator: Iterable[tuple[np.ndarray, np.ndarray]],
    seed: int,
    model_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper retained for older tests and scripts."""
    y_preds = []
    y_scores = []
    y_tests = []

    as_device_array = model_params.pop("as_device_array", identity)
    append_mode = model_params.pop("append_mode", "extend")

    for train_idxs, test_idxs in split_iterator:
        y_train = labels[train_idxs].astype(int)
        y_test = labels[test_idxs].astype(int)
        model = XGBClassifier(**build_xgb_params(model_params, seed, y_train, model_params.get("device", "cpu")))
        model.fit(as_device_array(features[train_idxs]), as_device_array(labels[train_idxs]))

        y_pred = model.predict(as_device_array(features[test_idxs]))
        y_score = model.predict_proba(as_device_array(features[test_idxs]))[:, 1]

        if append_mode == "append":
            y_preds.append(y_pred)
            y_scores.append(y_score)
            y_tests.append(y_test)
        else:
            y_preds.extend(y_pred)
            y_scores.extend(y_score)
            y_tests.extend(y_test)

    return np.array(y_preds).flatten(), np.array(y_scores).flatten(), np.array(y_tests).flatten()


__all__ = [
    "BaseExperimentConfig",
    "EnsembleExperimentConfig",
    "SimplexLogistic",
    "SingleSetExperimentConfig",
    "build_loocv_splits",
    "build_output_paths",
    "build_subject_loso_splits",
    "build_train_val_test_indices",
    "calculate_bac",
    "compute_classification_metrics",
    "find_optimal_threshold",
    "generate_feature_combinations",
    "get_cached_feature_data",
    "handle_complex_numbers",
    "load_data_both_aligned",
    "load_feature_array",
    "load_feature_from_best_params",
    "load_subject_data",
    "log_metrics_to_wandb",
    "preload_combined_feature_cache",
    "preload_feature_cache",
    "run_ensemble_experiment",
    "run_single_set_experiment",
    "save_ensemble_results",
    "save_predictions_csv",
    "save_run_outputs",
    "setup_environment",
    "train_ensemble_models",
    "train_feature_model_parallel",
    "train_simplex_logistic",
    "train_xgb_folds",
]
