import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import confusion_matrix, roc_curve


def train_simplex_logistic(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 2500,
    alpha: float = 1.05,
) -> np.ndarray:
    """Fit simplex-constrained logistic weights with optional Dirichlet prior."""
    d = X.shape[1]
    init_w = np.full(d, 1.0 / d)

    def nll(w):
        logits = X @ w
        ce = -np.sum(y * np.log(expit(logits)) + (1 - y) * np.log(1 - expit(logits)))
        dirichlet_pen = (alpha - 1) * -np.sum(np.log(w + 1e-12))
        return ce + dirichlet_pen

    bounds = [(0.00, None)] * d
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    res = minimize(
        nll,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter},
    )

    if not res.success:
        raise RuntimeError("Simplex LR did not converge: " + res.message)

    return res.x


class SimplexLogistic:
    """Tiny wrapper so ensemble pipelines keep working unchanged."""

    def __init__(self, w: np.ndarray):
        self.coef_ = w[None, :]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = expit(X @ self.coef_.ravel())
        return np.column_stack([1 - p, p])


def setup_environment(device: str, cuda_idx: int, wandb_key: Optional[str] = None) -> None:
    """Initialize CUDA device (when enabled) and wandb authentication."""
    if device != "cpu":
        from cupy.cuda import Device

        Device(cuda_idx).use()

    import wandb

    if wandb_key:
        wandb.login(key=wandb_key)
    
    else:
        #fetch from .bashrc 
        
        wandb.login(key=os.getenv("WANDB_API_KEY"))


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes geometric mean of sensitivity and specificity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmeans = np.sqrt(tpr * (1 - fpr))
    opt_index = np.argmax(gmeans)
    return thresholds[opt_index]


def calculate_bac(
    labels: np.ndarray,
    scores: np.ndarray,
    sens_thresh: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate balanced accuracy at first threshold reaching target sensitivity."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold_sensitivity = thresholds[np.where(tpr >= sens_thresh)[0][0]]
    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bac = (sensitivity + specificity) / 2
    return bac, fpr, tpr, thresholds


def handle_complex_numbers(features: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """Convert complex values and sanitize infinities."""
    if isinstance(features, pd.DataFrame):
        for column in features.columns:
            if np.iscomplexobj(features[column]):
                features[column] = features[column].apply(np.abs)
            features[column].replace([np.inf, -np.inf], np.nan, inplace=True)
    elif isinstance(features, np.ndarray):
        if np.iscomplexobj(features):
            features = np.abs(features)
        features[~np.isfinite(features)] = np.nan
    return features


def load_subject_data(path: str):
    """Load description metadata and return labels/subject arrays."""
    description = pd.read_csv(f"{path}/description.csv")
    labels = description["epilepsy"].to_numpy()
    subjects = description["subject"].to_numpy()
    unique_subjects = np.unique(description["subject"])

    subject_labels = []
    for subj in unique_subjects:
        lbl = labels[subjects == subj][0]
        subject_labels.append([subj, lbl])
    subject_labels = np.array(subject_labels)

    return description, labels, subjects, unique_subjects, subject_labels


def load_feature_from_best_params(
    data_folder: str,
    best_parameters: dict,
    feature_name: str,
    array_converter,
):
    """Load one feature matrix according to per-feature best parameters."""
    montage, segment_length, combiner = best_parameters[feature_name]
    data = np.load(f"{data_folder}/{feature_name}_{montage}_{segment_length}s_{combiner}.npy")
    data = handle_complex_numbers(data)

    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    return array_converter(data), montage, segment_length, combiner


def preload_feature_cache(
    feature_names,
    best_parameters,
    data_folder: str,
    array_converter,
):
    """Preload all feature matrices into a dictionary cache."""
    print("Preloading all feature data...")
    cache = {}

    for feature_name in feature_names:
        montage, segment_length, combiner = best_parameters[feature_name]
        data_path = f"{data_folder}/{feature_name}_{montage}_{segment_length}s_{combiner}.npy"
        data = np.load(data_path)
        data = handle_complex_numbers(data)

        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)

        cache[feature_name] = array_converter(data)
        print(f"Loaded {feature_name}: {data.shape}")

    return cache


def load_data_both_aligned(data_folder_bg: str, data_folder_ips: str):
    """Load background+IPS data and align IPS order to background subjects."""
    bg_description = pd.read_csv(f"{data_folder_bg}/description.csv")
    bg_labels = bg_description["epilepsy"].to_numpy()
    bg_subjects = bg_description["subject"].to_numpy()
    bg_unique_subjects = np.unique(bg_description["subject"])

    ips_description = pd.read_csv(f"{data_folder_ips}/description.csv")
    ips_labels = ips_description["epilepsy"].to_numpy()
    ips_subjects = ips_description["subject"].to_numpy()

    common_subjects = np.intersect1d(bg_unique_subjects, np.unique(ips_subjects))
    print(f"Found {len(common_subjects)} common subjects between background and IPS data")
    print(f"Background subjects: {len(bg_unique_subjects)}, IPS subjects: {len(np.unique(ips_subjects))}")

    bg_mask = np.isin(bg_subjects, common_subjects)
    filtered_bg_description = bg_description[bg_mask].reset_index(drop=True)
    filtered_bg_labels = bg_labels[bg_mask]
    filtered_bg_subjects = bg_subjects[bg_mask]

    ips_mask = np.isin(ips_subjects, common_subjects)
    filtered_ips_description = ips_description[ips_mask].reset_index(drop=True)
    filtered_ips_labels = ips_labels[ips_mask]
    filtered_ips_subjects = ips_subjects[ips_mask]

    assert np.array_equal(np.sort(filtered_bg_subjects), np.sort(filtered_ips_subjects)), (
        "Subject mismatch between datasets"
    )
    assert np.array_equal(
        filtered_bg_labels[np.argsort(filtered_bg_subjects)],
        filtered_ips_labels[np.argsort(filtered_ips_subjects)],
    ), "Label mismatch between datasets"

    print(f"Sanity check - Filtered background data: {len(filtered_bg_subjects)} samples")
    print(f"Sanity check - Filtered IPS data: {len(filtered_ips_subjects)} samples")
    print(f"Sanity check - Background epilepsy ratio: {np.mean(filtered_bg_labels):.3f}")
    print(f"Sanity check - IPS epilepsy ratio: {np.mean(filtered_ips_labels):.3f}")

    ips_reorder_indices = [np.where(filtered_ips_subjects == subj)[0][0] for subj in filtered_bg_subjects]

    subject_labels = []
    for subj in common_subjects:
        lbl = filtered_bg_labels[filtered_bg_subjects == subj][0]
        subject_labels.append([subj, lbl])
    subject_labels = np.array(subject_labels)

    return (
        filtered_bg_description,
        filtered_bg_labels,
        filtered_bg_subjects,
        common_subjects,
        subject_labels,
        ips_reorder_indices,
    )


def preload_combined_feature_cache(
    feature_names,
    best_parameters_background,
    best_parameters_ips,
    data_folder_bg: str,
    data_folder_ips: str,
    ips_reorder_indices,
    array_converter,
):
    """Preload and concatenate background+IPS features by aligned subject order."""
    print("Preloading and concatenating IPS and background feature data...")
    cache = {}

    for feature_name in feature_names:
        bg_montage, bg_segment_length, bg_combiner = best_parameters_background[feature_name]
        bg_data_path = (
            f"{data_folder_bg}/{feature_name}_{bg_montage}_{bg_segment_length}s_{bg_combiner}.npy"
        )
        bg_data = np.load(bg_data_path)
        bg_data = handle_complex_numbers(bg_data)
        if len(bg_data.shape) > 2:
            bg_data = bg_data.reshape(bg_data.shape[0], -1)

        ips_montage, ips_segment_length, ips_combiner = best_parameters_ips[feature_name]
        ips_data_path = (
            f"{data_folder_ips}/{feature_name}_{ips_montage}_{ips_segment_length}s_{ips_combiner}.npy"
        )
        ips_data = np.load(ips_data_path)
        ips_data = handle_complex_numbers(ips_data)
        if len(ips_data.shape) > 2:
            ips_data = ips_data.reshape(ips_data.shape[0], -1)

        print(f"Sanity check - {feature_name}:")
        print(f"  Background original shape: {bg_data.shape}")
        print(f"  IPS original shape: {ips_data.shape}")

        ips_data_reordered = ips_data[ips_reorder_indices]
        print(f"  IPS reordered shape: {ips_data_reordered.shape}")

        combined_data = np.concatenate([bg_data, ips_data_reordered], axis=1)
        print(f"  Combined shape: {combined_data.shape}")

        cache[feature_name] = array_converter(combined_data)
        print(f"Loaded and combined {feature_name}: {combined_data.shape}")

    return cache


def save_ensemble_results(
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    log_folder: str,
    project_name: str,
    run_name: str,
    run_n: int,
    seed: int,
) -> None:
    """Persist ensemble results and predictions using historical naming."""
    os.makedirs(f"{log_folder}/{project_name}", exist_ok=True)
    results_df.to_csv(
        f"{log_folder}/{project_name}/{run_name}_run_{run_n}_results_seed_{seed}.csv",
        index=False,
    )
    predictions_df.to_csv(
        f"{log_folder}/{project_name}/{run_name}_run_{run_n}_predictions_seed_{seed}.csv",
        index=False,
    )
