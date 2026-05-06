import os
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import wandb
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
from xgboost import XGBClassifier


def setup_environment(cuda_idx: int, wandb_key: Optional[str] = None) -> None:
    """Initialize CUDA device and wandb authentication."""
    from cupy.cuda import Device

    Device(cuda_idx).use()
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        # Attempt to fetch the key from environment variables
        env_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=env_key)


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


def calculate_bac(
    labels: np.ndarray, scores: np.ndarray, sens_thresh: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate balanced accuracy at the first threshold satisfying sensitivity."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid_idxs = np.where(tpr >= sens_thresh)[0]
    if len(valid_idxs) == 0:
        threshold_sensitivity = thresholds[-1] if len(thresholds) > 0 else 0.5
    else:
        threshold_sensitivity = thresholds[valid_idxs[0]]

    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    bac = (sensitivity + specificity) / 2
    return bac, fpr, tpr, thresholds


def handle_complex_numbers(features: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """Convert complex values and sanitize non-finite entries."""
    if isinstance(features, pd.DataFrame):
        for column in features.columns:
            if np.iscomplexobj(features[column]):
                features[column] = features[column].apply(np.abs)
            features[column].replace([np.inf, -np.inf], np.nan, inplace=True)
        return features

    if np.iscomplexobj(features):
        features = np.abs(features)
    features[~np.isfinite(features)] = np.nan
    return features


def load_feature_array(
    base_folder: str,
    feature_name: str,
    montage: str,
    segment_length: int,
    combiner: str,
) -> np.ndarray:
    """Load a feature array and sanitize it."""
    features = np.load(f"{base_folder}{feature_name}_{montage}_{segment_length}s_{combiner}.npy")
    return handle_complex_numbers(features)


def build_subject_loso_splits(
    description_subjects: Sequence,
    unique_subjects: Sequence,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build subject-level LOSO folds."""
    description_subjects = np.asarray(description_subjects)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for subject in unique_subjects:
        test_idxs = np.where(description_subjects == subject)[0]
        train_idxs = np.where(description_subjects != subject)[0]
        splits.append((train_idxs, test_idxs))
    return splits


def build_loocv_splits(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build sample-level leave-one-out folds."""
    all_indices = np.arange(n_samples)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(n_samples):
        test_idx = np.array([idx])
        train_idx = np.delete(all_indices, test_idx)
        splits.append((train_idx, test_idx))
    return splits


def train_xgb_folds(
    features,
    labels,
    split_iterator: Iterable[tuple[np.ndarray, np.ndarray]],
    seed,
    model_params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/evaluate XGBoost across precomputed folds."""
    y_preds = []
    y_scores = []
    y_tests = []

    as_device_array: Callable[[np.ndarray], np.ndarray] = model_params.pop(
        "as_device_array", lambda x: x
    )
    append_mode = model_params.pop("append_mode", "extend")

    for train_idxs, test_idxs in split_iterator:
        y_train = labels[train_idxs].astype(int)
        y_test = labels[test_idxs].astype(int)

        ratio = (len(y_train) - sum(y_train)) / sum(y_train)
        fold_params = dict(model_params)
        fold_params.update({"seed": seed, "scale_pos_weight": ratio})

        model = XGBClassifier(**fold_params)
        model.fit(as_device_array(features[train_idxs]), as_device_array(labels[train_idxs]))

        print("Training data shape:", features[train_idxs].shape)
        print("Test data shape:", features[test_idxs].shape)

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

    return (
        np.array(y_preds).flatten(),
        np.array(y_scores).flatten(),
        np.array(y_tests).flatten(),
    )


def log_metrics_to_wandb(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute and log common single-set classification metrics."""
    bac = balanced_accuracy_score(y_true, y_pred)
    bac80, fpr, tpr, _ = calculate_bac(y_true, y_score, 0.8)
    auc = roc_auc_score(y_true, y_score)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    score = auc + bac80

    c_m = wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=["healthy", "epileptic"])

    data_roc = [[f, t] for (f, t) in zip(fpr, tpr)]
    table_roc = wandb.Table(data=data_roc, columns=["fpr", "tpr"])
    roc_line = wandb.plot.line(table_roc, "fpr", "tpr", title="ROC Curve")

    p, r, _ = precision_recall_curve(y_true, y_score)
    data_pr = [[f, t] for (f, t) in zip(p, r)]
    table_pr = wandb.Table(data=data_pr, columns=["precision", "recall"])
    pr_line = wandb.plot.line(table_pr, "precision", "recall", title="Precision-Recall Curve")

    wandb.log(
        {
            "BAC": bac,
            "BAC80": bac80,
            "AUC": auc,
            "Score": score,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "Confusion Matrix": c_m,
            "ROC Curve": roc_line,
            "Precision-Recall Curve": pr_line,
            "Accuracy": accuracy,
        }
    )

    return bac, bac80, auc, score, recall, precision, f1, accuracy


def save_predictions_csv(
    y_pred: np.ndarray,
    y_score: np.ndarray,
    y_true: np.ndarray,
    output_dir: str,
    name_parts: tuple[str, str, int, str],
    run_n: int,
    seed: int,
) -> str:
    """Persist prediction triplets in the historical naming format."""
    df = pd.DataFrame({"y_preds": y_pred, "y_scores": y_score, "y_tests": y_true})

    os.makedirs(output_dir, exist_ok=True)
    montage, feature_name, segment_length, combiner = name_parts
    filename = (
        f"{output_dir}predictions_{montage}_{feature_name}_{segment_length}s_"
        f"{combiner}_run_{run_n}_seed_{seed}.csv"
    )
    df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")
    return filename
