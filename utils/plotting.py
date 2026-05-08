from __future__ import annotations

import itertools
import os
import pickle
from datetime import datetime
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy import interpolate as scipy_interpolate
from scipy.stats import mannwhitneyu, t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
)
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier


def set_plot_style(style: str = "ggplot") -> None:
    plt.style.use(style)

def handle_complex_numbers(features: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
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


def load_subject_data(data_folder: str):
    description = pd.read_csv(os.path.join(data_folder, "description.csv"))
    labels = description["epilepsy"].to_numpy()
    subjects = description["subject"].to_numpy()
    unique_subjects = np.unique(subjects)

    subject_labels = []
    for subject in unique_subjects:
        label = labels[subjects == subject][0]
        subject_labels.append([subject, label])

    return description, labels, subjects, unique_subjects, np.array(subject_labels)


def load_feature_data_from_best_params(
    data_folder: str,
    best_parameters: dict[str, tuple[str, int, str]],
    feature_name: str,
):
    montage, segment_length, combiner = best_parameters[feature_name]
    data_path = os.path.join(data_folder, f"{feature_name}_{montage}_{segment_length}s_{combiner}.npy")
    data = np.load(data_path)
    data = handle_complex_numbers(data)
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    return data, montage, segment_length, combiner


def interpolate(p1_fpr: float, p1_tpr: float, p2_fpr: float, p2_tpr: float, x: float) -> float:
    slope = (p2_tpr - p1_tpr) / (p2_fpr - p1_fpr)
    return p1_tpr + slope * (x - p1_fpr)


def tpr_for_fpr(fprsample: float, fpr_arr: np.ndarray, tpr_arr: np.ndarray) -> float:
    i = np.searchsorted(fpr_arr, fprsample, side="right") - 1
    if i < 0:
        return float(tpr_arr[0])
    if fpr_arr[i] == fprsample or i == len(fpr_arr) - 1:
        return float(tpr_arr[i])
    return float(interpolate(fpr_arr[i], tpr_arr[i], fpr_arr[i + 1], tpr_arr[i + 1], fprsample))


def vertical_avg_roc(roc_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    fprs = np.concatenate([fpr for fpr, _, _ in roc_data])

    if fprs[0] != 0:
        fprs = np.insert(fprs, 0, 0)
    if fprs[-1] != 1:
        fprs = np.append(fprs, 1)

    fprs = np.sort(np.unique(fprs))
    tprs = np.zeros(shape=(len(roc_data), len(fprs)))

    for i, fpr in enumerate(fprs):
        for curve_idx, (fpr_arr, tpr_arr, _) in enumerate(roc_data):
            tprs[curve_idx, i] = tpr_for_fpr(float(fpr), fpr_arr, tpr_arr)

    return fprs, np.mean(tprs, axis=0)


def calculate_bac(labels: np.ndarray, scores: np.ndarray, sens_thresh: float):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold_sensitivity = thresholds[np.where(tpr >= sens_thresh)[0][0]]
    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bac = (sensitivity + specificity) / 2
    return bac, fpr, tpr, thresholds


def compute_threshold_slope(
    n_epileptic: int,
    n_healthy: int,
    posterior_threshold: float = 0.6,
) -> float:
    p_epileptic = n_epileptic / (n_epileptic + n_healthy)
    p_healthy = 1 - p_epileptic
    return (posterior_threshold * p_healthy) / (p_epileptic * (1 - posterior_threshold))


def analyze_ensemble_performance(
    datafolder: str,
    feature_names: Optional[list[str]] = None,
    save_plots: bool = True,
    plot_prefix: str = "ensemble",
    length_to_save: Optional[int] = None,
    filename_for_saving: Optional[str] = None,
    save_dir: Optional[str] = None,
    colors: Optional[list] = None,
):
    if feature_names is None:
        feature_names = ["cc", "cwt", "dwt", "plv", "mst", "sst", "spectral", "utm", "gcc", "gplv"]

    files = [file for file in os.listdir(datafolder) if file.endswith(".csv")]
    if not files:
        return None, None, None

    combinations = []
    comb_names = []
    for i in range(2, len(feature_names) + 1):
        for comb in itertools.combinations(feature_names, i):
            combinations.append(list(comb))
            comb_names.append("+".join(list(comb)))

    df_summary = pd.DataFrame(
        columns=[
            "combination",
            "length",
            "accuracy",
            "bac",
            "bac80",
            "f1_score",
            "precision",
            "recall",
            "auc",
            "auprc",
            "AP",
        ]
    )

    for comb, name in zip(combinations, comb_names):
        comb_files = [file for file in files if file.startswith(name + "_") and "predictions" in file]
        for file in comb_files:
            df_file = pd.read_csv(os.path.join(datafolder, file))
            accuracy = accuracy_score(df_file["y_true"], df_file["y_pred"])
            f1 = f1_score(df_file["y_true"], df_file["y_pred"])
            precision = precision_score(df_file["y_true"], df_file["y_pred"])
            recall = recall_score(df_file["y_true"], df_file["y_pred"])
            auc = roc_auc_score(df_file["y_true"], df_file["y_prob"])
            bac = balanced_accuracy_score(df_file["y_true"], df_file["y_pred"])
            bac80, _, _, _ = calculate_bac(df_file["y_true"], df_file["y_prob"], 0.80)
            p, r, _ = precision_recall_curve(df_file["y_true"], df_file["y_prob"])
            auprc = sk_auc(r, p)
            ap = average_precision_score(df_file["y_true"], df_file["y_prob"])

            df_summary = pd.concat(
                [
                    df_summary,
                    pd.DataFrame(
                        [[name, len(comb), accuracy, bac, bac80, f1, precision, recall, auc, auprc, ap]],
                        columns=df_summary.columns,
                    ),
                ],
                ignore_index=True,
            )

    avg_df = df_summary.groupby("combination").mean(numeric_only=True).reset_index()
    avg_df["length"] = avg_df["combination"].apply(lambda x: len(x.split("+")))

    df_best = pd.DataFrame(
        columns=["length", "combination", "accuracy", "bac", "bac80", "f1_score", "precision", "recall", "auc", "auprc"]
    )

    for length in range(2, len(feature_names) + 1):
        df_length = avg_df[avg_df["length"] == length].copy()
        if df_length.empty:
            continue
        df_length.loc[:, "metric"] = df_length["auc"] + df_length["auprc"]
        best_combination = df_length.loc[df_length["metric"].idxmax()]
        df_best = pd.concat([df_best, pd.DataFrame([best_combination], columns=df_best.columns)], ignore_index=True)

    std_df = df_summary.groupby("combination").std(numeric_only=True).reset_index()
    std_df["length"] = std_df["combination"].apply(lambda x: len(x.split("+")))
    keep = df_best["combination"].tolist()
    std_df = std_df[std_df["combination"].isin(keep)].sort_values(by="length").reset_index(drop=True)

    if colors is None:
        colors = list(plt.cm.tab10.colors) * 5

    plt.figure(figsize=(8, 8))
    last_df_file = None
    for index, row in df_best.iterrows():
        comb = row["combination"]
        comb_files = [file for file in files if file.startswith(comb + "_") and "predictions" in file]
        roc_data = []

        for file in comb_files:
            df_file = pd.read_csv(os.path.join(datafolder, file))
            last_df_file = df_file
            fpr, tpr, thresholds = roc_curve(df_file["y_true"], df_file["y_prob"])
            roc_data.append((fpr, tpr, thresholds))

        if not roc_data:
            continue

        fpr_avg, tpr_avg = vertical_avg_roc(roc_data)
        auc_avg = sk_auc(fpr_avg, tpr_avg)
        if length_to_save is not None and filename_for_saving and save_dir and row["length"] == length_to_save:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, f"{filename_for_saving}.npz"), fpr=fpr_avg, tpr=tpr_avg, auc=auc_avg)

        plt.step(
            fpr_avg,
            tpr_avg,
            label=f"{comb} (AUC = {row['auc']:.2f})",
            color=colors[index % len(colors)],
            alpha=0.7,
        )

    if last_df_file is not None:
        n_epileptic = int(np.sum(last_df_file["y_true"]))
        n_healthy = int(len(last_df_file["y_true"]) - n_epileptic)
        slope = compute_threshold_slope(n_epileptic, n_healthy, 0.6)
        plt.plot([0, 1], [0, slope], "--", color="green", alpha=0.5)
        plt.fill_between([0, 1], [0, slope], [1, slope], color="green", alpha=0.1, label="P(Epilepsy|y=1)$\\geq$0.6")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc="lower right", fontsize=9)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves of Ensemble Models")
    plt.grid(True)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_prefix}_roc_curves.pdf", dpi=300)
    plt.show()

    return df_summary, df_best, std_df


def analyze_single_set_performance(
    datafolder: str,
    feature_types: list[str],
    feature_display_names: dict[str, str],
    save_plots: bool = True,
    plot_prefix: str = "single_set",
    type_to_save: Optional[str] = None,
    save_dir: Optional[str] = None,
    colors: Optional[list] = None,
):
    files = [file for file in os.listdir(datafolder) if file.endswith(".csv")]
    if not files:
        return None, None, None

    df_summary = pd.DataFrame(
        columns=[
            "feature_type",
            "montage",
            "length",
            "combiner",
            "accuracy",
            "bac",
            "bac80",
            "f1_score",
            "precision",
            "recall",
            "auc",
            "auprc",
            "AP",
        ]
    )

    for feature_type in feature_types:
        feature_files = [file for file in files if f"_{feature_type}_" in file and "predictions" in file]
        for file in feature_files:
            try:
                parts = file.replace("predictions_", "").replace(".csv", "").split("_")
                feature_idx = next(i for i, part in enumerate(parts) if part == feature_type)
                montage = parts[0] if feature_idx > 0 else "unknown"
                length_part = parts[feature_idx + 1] if feature_idx + 1 < len(parts) else "unknown"
                length = length_part.replace("s", "") if length_part.endswith("s") else length_part
                combiner = parts[feature_idx + 2] if feature_idx + 2 < len(parts) else "unknown"

                df_file = pd.read_csv(os.path.join(datafolder, file))
                accuracy = accuracy_score(df_file["y_tests"], df_file["y_preds"])
                f1 = f1_score(df_file["y_tests"], df_file["y_preds"])
                precision = precision_score(df_file["y_tests"], df_file["y_preds"])
                recall = recall_score(df_file["y_tests"], df_file["y_preds"])
                auc = roc_auc_score(df_file["y_tests"], df_file["y_scores"])
                bac = balanced_accuracy_score(df_file["y_tests"], df_file["y_preds"])
                bac80, _, _, _ = calculate_bac(df_file["y_tests"], df_file["y_scores"], 0.80)
                p, r, _ = precision_recall_curve(df_file["y_tests"], df_file["y_scores"])
                auprc = sk_auc(r, p)
                ap = average_precision_score(df_file["y_tests"], df_file["y_scores"])

                df_summary = pd.concat(
                    [
                        df_summary,
                        pd.DataFrame(
                            [[feature_type, montage, length, combiner, accuracy, bac, bac80, f1, precision, recall, auc, auprc, ap]],
                            columns=df_summary.columns,
                        ),
                    ],
                    ignore_index=True,
                )
            except Exception:
                continue

    if df_summary.empty:
        return None, None, None

    param_combinations = (
        df_summary.groupby(["feature_type", "montage", "length", "combiner"])
        .agg(
            {
                "accuracy": "mean",
                "bac": "mean",
                "bac80": "mean",
                "f1_score": "mean",
                "precision": "mean",
                "recall": "mean",
                "auc": "mean",
                "auprc": "mean",
                "AP": "mean",
            }
        )
        .reset_index()
    )

    df_best = []
    for feature_type in df_summary["feature_type"].unique():
        feature_data = param_combinations[param_combinations["feature_type"] == feature_type].copy()
        feature_data["score"] = feature_data["auc"] + feature_data["bac80"]
        if feature_data.empty:
            continue
        best_idx = feature_data["score"].idxmax()
        df_best.append(feature_data.loc[best_idx].copy())

    df_best = pd.DataFrame(df_best)
    if df_best.empty:
        return None, None, None

    df_best = df_best.sort_values(by="auc", ascending=False).reset_index(drop=True)

    std_df_rows = []
    for _, row in df_best.iterrows():
        combo_data = df_summary[
            (df_summary["feature_type"] == row["feature_type"])
            & (df_summary["montage"] == row["montage"])
            & (df_summary["length"] == row["length"])
            & (df_summary["combiner"] == row["combiner"])
        ]
        if combo_data.empty:
            continue
        std_row = combo_data.std(numeric_only=True)
        std_row["feature_type"] = row["feature_type"]
        std_row["montage"] = row["montage"]
        std_row["length"] = row["length"]
        std_row["combiner"] = row["combiner"]
        std_df_rows.append(std_row)

    std_df = pd.DataFrame(std_df_rows)

    if colors is None:
        colors = list(plt.cm.tab10.colors) * 5

    plt.figure(figsize=(8, 8))
    plot_index = 0
    last_df_file = None

    for feature_type in feature_types:
        feature_best = df_best[df_best["feature_type"] == feature_type]
        if feature_best.empty:
            continue

        row = feature_best.iloc[0]
        montage = row["montage"]
        length = row["length"]
        combiner = row["combiner"]

        pattern_files = []
        for file in files:
            if (
                f"_{feature_type}_" in file
                and f"predictions_{montage}_" in file
                and f"_{length}s_" in file
                and f"_{combiner}_" in file
            ):
                pattern_files.append(file)

        roc_data = []
        for file in pattern_files:
            df_file = pd.read_csv(os.path.join(datafolder, file))
            last_df_file = df_file
            fpr, tpr, thresholds = roc_curve(df_file["y_tests"], df_file["y_scores"])
            roc_data.append((fpr, tpr, thresholds))

        if not roc_data:
            continue

        fpr_avg, tpr_avg = vertical_avg_roc(roc_data)
        if type_to_save is not None and save_dir and feature_type == type_to_save:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, f"roc_{feature_type}_{montage}_{length}s_{combiner}.npz"), fpr=fpr_avg, tpr=tpr_avg)

        label_name = feature_display_names.get(feature_type, feature_type)
        plt.step(
            fpr_avg,
            tpr_avg,
            label=f"{label_name} - AUC={row['auc']:.2f}",
            color=colors[plot_index % len(colors)],
            alpha=0.7,
        )
        plot_index += 1

    if last_df_file is not None:
        n_epileptic = int(np.sum(last_df_file["y_tests"]))
        n_healthy = int(len(last_df_file["y_tests"]) - n_epileptic)
        slope = compute_threshold_slope(n_epileptic, n_healthy, 0.6)
        plt.plot([0, 1], [0, slope], "--", color="green", alpha=0.5)
        plt.fill_between([0, 1], [0, slope], [1, slope], color="green", alpha=0.1, label="P(Epilepsy|y=1)$\\geq$0.6")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc="lower right", fontsize=8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves of Best Feature Types")
    plt.grid(True)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{plot_prefix}_best_feature_roc_curves.pdf", dpi=300)
    plt.show()

    return df_summary, df_best, std_df


def compute_shap_values_loso(
    data: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    unique_subjects: np.ndarray,
    xgb_params: Optional[dict] = None,
):
    import shap

    if xgb_params is None:
        xgb_params = {
            "n_jobs": 4,
            "device": "cpu",
            "n_estimators": 100,
            "max_depth": 6,
            "subsample": 0.9,
            "gamma": 0.1,
            "learning_rate": 0.01,
        }

    shap_values_folds = []
    for subj in tqdm.tqdm(unique_subjects):
        test_indices = np.where(subjects == subj)[0]
        train_indices = np.where(subjects != subj)[0]

        X_train, X_test = data[train_indices], data[test_indices]
        y_train = labels[train_indices]

        ratio = (len(y_train) - np.sum(y_train)) / max(np.sum(y_train), 1)
        model = XGBClassifier(scale_pos_weight=ratio, **xgb_params)
        model.fit(X_train, y_train)

        explainer = shap.Explainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
        for shap_value in shap_values:
            shap_values_folds.append(shap_value)

    return np.array(shap_values_folds)


def create_shap_summary_plot(
    shap_values: np.ndarray,
    data: np.ndarray,
    feature_name: str,
    montage: str,
    segment_length: int,
    combiner: str,
    max_display: int = 5,
    custom_feature_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
):
    import shap

    fig = plt.figure(figsize=(8, 5))
    plt.suptitle(
        f"SHAP Summary Plot for {feature_name} - montage: {montage}, "
        f"segment_length: {segment_length}, combiner: {combiner}"
    )
    _ = fig.add_subplot(111)
    shap.summary_plot(
        shap_values,
        data,
        max_display=max_display,
        show=False,
        feature_names=custom_feature_names,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plv_index_to_name(idx: int, channels: Optional[Iterable[str]] = None) -> str:
    bands = ["raw", "delta", "theta", "alpha", "beta", "gamma"]
    if channels is None:
        channels = [
            "FP1_Lap",
            "F3_Lap",
            "C3_Lap",
            "P3_Lap",
            "F7_Lap",
            "T3_Lap",
            "T5_Lap",
            "O1_Lap",
            "FZ_Lap",
            "CZ_Lap",
            "PZ_Lap",
            "FP2_Lap",
            "F4_Lap",
            "C4_Lap",
            "P4_Lap",
            "F8_Lap",
            "T4_Lap",
            "T6_Lap",
            "O2_Lap",
        ]

    ch = list(channels)
    p = len(ch) * (len(ch) - 1) // 2
    band_id, pair_id = divmod(idx, p)
    j = next(jj for jj in range(1, len(ch)) if jj * (jj + 1) // 2 > pair_id)
    k = pair_id - j * (j - 1) // 2
    return f"PLV_{bands[band_id]}_{ch[j]}-{ch[k]}"


UTM_METRICS = [
    "Mean",
    "Median",
    "SD",
    "Skew",
    "Kurtosis",
    "ZC",
    "NLEO_Env_Diff",
    "NLEO_Teager",
    "Energy_T",
    "Energy_F",
    "Entropy",
    "Vpp",
    "NPks",
]


def utm_index_to_name(idx: int, channels: Iterable[str]) -> str:
    channels = list(channels)
    metric_count = len(UTM_METRICS)
    ch_id, met_id = divmod(idx, metric_count)
    ch_label = channels[ch_id]
    metric = UTM_METRICS[met_id]
    return "_".join(["UTM", metric, ch_label])


def spectral_index_to_name(idx: int, channels: Iterable[str]) -> str:
    channels = list(channels)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    channel_count = len(channels)
    band_id, ch_id = divmod(idx, channel_count)
    return "_".join(["S", bands[band_id], channels[ch_id]])


def compute_ale_1d(model, X_train: np.ndarray, feature_idx: int, n_bins: int = 20):
    feature_values = X_train[:, feature_idx]

    valid_mask = ~np.isnan(feature_values)
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])

    feature_values = feature_values[valid_mask]
    X_valid = X_train[valid_mask]

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(feature_values, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 3:
        return np.array([]), np.array([]), np.array([])

    ale_values = []
    bin_centers = []
    bin_counts = []

    for i in range(len(bin_edges) - 1):
        in_bin = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i + 1])
        if i == len(bin_edges) - 2:
            in_bin = (feature_values >= bin_edges[i]) & (feature_values <= bin_edges[i + 1])

        if np.sum(in_bin) == 0:
            continue

        X_bin = X_valid[in_bin]
        X_lower = X_bin.copy()
        X_upper = X_bin.copy()
        X_lower[:, feature_idx] = bin_edges[i]
        X_upper[:, feature_idx] = bin_edges[i + 1]

        pred_lower = model.predict_proba(X_lower)[:, 1]
        pred_upper = model.predict_proba(X_upper)[:, 1]
        local_effect = np.mean(pred_upper - pred_lower)

        ale_values.append(local_effect)
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        bin_counts.append(np.sum(in_bin))

    if not ale_values:
        return np.array([]), np.array([]), np.array([])

    ale_values = np.array(ale_values)
    bin_centers = np.array(bin_centers)
    bin_counts = np.array(bin_counts)

    ale_cumsum = np.cumsum(ale_values)
    weighted_mean = np.average(ale_cumsum, weights=bin_counts)
    ale_cumsum -= weighted_mean

    return ale_cumsum, bin_centers, bin_counts


def interpolate_ale_to_common_grid(
    ale_values: np.ndarray,
    bin_centers: np.ndarray,
    common_grid: np.ndarray,
) -> np.ndarray:
    if len(ale_values) == 0 or len(bin_centers) == 0:
        return np.full(len(common_grid), np.nan)

    min_val, max_val = bin_centers.min(), bin_centers.max()
    valid_mask = (common_grid >= min_val) & (common_grid <= max_val)
    interpolated_ale = np.full(len(common_grid), np.nan)

    if np.any(valid_mask):
        interp_fn = scipy_interpolate.interp1d(
            bin_centers,
            ale_values,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        interpolated_ale[valid_mask] = interp_fn(common_grid[valid_mask])

    return interpolated_ale


def aggregate_ale_across_folds(fold_results: list[tuple], n_grid_points: int = 100):
    valid_results = [(ale, centers, counts, n_train) for ale, centers, counts, n_train in fold_results if len(ale) > 0 and len(centers) > 0]

    if not valid_results:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    all_centers = np.concatenate([centers for _, centers, _, _ in valid_results])
    min_val, max_val = np.percentile(all_centers, [5, 95])
    common_grid = np.linspace(min_val, max_val, n_grid_points)

    interpolated_ales = []
    fold_weights = []

    for ale_values, bin_centers, bin_counts, n_train in valid_results:
        ale_centered = ale_values - np.average(ale_values, weights=bin_counts)
        interpolated = interpolate_ale_to_common_grid(ale_centered, bin_centers, common_grid)
        interpolated_ales.append(interpolated)
        fold_weights.append(n_train)

    interpolated_ales = np.array(interpolated_ales)
    fold_weights = np.array(fold_weights)
    valid_mask = ~np.isnan(interpolated_ales)

    mean_ale = np.full(n_grid_points, np.nan)
    std_ale = np.full(n_grid_points, np.nan)
    ci_lower = np.full(n_grid_points, np.nan)
    ci_upper = np.full(n_grid_points, np.nan)

    for i in range(n_grid_points):
        valid_i = valid_mask[:, i]
        if np.sum(valid_i) <= 1:
            continue
        values = interpolated_ales[valid_i, i]
        weights = fold_weights[valid_i]

        mean_ale[i] = np.average(values, weights=weights)
        std_ale[i] = np.std(values, ddof=1)

        n_folds = len(values)
        t_val = t.ppf(0.975, n_folds - 1)
        margin = t_val * std_ale[i] / np.sqrt(n_folds)
        ci_lower[i] = mean_ale[i] - margin
        ci_upper[i] = mean_ale[i] + margin

    return common_grid, mean_ale, std_ale, ci_lower, ci_upper


def prepare_ale_plot_data(
    common_grid: np.ndarray,
    mean_ale: np.ndarray,
    std_ale: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    feature_name: str,
    feature_idx: int,
    montage: str,
    segment_length: int,
    combiner: str,
    individual_folds: Optional[list[np.ndarray]] = None,
    p_value: Optional[float] = None,
    feature_values: Optional[np.ndarray] = None,
):
    valid_mask = ~np.isnan(mean_ale)
    return {
        "common_grid": common_grid,
        "mean_ale": mean_ale,
        "std_ale": std_ale,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "valid_mask": valid_mask,
        "feature_name": feature_name,
        "feature_idx": feature_idx,
        "montage": montage,
        "segment_length": segment_length,
        "combiner": combiner,
        "individual_folds": individual_folds,
        "p_value": p_value,
        "feature_values": feature_values,
    }


def plot_ale_from_data(
    plot_data: dict,
    feature_names: list[str],
    figsize: tuple[int, int] = (10, 6),
    show_rugplot: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize)

    if plot_data["individual_folds"] is not None:
        for fold_ale in plot_data["individual_folds"]:
            ax.plot(plot_data["common_grid"], fold_ale, "lightgray", alpha=0.4, linewidth=0.5)

    valid_mask = plot_data["valid_mask"]
    if np.any(valid_mask):
        ax.fill_between(
            plot_data["common_grid"][valid_mask],
            plot_data["ci_lower"][valid_mask],
            plot_data["ci_upper"][valid_mask],
            alpha=0.3,
            color="blue",
            label="95% CI",
        )
        ax.plot(
            plot_data["common_grid"][valid_mask],
            plot_data["mean_ale"][valid_mask],
            "b-",
            linewidth=2,
            label="Mean ALE",
        )
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    if show_rugplot and plot_data["feature_values"] is not None:
        valid_feature_vals = plot_data["feature_values"][~np.isnan(plot_data["feature_values"])]
        if len(valid_feature_vals) > 0:
            y_min, y_max = ax.get_ylim()
            rug_height = (y_max - y_min) * 0.02
            rug_y = y_min + rug_height
            if len(valid_feature_vals) > 1000:
                sample_idx = np.random.choice(len(valid_feature_vals), 1000, replace=False)
                rug_values = valid_feature_vals[sample_idx]
            else:
                rug_values = valid_feature_vals
            ax.plot(
                rug_values,
                np.full(len(rug_values), rug_y),
                "|",
                color="darkgray",
                alpha=0.6,
                markersize=20,
                markeredgewidth=0.5,
                label="Data points",
            )

    title = f"ALE Plot for {feature_names[plot_data['feature_idx']]}\n"
    title += f"Montage: {plot_data['montage']}, Length: {plot_data['segment_length']}s, Combiner: {plot_data['combiner']}"
    if plot_data["p_value"] is not None:
        title += f"\nMann-Whitney U p-value: {plot_data['p_value']:.4f}"

    ax.set_xlabel(f"Feature {plot_data['feature_idx']} Value")
    ax.set_ylabel("ALE (Change in Prediction)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def compute_mann_whitney_features(
    data: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    top_n: int = 5,
):
    n_features = data.shape[1]
    p_values = np.full(n_features, 1.0)
    statistics = np.full(n_features, 0.0)

    for i in tqdm.tqdm(range(n_features), desc="Computing Mann-Whitney U tests"):
        feature_data = data[:, i]
        valid_mask = ~np.isnan(feature_data)
        if np.sum(valid_mask) < 10:
            continue

        feature_valid = feature_data[valid_mask]
        labels_valid = labels[valid_mask]

        group_0 = feature_valid[labels_valid == 0]
        group_1 = feature_valid[labels_valid == 1]
        if len(group_0) < 3 or len(group_1) < 3:
            continue

        try:
            statistic, p_value = mannwhitneyu(group_0, group_1, alternative="two-sided")
            p_values[i] = p_value
            statistics[i] = statistic
        except Exception:
            continue

    valid_tests = p_values < 1.0
    corrected_p_values = np.full(n_features, 1.0)

    if np.any(valid_tests):
        valid_p_values = p_values[valid_tests]
        _, corrected_valid, _, _ = multipletests(valid_p_values, alpha=alpha, method="bonferroni")
        corrected_p_values[valid_tests] = corrected_valid

    valid_features = np.where(valid_tests)[0]
    if len(valid_features) == 0:
        return np.array([]), p_values, corrected_p_values

    sorted_indices = np.argsort(corrected_p_values[valid_features])
    top_n_valid = valid_features[sorted_indices[: min(top_n, len(valid_features))]]

    return top_n_valid, p_values, corrected_p_values


def save_ale_results(
    ale_results: dict,
    plot_data_collection: dict,
    feature_name: str,
    montage: str,
    segment_length: int,
    combiner: str,
    top_features: np.ndarray,
    mw_p_values: np.ndarray,
    mw_corrected_p_values: np.ndarray,
    save_dir: str = "ale_results",
):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ale_results_{feature_name}_{montage}_{segment_length}s_{combiner}_{timestamp}.pkl"
    save_path = os.path.join(save_dir, filename)

    save_data = {
        "ale_results": ale_results,
        "plot_data_collection": plot_data_collection,
        "metadata": {
            "feature_name": feature_name,
            "montage": montage,
            "segment_length": segment_length,
            "combiner": combiner,
            "timestamp": timestamp,
            "n_features_analyzed": len(ale_results),
            "n_plot_data_prepared": len(plot_data_collection),
        },
        "statistical_tests": {
            "top_features": top_features,
            "mw_p_values": mw_p_values,
            "mw_corrected_p_values": mw_corrected_p_values,
        },
    }

    with open(save_path, "wb") as file_obj:
        pickle.dump(save_data, file_obj)

    return save_path


def load_ale_results(load_path: str):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Results file not found: {load_path}")

    with open(load_path, "rb") as file_obj:
        save_data = pickle.load(file_obj)

    return (
        save_data["ale_results"],
        save_data["plot_data_collection"],
        save_data["metadata"],
        save_data["statistical_tests"],
    )


def list_saved_results(save_dir: str = "ale_results"):
    if not os.path.exists(save_dir):
        return []

    results_files = []
    pkl_files = [file for file in os.listdir(save_dir) if file.endswith(".pkl")]

    for filename in sorted(pkl_files):
        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, "rb") as file_obj:
                save_data = pickle.load(file_obj)
            metadata = save_data["metadata"]
            results_files.append((filename, metadata))
        except Exception:
            continue

    return results_files
