import itertools
from typing import Callable, Iterable, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from utils.ensemble_common import (
    load_data_both_aligned,
    load_feature_from_best_params,
    load_subject_data,
    preload_combined_feature_cache,
    preload_feature_cache,
)


def build_train_val_test_indices(
    description,
    labels: np.ndarray,
    subject,
    split_ratio: float,
    seed: int,
    unique_subjects: Optional[Iterable] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train/val/test indices for LOSO CV with stratified train/val split."""
    test_idxs = np.where(description["subject"] == subject)[0]
    subjects = description["subject"]

    if unique_subjects is None:
        unique_subjects = np.unique(subjects)

    other_subjects = [subj for subj in unique_subjects if subj != subject]
    other_subjects_labels = np.array([[subj, labels[subjects == subj][0]] for subj in other_subjects])

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
    feature_names: list[str],
    min_len: int = 2,
    max_len: int = 10,
) -> list[list[str]]:
    """Generate combinations for the given feature list."""
    combinations: list[list[str]] = []
    for length in range(min_len, max_len):
        for combo in itertools.combinations(feature_names, length):
            combinations.append(list(combo))
    return combinations


def get_cached_feature_data(cache: dict, feature_name: str):
    """Fetch a cached feature array by name."""
    return cache[feature_name]


def load_feature_data(
    data_folder: str,
    best_parameters: dict,
    feature_name: str,
    array_converter,
):
    """Load one feature matrix based on the best parameters table."""
    return load_feature_from_best_params(data_folder, best_parameters, feature_name, array_converter)


def preload_feature_data_cache(
    feature_names: list[str],
    best_parameters: dict,
    data_folder: str,
    array_converter,
) -> dict:
    """Preload all feature matrices into a cache."""
    return preload_feature_cache(feature_names, best_parameters, data_folder, array_converter)


def preload_combined_feature_data_cache(
    feature_names: list[str],
    best_parameters_background: dict,
    best_parameters_ips: dict,
    data_folder_bg: str,
    data_folder_ips: str,
    ips_reorder_indices,
    array_converter,
) -> dict:
    """Preload and concatenate IPS + background features into a cache."""
    return preload_combined_feature_cache(
        feature_names,
        best_parameters_background,
        best_parameters_ips,
        data_folder_bg,
        data_folder_ips,
        ips_reorder_indices,
        array_converter,
    )


__all__ = [
    "build_train_val_test_indices",
    "generate_feature_combinations",
    "get_cached_feature_data",
    "load_data_both_aligned",
    "load_feature_data",
    "load_subject_data",
    "preload_combined_feature_data_cache",
    "preload_feature_data_cache",
]
