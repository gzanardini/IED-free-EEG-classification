from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.special import logit


def train_feature_model_parallel(
    args,
    *,
    cache: dict,
    n_jobs_xgb: int,
    device,
):
    """Train an XGBoost model for a single feature, optionally retraining on train+val."""
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

    # Set random seeds for this process
    np.random.seed(seed)

    data = cache[feature_name]

    if retrain_on_trainval:
        train_val_idxs = np.concatenate([train_idxs, val_idxs])
        y_train_val = np.concatenate([y_train, y_val])
        ratio = (len(y_train_val) - sum(y_train_val)) / sum(y_train_val)

        from xgboost import XGBClassifier

        model = XGBClassifier(
            scale_pos_weight=ratio,
            n_jobs=n_jobs_xgb,
            device=device,
            n_estimators=100,
            seed=seed,
            max_depth=6,
            subsample=0.9,
            gamma=0.1,
            learning_rate=0.01,
        )
        model.fit(data[train_val_idxs], y_train_val)

        test_probs = model.predict_proba(data[test_idxs])[:, 1]
        return {
            "feature_name": feature_name,
            "test_probs": test_probs,
            "model": model,
        }

    ratio = (len(y_train) - sum(y_train)) / sum(y_train)

    from xgboost import XGBClassifier

    model = XGBClassifier(
        scale_pos_weight=ratio,
        n_jobs=n_jobs_xgb,
        device=device,
        n_estimators=100,
        seed=seed,
        max_depth=6,
        subsample=0.9,
        gamma=0.1,
        learning_rate=0.01,
    )
    model.fit(data[train_idxs], y_train)

    train_probs = model.predict_proba(data[train_idxs])[:, 1]
    val_probs = model.predict_proba(data[val_idxs])[:, 1]
    test_probs = model.predict_proba(data[test_idxs])[:, 1]

    auc = roc_auc_score(y_val, val_probs)
    bac = balanced_accuracy_score(y_val, val_probs >= 0.5)
    bac80, _, _, _ = calculate_bac(y_val, val_probs, 0.8)
    score = auc + bac

    return {
        "feature_name": feature_name,
        "train_probs": train_probs,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "auc": auc,
        "bac": bac,
        "bac80": bac80,
        "score": score,
    }


def train_ensemble_models(
    feature_combination,
    train_idxs,
    val_idxs,
    test_idxs,
    y_train,
    y_val,
    seed,
    *,
    cache: dict,
    n_jobs_xgb: int,
    device,
    n_parallel_features: int,
    simplex_alpha: float,
    calculate_bac: Callable,
    find_optimal_threshold: Callable,
    train_simplex_logistic: Callable,
    SimplexLogistic,
    log_stage1: Optional[Callable[[dict], None]] = None,
    log_stage2: Optional[Callable[[dict], None]] = None,
):
    """Train a two-stage ensemble and return predictions/weights."""
    print("Stage 1: Training individual models and learning meta-learner weights...")

    args_list_stage1 = [
        (feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed + i, False)
        for i, feature_name in enumerate(feature_combination)
    ]

    feature_results_stage1 = Parallel(
        n_jobs=min(n_parallel_features, len(feature_combination)),
        backend="threading",
    )(
        delayed(train_feature_model_parallel)(
            args,
            cache=cache,
            n_jobs_xgb=n_jobs_xgb,
            device=device,
        )
        for args in args_list_stage1
    )

    feature_models_stage1 = sorted(
        feature_results_stage1,
        key=lambda x: feature_combination.index(x["feature_name"]),
    )

    val_probs_list = [model["val_probs"] for model in feature_models_stage1]

    calibrated_probs = []
    calibrators = []
    for probs in val_probs_list:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        cal_probs = calibrator.fit_transform(probs, y_val)
        calibrated_probs.append(cal_probs)
        calibrators.append(calibrator)

    X_meta_val = np.column_stack(
        [logit(np.clip(probs, 0.001, 0.999)) for probs in calibrated_probs]
    )

    w_simplex = train_simplex_logistic(X_meta_val, y_val, alpha=simplex_alpha)
    meta_model = SimplexLogistic(w_simplex)

    meta_val_probs = meta_model.predict_proba(X_meta_val)[:, 1]
    stage1_auc = roc_auc_score(y_val, meta_val_probs)
    stage1_bac = balanced_accuracy_score(y_val, meta_val_probs >= 0.5)
    stage1_bac80, _, _, _ = calculate_bac(y_val, meta_val_probs, 0.8)

    print(
        "Stage 1 - Meta-learner weights: {}".format(w_simplex)
    )
    print(
        "Stage 1 - Validation AUC: {:.4f}, BAC: {:.4f}, BAC80: {:.4f}".format(
            stage1_auc, stage1_bac, stage1_bac80
        )
    )

    if log_stage1:
        log_stage1(
            {
                "auc": stage1_auc,
                "bac": stage1_bac,
                "bac80": stage1_bac80,
                "weights": w_simplex,
                "meta_probs": meta_val_probs,
                "raw_probs": np.column_stack(val_probs_list),
                "calibrated_probs": np.column_stack(calibrated_probs),
                "logits": X_meta_val,
            }
        )

    print("Stage 2: Retraining models on train+val data for final predictions...")

    args_list_stage2 = [
        (feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed + i, True)
        for i, feature_name in enumerate(feature_combination)
    ]

    feature_results_stage2 = Parallel(
        n_jobs=min(n_parallel_features, len(feature_combination)),
        backend="threading",
    )(
        delayed(train_feature_model_parallel)(
            args,
            cache=cache,
            n_jobs_xgb=n_jobs_xgb,
            device=device,
        )
        for args in args_list_stage2
    )

    feature_models_stage2 = sorted(
        feature_results_stage2,
        key=lambda x: feature_combination.index(x["feature_name"]),
    )

    for model in feature_models_stage2:
        del model["feature_name"]

    test_probs_list = [model["test_probs"] for model in feature_models_stage2]

    calibrated_test_probs = []
    for i, probs in enumerate(test_probs_list):
        cal_test_probs = calibrators[i].transform(probs)
        calibrated_test_probs.append(cal_test_probs)

    X_meta_test = np.column_stack(
        [logit(np.clip(probs, 0.001, 0.999)) for probs in calibrated_test_probs]
    )

    meta_test_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    opt_threshold = find_optimal_threshold(y_val, meta_val_probs)
    meta_test_preds = (meta_test_probs >= opt_threshold).astype(int)

    print("Stage 2 - Final predictions generated using learned weights")

    if log_stage2:
        log_stage2(
            {
                "meta_probs": meta_test_probs,
                "meta_preds": meta_test_preds,
                "opt_threshold": opt_threshold,
                "raw_probs": np.column_stack(test_probs_list),
                "calibrated_probs": np.column_stack(calibrated_test_probs),
                "logits": X_meta_test,
            }
        )

    return {
        "feature_models": feature_models_stage2,
        "meta_model": meta_model,
        "calibrators": calibrators,
        "val_probs": meta_val_probs,
        "test_probs": meta_test_probs,
        "test_preds": meta_test_preds,
        "opt_threshold": opt_threshold,
        "auc": stage1_auc,
        "bac": stage1_bac,
        "bac80": stage1_bac80,
        "lr_weights": meta_model.coef_[0],
    }


def calculate_bac(labels: np.ndarray, scores: np.ndarray, sens_thresh: float):
    """Local fallback for BAC when a callable is not provided."""
    from sklearn.metrics import confusion_matrix, roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold_sensitivity = thresholds[np.where(tpr >= sens_thresh)[0][0]]
    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bac = (sensitivity + specificity) / 2
    return bac, fpr, tpr, thresholds


__all__ = [
    "train_feature_model_parallel",
    "train_ensemble_models",
]
