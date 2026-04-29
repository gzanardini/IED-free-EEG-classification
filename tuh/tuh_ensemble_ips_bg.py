import os
import wandb.plot
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score, average_precision_score
from sklearn.metrics import auc as auc_sklearn
import itertools 
import secrets
from sklearn.model_selection import train_test_split
import warnings
import cupy as cp
from cupy.cuda import Device
import random
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit, logit            # σ(x) = 1 / (1+e^{-x})
from sklearn.isotonic import IsotonicRegression

np.set_printoptions(linewidth=200, precision=4)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
N_RUNS = 5
N_CUDA = 0
DEVICE = f'cuda:{N_CUDA}'
SPLIT_RATIO = 0.3
PROJECT_NAME = 'tuh_ips+bg'
WANDB_KEY = '96e9a92e52e807ed253b3872afd1de1bafc3640a'
DATA_FOLDER_IPS = '/space/gzanardini/tuh_whole/split'
DATA_FOLDER_BG = '/space/gzanardini/tuh_background/split'
LOG_FOLDER = '/space/gzanardini/tuh/'
N_JOBS_XGB = 1  # Set to 1 for compatibility with CUDA
NUM_WORKERS = 10  # Number of max parallel workers for training
N_PARALLEL_FEATURES = NUM_WORKERS  # Parallel feature training within combination
SCIPY_ARRAY_API=1  # Enable SciPy array API for compatibility with cupy

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10, 20, 60]
feature_names = ['cc', 'cwt', 'dwt', 'plv', 'mst', 'sst', 'spectral', 'utm', 'gcc', 'gplv']
combiners = ['mean', 'median', 'std', 'skew', 'kurt']

best_parameters_ips = {
    'spectral': ('CAR', 1 ,'skew'),
    'cwt': ('BipolarDB', 60, 'std'),
    'dwt': ('Cz', 10, 'skew'),
    'mst': ('BipolarDB', 10, 'skew'),
    'sst': ('Laplacian', 20, 'skew'),
    'cc': ('Cz', 10, 'skew'),
    'plv': ('Laplacian', 2, 'std'),
    'gcc': ('CAR', 1, 'std'),
    'gplv': ('BipolarDB', 1, 'mean'),
    'utm': ('Laplacian', 60, 'mean')
}

'''     utm Laplacian     60   median 
        spectral BipolarDB      2     kurt 
        plv        Cz     60      std 
        cc       CAR    120     mean 
        cwt        Cz      1     skew 
        sst Laplacian     20      std 0
        gplv Laplacian     10     mean 
        dwt        Cz     10     skew 
        gcc        Cz     20     kurt 
        mst BipolarDB      1     median'''

best_parameters_background= {
    'spectral': ('BipolarDB', 2, 'kurt'),
    'cwt':      ('Cz',      1,      'skew'),
    'dwt':      ('Cz',     10,     'skew'),
    'mst':      ('BipolarDB', 1, 'median'),
    'sst':      ('Laplacian', 20, 'std'),
    'cc':       ('CAR', 120, 'mean'),
    'plv':      ('Cz', 60, 'std'),
    'gcc':      ('Cz', 20, 'kurt'),
    'gplv':     ('Laplacian', 10, 'mean'),
    'utm':      ('Laplacian', 60, 'median')
}

def train_simplex_logistic(X, y, max_iter=2500):

    d = X.shape[1]
    init_w = np.full(d, 1.0/d)                # uniform start

    # negative log-likelihood  (bias = 0)
    def nll(w):
        logits = X @ w
        ce = -np.sum(y * np.log(expit(logits)) +
                     (1 - y) * np.log(1 - expit(logits)))
        alpha = 1.05      # α = 1 is uniform prior; α > 1 discourages zeros
        dirichlet_pen = (alpha - 1) * -np.sum(np.log(w + 1e-12))
        return ce + dirichlet_pen

    bounds      = [(0.00, None)] * d             # w_i ≥ 0
    constraints = {'type': 'eq',
                   'fun': lambda w: np.sum(w) - 1}

    res = minimize(nll, init_w, method='SLSQP',
                   bounds=bounds,
                   constraints=constraints,
                   options={'maxiter': max_iter})

    if not res.success:
        raise RuntimeError("Simplex LR did not converge: " + res.message)

    return res.x

class SimplexLogistic:
    """Tiny wrapper so the rest of the pipeline keeps working."""
    def __init__(self, w):
        self.coef_  = w[None, :]              # scikit style (1, d)
    def predict_proba(self, X):
        p = expit(X @ self.coef_.ravel())
        return np.column_stack([1-p, p])

def setup_environment():
    """Initialize CUDA and wandb."""
    if DEVICE != 'cpu':
        Device(N_CUDA).use()
    wandb.login(key=WANDB_KEY)
    
def find_optimal_threshold(y_true, y_prob):
    """Find the optimal decision threshold based on the maximum geometric mean score."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmeans = np.sqrt(tpr * (1 - fpr))
    opt_index = np.argmax(gmeans)
    return thresholds[opt_index]

def calculate_bac(labels, scores, sens_thresh):
    """Calculate balanced accuracy with sensitivity threshold."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold_sensitivity = thresholds[np.where(tpr >= sens_thresh)[0][0]]
    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bac = ((sensitivity + specificity) / 2)
    return bac, fpr, tpr, thresholds

def handle_complex_numbers(features):
    """Handle complex numbers and infinite values in features."""
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

def load_data(path):
    """Load and prepare the dataset."""
    description = pd.read_csv(f'{path}/description.csv')
    labels = description['epilepsy'].to_numpy()
    subjects = description['subject'].to_numpy()
    unique_subjects = np.unique(description['subject'])
    
    subject_labels = []
    for subj in unique_subjects:
        lbl = labels[subjects == subj][0]
        subject_labels.append([subj, lbl])
    subject_labels = np.array(subject_labels)
    
    return description, labels, subjects, unique_subjects, subject_labels

def load_data_both():
    """Load both background and IPS data, filter IPS to match background subjects."""
    # Load background data (this determines the subject set)
    bg_description = pd.read_csv(f'{DATA_FOLDER_BG}/description.csv')
    bg_labels = bg_description['epilepsy'].to_numpy()
    bg_subjects = bg_description['subject'].to_numpy()
    bg_unique_subjects = np.unique(bg_description['subject'])
    
    # Load IPS data
    ips_description = pd.read_csv(f'{DATA_FOLDER_IPS}/description.csv')
    ips_labels = ips_description['epilepsy'].to_numpy()
    ips_subjects = ips_description['subject'].to_numpy()
    
    # Find common subjects
    common_subjects = np.intersect1d(bg_unique_subjects, np.unique(ips_subjects))
    print(f"Found {len(common_subjects)} common subjects between background and IPS data")
    print(f"Background subjects: {len(bg_unique_subjects)}, IPS subjects: {len(np.unique(ips_subjects))}")
    
    # Filter background data to only include common subjects
    bg_mask = np.isin(bg_subjects, common_subjects)
    filtered_bg_description = bg_description[bg_mask].reset_index(drop=True)
    filtered_bg_labels = bg_labels[bg_mask]
    filtered_bg_subjects = bg_subjects[bg_mask]
    
    # Filter IPS data to only include common subjects
    ips_mask = np.isin(ips_subjects, common_subjects)
    filtered_ips_description = ips_description[ips_mask].reset_index(drop=True)
    filtered_ips_labels = ips_labels[ips_mask]
    filtered_ips_subjects = ips_subjects[ips_mask]
    
    # Verify that subjects and labels match between filtered datasets
    assert np.array_equal(np.sort(filtered_bg_subjects), np.sort(filtered_ips_subjects)), "Subject mismatch between datasets"
    assert np.array_equal(filtered_bg_labels[np.argsort(filtered_bg_subjects)], 
                         filtered_ips_labels[np.argsort(filtered_ips_subjects)]), "Label mismatch between datasets"
    
    # Additional sanity checks
    print(f"Sanity check - Filtered background data: {len(filtered_bg_subjects)} samples")
    print(f"Sanity check - Filtered IPS data: {len(filtered_ips_subjects)} samples")
    print(f"Sanity check - Background epilepsy ratio: {np.mean(filtered_bg_labels):.3f}")
    print(f"Sanity check - IPS epilepsy ratio: {np.mean(filtered_ips_labels):.3f}")
    
    # Create mapping for reordering IPS data to match background order
    bg_subject_order = {subj: idx for idx, subj in enumerate(filtered_bg_subjects)}
    ips_reorder_indices = [np.where(filtered_ips_subjects == subj)[0][0] for subj in filtered_bg_subjects]
    
    subject_labels = []
    for subj in common_subjects:
        lbl = filtered_bg_labels[filtered_bg_subjects == subj][0]
        subject_labels.append([subj, lbl])
    subject_labels = np.array(subject_labels)
    
    return (filtered_bg_description, filtered_bg_labels, filtered_bg_subjects, common_subjects, subject_labels, 
            ips_reorder_indices)

def preload_all_feature_data_combined():
    """Preload and concatenate both IPS and background feature data."""
    print("Preloading and concatenating IPS and background feature data...")
    cache = {}
    
    # Load subject mapping information
    _, _, _, _, _, ips_reorder_indices = load_data_both()
    
    for feature_name in feature_names:
        # Load background data
        bg_montage, bg_segment_length, bg_combiner = best_parameters_background[feature_name]
        bg_data_path = f'{DATA_FOLDER_BG}/{feature_name}_{bg_montage}_{bg_segment_length}s_{bg_combiner}.npy'
        bg_data = np.load(bg_data_path)
        bg_data = handle_complex_numbers(bg_data)
        
        if len(bg_data.shape) > 2:
            bg_data = bg_data.reshape(bg_data.shape[0], -1)
        
        # Load IPS data
        ips_montage, ips_segment_length, ips_combiner = best_parameters_ips[feature_name]
        ips_data_path = f'{DATA_FOLDER_IPS}/{feature_name}_{ips_montage}_{ips_segment_length}s_{ips_combiner}.npy'
        ips_data = np.load(ips_data_path)
        ips_data = handle_complex_numbers(ips_data)
        
        if len(ips_data.shape) > 2:
            ips_data = ips_data.reshape(ips_data.shape[0], -1)
        
        # Sanity checks before reordering and concatenation
        print(f"Sanity check - {feature_name}:")
        print(f"  Background original shape: {bg_data.shape}")
        print(f"  IPS original shape: {ips_data.shape}")
        
        # Verify we have enough samples after filtering
        expected_samples = len(ips_reorder_indices)
        assert bg_data.shape[0] == expected_samples, f"Background {feature_name}: expected {expected_samples} samples, got {bg_data.shape[0]}"
        assert ips_data.shape[0] >= expected_samples, f"IPS {feature_name}: expected at least {expected_samples} samples, got {ips_data.shape[0]}"
        
        # Reorder IPS data to match background subject order
        ips_data_reordered = ips_data[ips_reorder_indices]
        
        # Final sanity check after reordering
        assert ips_data_reordered.shape[0] == bg_data.shape[0], f"Shape mismatch after reordering for {feature_name}: BG={bg_data.shape[0]}, IPS={ips_data_reordered.shape[0]}"
        
        # Concatenate along feature axis (axis=1)
        combined_data = np.concatenate([bg_data, ips_data_reordered], axis=1)
        
        # Convert to cupy array for GPU processing
        cache[feature_name] = cp.array(combined_data)
        print(f"  Final combined shape: {combined_data.shape} (BG: {bg_data.shape[1]} + IPS: {ips_data_reordered.shape[1]} features)")
    
    print("All feature data loaded and concatenated successfully!")
    return cache

def generate_feature_combinations(start=2, end=None):
    """Generate all possible combinations of features."""
    all_combinations = []
    
    if end is None:
        end = len(feature_names)
    
    # Generate combinations of different lengths (start to end features)
    for length in range(start, end + 1):
        combos = itertools.combinations(feature_names, length)
        all_combinations.extend(list(combos))
    
    return all_combinations

def get_cached_feature_data(feature_name):
    """Get preloaded feature data."""
    return _feature_data_cache[feature_name]

def train_feature_model_parallel(args):
    """Wrapper function for parallel feature model training."""
    feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed, retrain_on_trainval = args
    
    # Set random seeds for this process
    random.seed(seed)
    np.random.seed(seed)
    
    # Get cached data instead of loading
    data = get_cached_feature_data(feature_name)
    
    if retrain_on_trainval:
        # Second stage: Train on train+val data for final predictions
        train_val_idxs = np.concatenate([train_idxs, val_idxs])
        y_train_val = np.concatenate([y_train, y_val])
        ratio = (len(y_train_val) - sum(y_train_val)) / sum(y_train_val)
        
        model = XGBClassifier(
            scale_pos_weight=ratio,
            n_jobs=N_JOBS_XGB,
            device=DEVICE,
            n_estimators=100,
            seed=seed,
            max_depth=6,
            subsample=0.9,
            gamma=0.1,
            learning_rate=0.01
        )
        model.fit(data[train_val_idxs], y_train_val)
        
        # Only generate test predictions for final stage
        test_probs = model.predict_proba(data[test_idxs])[:, 1]
        
        return {
            'feature_name': feature_name,
            'test_probs': test_probs,
            'model': model
        }
    else:
        # First stage: Train on train data, validate on val data
        ratio = (len(y_train) - sum(y_train)) / sum(y_train)
        
        model = XGBClassifier(
            scale_pos_weight=ratio,
            n_jobs=N_JOBS_XGB,
            device=DEVICE,
            n_estimators=100,
            seed=seed,
            max_depth=6,
            subsample=0.9,
            gamma=0.1,
            learning_rate=0.01
        )
        model.fit(data[train_idxs], y_train)
        
        # Generate predictions
        train_probs = model.predict_proba(data[train_idxs])[:, 1]
        val_probs = model.predict_proba(data[val_idxs])[:, 1]
        test_probs = model.predict_proba(data[test_idxs])[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_val, val_probs)
        bac = balanced_accuracy_score(y_val, val_probs >= 0.5)
        bac80, _, _, _ = calculate_bac(y_val, val_probs, 0.8)
        score = auc + bac
            
        return {
            'feature_name': feature_name,
            'train_probs': train_probs,
            'val_probs': val_probs,
            'test_probs': test_probs,
            'auc': auc,
            'bac': bac,
            'bac80': bac80,
            'score': score
        }

def train_feature_model(feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed):
    """Train a model for a single feature using cached data."""
    data = get_cached_feature_data(feature_name)
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = XGBClassifier(
        scale_pos_weight=ratio,
        n_jobs=N_JOBS_XGB,
        device=DEVICE,
        n_estimators=100,
        seed=seed,
        max_depth=6,
        subsample=0.9,
        gamma=0.1,
        learning_rate=0.01
    )
    model.fit(data[train_idxs], y_train)
    
    # Generate predictions
    train_probs = model.predict_proba(data[train_idxs])[:, 1]
    val_probs = model.predict_proba(data[val_idxs])[:, 1]
    test_probs = model.predict_proba(data[test_idxs])[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_val, val_probs)
    bac = balanced_accuracy_score(y_val, val_probs >= 0.5)
    bac80, _, _, _ = calculate_bac(y_val, val_probs, 0.8)
    score = auc + bac
        
    return {
        'train_probs': train_probs,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'auc': auc,
        'bac': bac,
        'bac80': bac80,
        'score': score
    }

def train_ensemble_models(feature_combination, train_idxs, val_idxs, test_idxs, y_train, y_val, seed):
    """Train models for a combination of features in parallel and create an ensemble with two-stage training."""
    
    # STAGE 1: Train models on train data, validate on val data, learn meta-learner weights
    print(f"Stage 1: Training individual models and learning meta-learner weights...")
    
    # Prepare arguments for parallel processing (first stage)
    args_list_stage1 = [
        (feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed + i, False)
        for i, feature_name in enumerate(feature_combination)
    ]
    
    # Train feature models in parallel using joblib (thread-based for GPU compatibility)
    feature_results_stage1 = Parallel(n_jobs=min(N_PARALLEL_FEATURES, len(feature_combination)), backend='threading')(
        delayed(train_feature_model_parallel)(args) for args in args_list_stage1
    )
    
    # Sort results to maintain order
    feature_models_stage1 = sorted(feature_results_stage1, key=lambda x: feature_combination.index(x['feature_name']))
    
    # Stack validation probabilities and calibrate them
    
    val_probs_list = [model['val_probs'] for model in feature_models_stage1]
    
    # Calibrate each model's probabilities using isotonic regression
    calibrated_probs = []
    calibrators = []
    for i, probs in enumerate(val_probs_list):
        calibrator = IsotonicRegression(out_of_bounds='clip')
        cal_probs = calibrator.fit_transform(probs, y_val)
        calibrated_probs.append(cal_probs)
        calibrators.append(calibrator)
    
    # Convert calibrated probabilities to logits
    X_meta_val = np.column_stack([logit(np.clip(probs, 0.001, 0.999)) for probs in calibrated_probs])

    # Train logistic regression meta-model to learn weights
    w_simplex = train_simplex_logistic(X_meta_val, y_val)
    meta_model = SimplexLogistic(w_simplex)
    
    # Calculate validation metrics for logging
    meta_val_probs = meta_model.predict_proba(X_meta_val)[:, 1]
    stage1_auc = roc_auc_score(y_val, meta_val_probs)
    stage1_bac = balanced_accuracy_score(y_val, meta_val_probs >= 0.5)
    stage1_bac80, _, _, _ = calculate_bac(y_val, meta_val_probs, 0.8)
    
    print(f"Stage 1 - Meta-learner weights: {w_simplex}")
    print(f"Stage 1 - Validation AUC: {stage1_auc:.4f}, BAC: {stage1_bac:.4f}, BAC80: {stage1_bac80:.4f}")
    
    # === STAGE 1 LOGGING ===
    # Prepare probability data for logging
    raw_val_probs = np.column_stack(val_probs_list)
    calibrated_val_probs = np.column_stack(calibrated_probs)
    logits_val = X_meta_val
    
    stage1_probs = {
        'raw': raw_val_probs,
        'calibrated': calibrated_val_probs, 
        'logits': logits_val,
        'meta': meta_val_probs
    }
    
    
    # # Log stage 1 specific metrics
    # wandb.log({
    #     'stage1/auc': stage1_auc,
    #     'stage1/bac': stage1_bac,
    #     'stage1/bac80': stage1_bac80,
    #     'stage1/weights': wandb.Histogram(w_simplex),
    #     'stage1/meta_probs': wandb.Histogram(meta_val_probs)
    # })
    
    # STAGE 2: Retrain models on train+val data using learned weights, predict on test
    print(f"Stage 2: Retraining models on train+val data for final predictions...")
    
    # Prepare arguments for parallel processing (second stage)
    args_list_stage2 = [
        (feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed + i, True)
        for i, feature_name in enumerate(feature_combination)
    ]
    
    # Retrain feature models on train+val data
    feature_results_stage2 = Parallel(n_jobs=min(N_PARALLEL_FEATURES, len(feature_combination)), backend='threading')(
        delayed(train_feature_model_parallel)(args) for args in args_list_stage2
    )
    
    # Sort results to maintain order
    feature_models_stage2 = sorted(feature_results_stage2, key=lambda x: feature_combination.index(x['feature_name']))
    
    # Remove feature_name from results as it's not needed anymore
    for model in feature_models_stage2:
        del model['feature_name']
    
    # Stack test probabilities and apply calibrators
    test_probs_list = [model['test_probs'] for model in feature_models_stage2]
    
    # Apply calibrators to test probabilities
    calibrated_test_probs = []
    for i, probs in enumerate(test_probs_list):
        cal_test_probs = calibrators[i].transform(probs)
        calibrated_test_probs.append(cal_test_probs)
    
    # Convert calibrated test probabilities to logits
    X_meta_test = np.column_stack([logit(np.clip(probs, 0.001, 0.999)) for probs in calibrated_test_probs])

    # Generate final test predictions using the learned meta-model weights
    meta_test_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    
    # Use validation data to find optimal threshold (from stage 1)
    opt_threshold = find_optimal_threshold(y_val, meta_val_probs)
    meta_test_preds = (meta_test_probs >= opt_threshold).astype(int)
    
    print(f"Stage 2 - Final predictions generated using learned weights")
    
    # === STAGE 2 LOGGING ===
    # Prepare probability data for logging
    raw_test_probs = np.column_stack(test_probs_list)
    calibrated_test_probs_matrix = np.column_stack(calibrated_test_probs)
    logits_test = X_meta_test
    
    stage2_probs = {
        'raw': raw_test_probs,
        'calibrated': calibrated_test_probs_matrix,
        'logits': logits_test,
        'meta': meta_test_probs
    }
    
    
    return {
        'feature_models': feature_models_stage2,  # Final retrained models
        'meta_model': meta_model,
        'calibrators': calibrators,
        'val_probs': meta_val_probs,  # From stage 1 for threshold selection
        'test_probs': meta_test_probs,  # From stage 2 for final evaluation
        'test_preds': meta_test_preds,
        'opt_threshold': opt_threshold,
        'auc': stage1_auc,  # Validation metrics from stage 1
        'bac': stage1_bac,
        'bac80': stage1_bac80,
        'lr_weights': meta_model.coef_[0],
    }
 

def get_train_val_test_indices(description, labels, test_subject, seed):
    """Get train, validation, and test indices for LOSO cross-validation."""
    # Test set: current subject
    test_idxs = np.where(description['subject'] == test_subject)[0]
    
    # Remaining subjects for train/val split
    remaining_idxs = np.where(description['subject'] != test_subject)[0]
    remaining_subjects = np.unique(description['subject'][remaining_idxs])
    
    # Split remaining subjects into train and validation
    train_subjects, val_subjects = train_test_split(
        remaining_subjects, test_size=SPLIT_RATIO, random_state=seed, 
        stratify=[labels[description['subject'] == subj][0] for subj in remaining_subjects]
    )
    
    # Get indices for train and validation sets
    train_idxs = np.where(np.isin(description['subject'], train_subjects))[0]
    val_idxs = np.where(np.isin(description['subject'], val_subjects))[0]
    
    return train_idxs, val_idxs, test_idxs


def save_results(results_df, predictions_df,RUN_NAME, run_n, seed):
    """Save results to CSV files."""
    os.makedirs(f'{LOG_FOLDER}/{PROJECT_NAME}', exist_ok=True)
    results_df.to_csv(f'{LOG_FOLDER}/{PROJECT_NAME}/{RUN_NAME}_run_{run_n}_results_seed_{seed}.csv', index=False)
    predictions_df.to_csv(f'{LOG_FOLDER}/{PROJECT_NAME}/{RUN_NAME}_run_{run_n}_predictions_seed_{seed}.csv', index=False)


def main():
    """Main execution function with combined data preloading."""
    setup_environment()
    
    # Load combined data information
    description, labels, subjects, unique_subjects, subject_labels, _ = load_data_both()
    
    # Store the combined data in global cache
    global _feature_data_cache
    _feature_data_cache = preload_all_feature_data_combined()

    checkpoint=False
    
    # Generate all feature combinations once before starting runs
    all_combinations = generate_feature_combinations()
    print(f"Generated {len(all_combinations)} feature combinations to evaluate")
    print(f"Using {len(unique_subjects)} subjects from combined IPS+Background data")
    
    # Evaluate each feature combination
    for combination in all_combinations:
        for run_n in range(N_RUNS):

            combination_name = '+'.join(combination)
            print(f"\nEvaluating combination: {combination_name}")

            seed = secrets.randbelow(5000)
            random.seed(seed)
            np.random.seed(seed)
            cp.random.seed(seed)

            RUN_NAME = f'{combination_name}_run_{run_n}'
            # if not checkpoint:
            #     if RUN_NAME == 'cc+dwt+mst+sst+gcc_run_1' :    #   for length 5 if crashes -> 'cc+cwt+utm+gcc+gplv_run_3'
            #         checkpoint = True
            #         print(f"Reached checkpoint: {RUN_NAME}, continuing with this and remaining runs...")
            #     else:
            #         print(f"Skipping run {RUN_NAME} (before checkpoint)...")
            #         continue

            # skip_flag = False
            # for existing_run in wandb.Api(timeout=99).runs(path=PROJECT_NAME):
            #     if existing_run.name == RUN_NAME:
            #         print(f"Run {RUN_NAME} already exists, skipping...")
            #         skip_flag = True
            #         break
            # if skip_flag:
            #     continue

            wandb.init(project=PROJECT_NAME, name=RUN_NAME, dir=LOG_FOLDER)

            wandb.config.seed = seed
            wandb.config.combination_length = len(combination)
            wandb.config.combination_name = combination_name
        
            print(f'RUN {run_n+1}/{N_RUNS} - Seed: {seed}')
            
            # Initialize arrays to store predictions for this combination
            y_true_all = []
            y_pred_all = []
            y_prob_all = []
            subject_ids = []
                      
            # Iterate through all subjects (LOSO)
            for subject in unique_subjects:
                
                # Leave current subject out for testing
                train_idxs, val_idxs, test_idxs = get_train_val_test_indices(
                    description, labels, subject, seed
                )
                
                y_train = labels[train_idxs]
                y_val = labels[val_idxs]
                y_test = labels[test_idxs]
                
                # Train ensemble models for this feature combination (now with parallelization)
                ensemble_result = train_ensemble_models(
                    combination, train_idxs, val_idxs, test_idxs, y_train, y_val, seed
                )

                wandb.log({
                    'validation/bac80:': ensemble_result['bac80'],
                    'validation/auc': ensemble_result['auc'],
                    'validation/bac': ensemble_result['bac'],
                    'validation/weights': wandb.Histogram(ensemble_result['lr_weights'])
                })

                print(f'LR Weights: {ensemble_result["lr_weights"]}')
                
                # Store predictions for this subject
                y_true_all.extend(y_test)
                y_pred_all.extend(ensemble_result['test_preds'])
                y_prob_all.extend(ensemble_result['test_probs'])
                subject_ids.extend([subject] * len(y_test))
                
            
            # After LOSO CV is complete for this combination, calculate overall metrics
            print(f"LOSO CV complete for combination: {combination_name}, calculating metrics...")
            
            # Calculate overall performance
            y_true_all = np.array(y_true_all)
            y_pred_all = np.array(y_pred_all)
            y_prob_all = np.array(y_prob_all)
            
            auc = roc_auc_score(y_true_all, y_prob_all)
            bac = balanced_accuracy_score(y_true_all, y_pred_all)
            bac80, fpr, tpr, _ = calculate_bac(y_true_all, y_prob_all, 0.8)
            accuracy = accuracy_score(y_true_all, y_pred_all)
            precision = precision_score(y_true_all, y_pred_all)
            recall = recall_score(y_true_all, y_pred_all)
            f1 = f1_score(y_true_all, y_pred_all)
            
            # Log results
            cm = wandb.plot.confusion_matrix(
                y_true=y_true_all, preds=y_pred_all, class_names=['healthy', 'epileptic']
            )
            
            roc_data = [[f, t] for f, t in zip(fpr, tpr)]
            roc_table = wandb.Table(data=roc_data, columns=["fpr", "tpr"])
            roc_line = wandb.plot.line(roc_table, "fpr", "tpr", title="ROC Curve")
            
            p, r, t = precision_recall_curve(y_true_all, y_prob_all)
            pr_data = [[f, t] for f, t in zip(r, p)]
            pr_table = wandb.Table(data=pr_data, columns=["precision", "recall"])
            pr_line = wandb.plot.line(pr_table, "precision", "recall", title="Precision-Recall Curve")
            
            auprc = auc_sklearn(r, p)
            ap = average_precision_score(y_true_all, y_prob_all)


            wandb.log({
                'auc': auc,
                'bac': bac,
                'bac80': bac80,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'roc_curve': roc_line,
                'precision_recall_curve': pr_line,
                'auprc': auprc,
                'AP': ap
            })
            # Prepare results DataFrame
            results = {
                'run': run_n,
                'seed': seed,
                'auc': auc,
                'bac': bac,
                'bac80': bac80,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auprc': auprc,
                'AP': ap,
            }
            results_df = pd.DataFrame([results])
            # Prepare predictions DataFrame
            predictions_df = pd.DataFrame({
                'subject_id': subject_ids,
                'y_true': y_true_all,
                'y_pred': y_pred_all,
                'y_prob': y_prob_all
            })
            # Save results and predictions
            save_results(results_df, predictions_df, RUN_NAME, run_n, seed)
            print(f"Results for combination {combination_name} saved successfully.")
            # Finish wandb run
            wandb.finish()
            print(f"Run {RUN_NAME} completed successfully.")

if __name__ == "__main__":
    main()