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
DEVICE = 'cuda'
SPLIT_RATIO = 0.3
PROJECT_NAME = 'emc_ensemble_hv_nonresponders'
WANDB_KEY = '96e9a92e52e807ed253b3872afd1de1bafc3640a'
DATA_FOLDER = '/space/gzanardini/emc_hv/'
LOG_FOLDER = '/space/gzanardini/emc/'
N_JOBS_XGB = 4  # Set to 1 for compatibility with CUDA
NUM_WORKERS = 10  # Number of max parallel workers for training
N_PARALLEL_FEATURES = NUM_WORKERS  # Parallel feature training within combination
SCIPY_ARRAY_API=1  # Enable SciPy array API for compatibility with cupy

feature_names = ['cc', 'cwt', 'dwt', 'plv', 'mst', 'sst', 'sr', 'utm', 'gcc', 'gplv']

best_parameters = {
    'sr':       ('BipolarDB',          60 ,    'skew'),
    'cwt':      ('CAR',   2,      'median'),
    'dwt':      ('Laplacian',   60,     'std'),
    'mst':      ('BipolarDB',   60,     'kurtosis'),
    'sst':      ('BipolarDB',   60,     'kurtosis'),
    'cc':       ('Laplacian',   60,      'std'),
    'plv':      ('CAR',   60,      'skew'),
    'gcc':      ('BipolarDB',         10,      'skew'),
    'gplv':     ('BipolarDB',   2,      'kurt'),
    'utm':      ('Cz',   10,     'kurt')
}

def train_simplex_logistic(X, y, max_iter=2500, alpha=1.05):

    d = X.shape[1]
    init_w = np.full(d, 1.0/d)                # uniform start

    # negative log-likelihood  (bias = 0)
    def nll(w):
        logits = X @ w
        ce = -np.sum(y * np.log(expit(logits)) +
                     (1 - y) * np.log(1 - expit(logits)))
       # α = 1 is uniform prior; α > 1 discourages zeros
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

def log_probability_analysis(probs_dict, feature_names, stage_name, subject_id=None):
    """
    Log probability distributions and covariance analysis to wandb.
    
    Args:
        probs_dict: Dictionary with keys like 'raw', 'calibrated', 'logits', 'meta'
        feature_names: List of feature names
        stage_name: String identifier for the stage (e.g., 'stage1_val', 'stage2_test')
        subject_id: Optional subject identifier for LOSO logging
    """
    prefix = f"{stage_name}" 

    # Log histograms for each type of probability
    for prob_type, prob_data in probs_dict.items():
        if prob_data is not None:
            if prob_type == 'meta':
                # Meta predictions are 1D
                wandb.log({
                    f"{prefix}/hist_{prob_type}": wandb.Histogram(prob_data)}, step=subject_id)
            else:
                # Individual feature predictions are 2D
                for i, feature_name in enumerate(feature_names):
                    if i < prob_data.shape[1]:
                        wandb.log({f"{prefix}/hist_{prob_type}_{feature_name}": wandb.Histogram(prob_data[:, i])}, step=subject_id)


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

def load_data():
    """Load and prepare the dataset."""
    description = pd.read_csv(f'{DATA_FOLDER}/description.csv')
    labels = description['epilepsy'].to_numpy()
    subjects = description['subject'].to_numpy()
    unique_subjects = np.unique(description['subject'])
    
    subject_labels = []
    for subj in unique_subjects:
        lbl = labels[subjects == subj][0]
        subject_labels.append([subj, lbl])
    subject_labels = np.array(subject_labels)
    
    return description, labels, subjects, unique_subjects, subject_labels

def load_feature_data(feature_name):
    """Load and preprocess feature data using the best parameters for the given feature."""
    montage, segment_length, combiner = best_parameters[feature_name]
    data = np.load(f'{DATA_FOLDER}/{feature_name}_{montage}_{segment_length}s_{combiner}.npy')
    data = handle_complex_numbers(data)
    
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    return cp.array(data), montage, segment_length, combiner

def get_train_val_test_indices(description, labels, subject, seed):
    """Get indices for train/validation/test splits for LOSO CV."""
    # Load HV responders and filter for only positive responders
    hv_responders = pd.read_csv('/users/gzanardini/eeg_thesis/emc/hv_responders.csv')
    responder_subjects = set(hv_responders[hv_responders['hv_success'] == 0]['subject'].values)
    
    test_idxs = np.where(description['subject'] == subject)[0]
    subjects = description['subject']
    unique_subjects = np.unique(subjects)
    
    # Filter to only include HV responders
    unique_subjects = np.array([subj for subj in unique_subjects if subj in responder_subjects])
    
    other_subjects = [subj for subj in unique_subjects if subj != subject]
    other_subjects_labels = np.array([[subj, labels[subjects == subj][0]] for subj in other_subjects])
    
    train_subjects, val_subjects = train_test_split(
        other_subjects, 
        test_size=SPLIT_RATIO, 
        stratify=other_subjects_labels[:, 1], 
        random_state=seed
    )
    
    train_idxs = np.where(np.isin(description['subject'], train_subjects))[0]
    val_idxs = np.where(np.isin(description['subject'], val_subjects))[0]
    
    return train_idxs, val_idxs, test_idxs

def generate_feature_combinations():
    """Generate combinations of features from 2 to all features."""
    combinations = []
    
    # Generate all combinations of 2 to len(feature_names) features
    for i in range(2, len(feature_names) + 1):
        combs = list(itertools.combinations(feature_names, i))
        for comb in combs:
            combinations.append(list(comb))
    
    return combinations

# Global variable to store preloaded data
_feature_data_cache = {}

def preload_all_feature_data():
    """Preload all feature data to avoid repeated loading."""
    print("Preloading all feature data...")
    global _feature_data_cache
    
    for feature_name in feature_names:
        montage, segment_length, combiner = best_parameters[feature_name]
        data_path = f'{DATA_FOLDER}{feature_name}_{montage}_{segment_length}s_{combiner}.npy'
        data = np.load(data_path)
        data = handle_complex_numbers(data)
        
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        # Convert to cupy array for GPU processing
        _feature_data_cache[feature_name] = cp.array(data)
        print(f"Loaded {feature_name}: {data.shape}")

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
    
    # Log Stage 1 analysis
    #log_probability_analysis(stage1_probs, feature_combination, 'stage1_validation')
    
    # Log stage 1 specific metrics
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
    
    # Log Stage 2 analysis
    #log_probability_analysis(stage2_probs, feature_combination, 'stage2_test')
    
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

def save_results(results_df, predictions_df,RUN_NAME, run_n, seed):
    """Save results to CSV files."""
    os.makedirs(f'{LOG_FOLDER}/{PROJECT_NAME}', exist_ok=True)
    results_df.to_csv(f'{LOG_FOLDER}/{PROJECT_NAME}/{RUN_NAME}_run_{run_n}_results_seed_{seed}.csv', index=False)
    predictions_df.to_csv(f'{LOG_FOLDER}/{PROJECT_NAME}/{RUN_NAME}_run_{run_n}_predictions_seed_{seed}.csv', index=False)

def main():
    """Main execution function with data preloading."""
    setup_environment()
    description, labels, subjects, unique_subjects, subject_labels = load_data()
    
    # Preload all feature data once
    preload_all_feature_data()

    checkpoint=False
    
    # Generate all feature combinations once before starting runs
    all_combinations = generate_feature_combinations()
    print(f"Generated {len(all_combinations)} feature combinations to evaluate")
    
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
            if not checkpoint:
                if RUN_NAME == 'dwt+plv+mst+sst+gcc+gplv_run_4' :    #   
                    checkpoint = True
                    print(f"Reached checkpoint: {RUN_NAME}, continuing with this and remaining runs...")
                else:
                    print(f"Skipping run {RUN_NAME} (before checkpoint)...")
                    continue
            
            # # Skip run if it already exists in wandb with timeout
            # try:
            #     api = wandb.Api(timeout=200)
            #     runs = api.runs(f"{wandb.config.entity}/{PROJECT_NAME}")
            #     if any(run.name == RUN_NAME for run in runs):
            #         print(f"Run {RUN_NAME} already exists in wandb, skipping...")
            #         continue
            # except Exception as e:
            #     print(f"Could not check wandb API: {e}")

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
            predictions = {
                'subject': subject_ids,
                'y_true': y_true_all,
                'y_pred': y_pred_all,
                'y_prob': y_prob_all
            }
            predictions_df = pd.DataFrame(predictions)
            # Save results and predictions
            save_results(results_df, predictions_df, RUN_NAME, run_n, seed)
            print(f"Results for combination {combination_name} saved successfully.")
            # Finish wandb run
            wandb.finish()
            print(f"Run {RUN_NAME} completed successfully.")

if __name__ == "__main__":
    main()