'''
import os
import wandb.plot
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    balanced_accuracy_score,
    average_precision_score,
)
from sklearn.metrics import auc as auc_sklearn
import itertools
import secrets
from sklearn.model_selection import train_test_split
import warnings
import cupy as cp
import random
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from utils.ensemble_common import (
    SimplexLogistic,
    calculate_bac,
    find_optimal_threshold,
    handle_complex_numbers,
    save_ensemble_results,
    setup_environment,
    train_simplex_logistic,
)
from utils.ensemble_pipeline import (
    train_ensemble_models as shared_train_ensemble_models,
    train_feature_model_parallel as shared_train_feature_model_parallel,
)
from utils.ensemble_runner import (
    build_train_val_test_indices,
    generate_feature_combinations,
    get_cached_feature_data,
    load_subject_data,
    preload_feature_data_cache,
)

np.set_printoptions(linewidth=200, precision=4)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
N_RUNS = 5
N_CUDA = 0
DEVICE = 'cpu'
SPLIT_RATIO = 0.3
PROJECT_NAME = 'tuh_background_noieds'
DATA_FOLDER = '/space/gzanardini/tuh_background/split'
LOG_FOLDER = '/space/gzanardini/tuh/'
N_JOBS_XGB = 4  # Set to 1 for compatibility with CUDA
NUM_WORKERS = 10  # Number of parallel workers for training
N_PARALLEL_FEATURES = NUM_WORKERS  # Parallel feature training within combination
SCIPY_ARRAY_API = 1  # Enable SciPy array API for compatibility with cupy
SIMPLEX_ALPHA = 1.05

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10, 20, 60]
feature_names = ['cc', 'cwt', 'dwt', 'plv', 'mst', 'sst', 'spectral', 'utm', 'gcc', 'gplv']
combiners = ['mean', 'median', 'std', 'skew', 'kurt']

# Best parameters from tuh_loso_whole_noIED.py
best_parameters = {
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

subjects_to_skip = ['aaaaajgj', 'aaaaakcd']


def get_train_val_test_indices(description, labels, subject, seed):
    """Get indices for train/validation/test splits for LOSO CV."""
    return build_train_val_test_indices(description, labels, subject, SPLIT_RATIO, seed)


# Global variable to store preloaded data
_feature_data_cache = {}

def train_feature_model_parallel(args):
    """Wrapper function for parallel feature model training."""
    seed = args[6]
    random.seed(seed)
    np.random.seed(seed)
    return shared_train_feature_model_parallel(
        args,
        cache=_feature_data_cache,
        n_jobs_xgb=N_JOBS_XGB,
        device=DEVICE,
    )

def train_feature_model(feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed):
    """Train a model for a single feature using cached data."""
    data = get_cached_feature_data(_feature_data_cache, feature_name)
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
    def log_stage1(stage1):
        wandb.log(
            {
                'stage1/auc': stage1['auc'],
                'stage1/bac': stage1['bac'],
                'stage1/bac80': stage1['bac80'],
                'stage1/weights': wandb.Histogram(stage1['weights']),
                'stage1/meta_probs': wandb.Histogram(stage1['meta_probs']),
            }
        )

    def log_stage2(stage2):
        wandb.log(
            {
                'stage2/meta_probs': wandb.Histogram(stage2['meta_probs']),
                'stage2/meta_preds': wandb.Histogram(stage2['meta_preds']),
                'stage2/opt_threshold': stage2['opt_threshold'],
            }
        )

    return shared_train_ensemble_models(
        feature_combination,
        train_idxs,
        val_idxs,
        test_idxs,
        y_train,
        y_val,
        seed,
        cache=_feature_data_cache,
        n_jobs_xgb=N_JOBS_XGB,
        device=DEVICE,
        n_parallel_features=N_PARALLEL_FEATURES,
        simplex_alpha=SIMPLEX_ALPHA,
        calculate_bac=calculate_bac,
        find_optimal_threshold=find_optimal_threshold,
        train_simplex_logistic=train_simplex_logistic,
        SimplexLogistic=SimplexLogistic,
        log_stage1=log_stage1,
        log_stage2=log_stage2,
    )


def main():
    """Main execution function with data preloading."""
    setup_environment(DEVICE, N_CUDA)
    description, labels, subjects, unique_subjects, subject_labels = load_data(no_ied=True)
    
    # Preload all feature data once
    preload_all_feature_data()
    
    # Generate all feature combinations once before starting runs
    all_combinations = generate_feature_combinations(feature_names)
    print(f"Generated {len(all_combinations)} feature combinations to evaluate")
    
    # Evaluate each feature combination
    for combination in all_combinations:
        print(f"\nEvaluating combination: {'+'.join(combination)}")

        for run_n in range(N_RUNS):

            combination_name = '+'.join(combination)

            seed = secrets.randbelow(5000)
            random.seed(seed)
            np.random.seed(seed)
            cp.random.seed(seed)

            RUN_NAME = f'{combination_name}_run_{run_n}'
            # Check if run already exists in wandb
            try:
                api = wandb.Api(timeout=29)
                existing_runs = {run.name for run in api.runs(path=PROJECT_NAME)}
                if RUN_NAME in existing_runs:
                    print(f"Run {RUN_NAME} already exists, skipping...")
                    continue
            except Exception as e:
                print(f"Warning: Could not check existing runs ({e}). Proceeding anyway...")

            wandb.init(project=PROJECT_NAME, name=RUN_NAME, reinit=True, dir=LOG_FOLDER)

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

                if subject in subjects_to_skip:
                    continue

                # Leave current subject out for testing
                train_idxs, val_idxs, test_idxs = get_train_val_test_indices(
                    description, labels, subject, seed
                )
                
                y_train = labels[train_idxs]
                y_val = labels[val_idxs]
                y_test = labels[test_idxs]
                
                print(f"Subject {subject}: Train={len(train_idxs)}, Val={len(val_idxs)}, Test={len(test_idxs)}")
                
                # Train ensemble models with two-stage approach
                ensemble_results = train_ensemble_models(combination, train_idxs, val_idxs, test_idxs, y_train, y_val, seed)
                
                if ensemble_results is None:
                    print(f"Failed to train models for subject {subject}, skipping...")
                    continue
                
                # Extract results
                test_probs = ensemble_results['test_probs']
                test_preds = ensemble_results['test_preds']
                
                # Store results
                y_true_all.extend(y_test)
                y_pred_all.extend(test_preds)
                y_prob_all.extend(test_probs)
                subject_ids.extend([subject] * len(y_test))
                
                # Log per-subject results
                if len(np.unique(y_test)) > 1:  # Only if test set has both classes
                    subject_auc = roc_auc_score(y_test, test_probs)
                    subject_bac = balanced_accuracy_score(y_test, test_preds)
                    wandb.log({
                        f'subject_{subject}_auc': subject_auc,
                    }, step=len(y_true_all))
                
                print(f"Subject {subject} completed. Test samples: {len(y_test)}")
            
            # Calculate overall performance
            y_true_all = np.array(y_true_all)
            y_pred_all = np.array(y_pred_all)
            y_prob_all = np.array(y_prob_all)
            
            auc = roc_auc_score(y_true_all, y_prob_all)
                   
                print(f"Subject {subject} completed. Test samples: {len(y_test)}")
            
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
            save_ensemble_results(results_df, predictions_df, LOG_FOLDER, PROJECT_NAME, RUN_NAME, run_n, seed)
            print(f"Results for combination {combination_name} saved successfully.")
            # Finish wandb run
            wandb.finish()
            print(f"Run {RUN_NAME} completed successfully.")

if __name__ == "__main__":
    main()
'''

import cupy as cp

from utils.model_training import EnsembleExperimentConfig, run_ensemble_experiment


N_RUNS = 5
N_CUDA = 0
DEVICE = "cpu"
PROJECT_NAME = "tuh_background_noieds"
DATA_FOLDER = "/space/gzanardini/tuh_background/split"
LOG_FOLDER = "/space/gzanardini/tuh/"
N_JOBS_XGB = 4
NUM_WORKERS = 10
SIMPLEX_ALPHA = 1.05
SUBJECTS_TO_SKIP = ["aaaaajgj", "aaaaakcd"]

FEATURE_NAMES = ["cc", "cwt", "dwt", "plv", "mst", "sst", "spectral", "utm", "gcc", "gplv"]

BEST_PARAMETERS = {
    "spectral": ("BipolarDB", 2, "kurt"),
    "cwt": ("Cz", 1, "skew"),
    "dwt": ("Cz", 10, "skew"),
    "mst": ("BipolarDB", 1, "median"),
    "sst": ("Laplacian", 20, "std"),
    "cc": ("CAR", 120, "mean"),
    "plv": ("Cz", 60, "std"),
    "gcc": ("Cz", 20, "kurt"),
    "gplv": ("Laplacian", 10, "mean"),
    "utm": ("Laplacian", 60, "median"),
}

def build_config():
    return EnsembleExperimentConfig(
        dataset_name="tuh_background_noieds",
        project_name=PROJECT_NAME,
        log_folder=LOG_FOLDER,
        n_runs=N_RUNS,
        run_name_template="{feature_set}_run_{run_n}",
        device=DEVICE,
        cuda_idx=N_CUDA,
        wandb_reinit=True,
        data_folder=DATA_FOLDER,
        source_type="single_source",
        feature_names=FEATURE_NAMES,
        best_parameters=BEST_PARAMETERS,
        cache_array_converter=cp.array,
        n_jobs_xgb=N_JOBS_XGB,
        n_parallel_features=NUM_WORKERS,
        simplex_alpha=SIMPLEX_ALPHA,
        subjects_to_skip=SUBJECTS_TO_SKIP,
        xgb_params={
            "n_estimators": 100,
            "max_depth": 6,
            "subsample": 0.9,
            "gamma": 0.1,
            "learning_rate": 0.01,
        },
    )


def main():
    run_ensemble_experiment(build_config())


if __name__ == "__main__":
    main()
