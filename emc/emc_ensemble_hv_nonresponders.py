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
DEVICE = 'cuda'
SPLIT_RATIO = 0.3
PROJECT_NAME = 'emc_ensemble_hv_nonresponders'
DATA_FOLDER = '/space/gzanardini/emc_hv/'
LOG_FOLDER = '/space/gzanardini/emc/'
N_JOBS_XGB = 4  # Set to 1 for compatibility with CUDA
NUM_WORKERS = 10  # Number of max parallel workers for training
N_PARALLEL_FEATURES = NUM_WORKERS  # Parallel feature training within combination
SCIPY_ARRAY_API=1  # Enable SciPy array API for compatibility with cupy
SIMPLEX_ALPHA = 1.05

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




def get_train_val_test_indices(description, labels, subject, seed):
    """Get indices for train/validation/test splits for LOSO CV."""
    # Load HV responders and filter for only positive responders
    hv_responders = pd.read_csv('/users/gzanardini/eeg_thesis/emc/hv_responders.csv')
    responder_subjects = set(hv_responders[hv_responders['hv_success'] == 0]['subject'].values)
    
    subjects = description['subject']
    unique_subjects = np.unique(subjects)

    # Filter to only include HV responders
    unique_subjects = np.array([subj for subj in unique_subjects if subj in responder_subjects])

    return build_train_val_test_indices(
        description,
        labels,
        subject,
        SPLIT_RATIO,
        seed,
        unique_subjects=unique_subjects,
    )

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
        log_stage1=None,
        log_stage2=None,
    )


def main():
    """Main execution function with data preloading."""
    setup_environment(DEVICE, N_CUDA)
    description, labels, subjects, unique_subjects, subject_labels = load_subject_data(DATA_FOLDER)
    
    # Preload all feature data once
    global _feature_data_cache
    _feature_data_cache = preload_feature_data_cache(
        feature_names,
        best_parameters,
        DATA_FOLDER,
        cp.array,
    )

    checkpoint=False
    
    # Generate all feature combinations once before starting runs
    all_combinations = generate_feature_combinations(feature_names)
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