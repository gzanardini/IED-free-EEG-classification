import os
import wandb.plot
import numpy as np 
import pandas as pd
import itertools 
import secrets
import cupy as cp
from utils.single_set_common import (
    build_loocv_splits,
    calculate_bac as common_calculate_bac,
    handle_complex_numbers as common_handle_complex_numbers,
    load_subject_data,
    load_feature_array,
    log_metrics_to_wandb,
    save_predictions_csv,
    setup_environment as common_setup_environment,
    train_xgb_folds,
)

# Configuration
N_RUNS = 5
N_CUDA = 1
N_JOBS_XGB= 1  # Set to 1 for single GPU usage

DATA_FOLDER = '/space/gzanardini/emc_whole/split/'
PROJECT_NAME = 'emc_singleset_final'
LOG_FOLDER = '/space/gzanardini/emc/'

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10, 20, 60]
feature_names = ['cc', 'cwt', 'dwt', 'gcc', 'gplv', 'plv', 'mst', 'sst', 'spectral', 'utm']
combiners=['mean', 'median', 'std', 'skew', 'kurt']


def setup_environment():
    common_setup_environment(N_CUDA)

def calculate_bac(labels, scores, sens_thresh):
    return common_calculate_bac(labels, scores, sens_thresh)

def handle_complex_numbers(features):
    return common_handle_complex_numbers(features)

def load_feature_data(feature_name, montage, segment_length, combiner):
    return load_feature_array(DATA_FOLDER, feature_name, montage, segment_length, combiner)

def train_and_evaluate(features, labels,seed):
    split_iterator = []
    for ss in range(len(labels)):
        print(f'Fold {ss}')
    split_iterator = build_loocv_splits(len(labels))
    model_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'device': f'cuda:{N_CUDA}',
        'subsample': 0.9,
        'n_jobs': 4,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'as_device_array': cp.array,
        'append_mode': 'append',
    }
    return train_xgb_folds(features, labels, split_iterator, seed, model_params)

def log_metrics(y_tests, y_preds, y_scores):
    return log_metrics_to_wandb(y_tests, y_preds, y_scores)

def save_predictions(y_preds, y_scores, y_tests, montage, feature_name, segment_length,combiner, run_n, seed):
    output_dir = f'{LOG_FOLDER}{PROJECT_NAME}/'
    save_predictions_csv(
        y_preds,
        y_scores,
        y_tests,
        output_dir,
        (montage, feature_name, segment_length, combiner),
        run_n,
        seed,
    )

def main():
    """Main execution function."""
    setup_environment()
    description, labels, subjects, unique_subjects, subject_labels = load_subject_data(DATA_FOLDER)
    
    for montage, feature_name, segment_length, combiner in itertools.product(montages, feature_names, segment_lengths, combiners):
        features = load_feature_data(feature_name, montage, segment_length, combiner)
        
        for run_n in range(N_RUNS):
            print(f'Run {run_n} - {montage} - {feature_name} - {segment_length}s -')
            
            wandb.init(
                project=PROJECT_NAME,
                name=f'{feature_name}_{montage}_{segment_length}s_{combiner}run_{run_n}',
                reinit=True
            )
            
            seed = secrets.randbelow(5000)
            np.random.seed(seed)
            cp.random.seed(seed)
            
            wandb.config.update({
                'seed': seed,
                'montage': montage,
                'feature_name': feature_name,
                'segment_length': segment_length,
                'combiner': combiner,
                'epochs' : False
            })
            
            y_preds, y_scores, y_tests = train_and_evaluate(features, labels, seed)
            
            save_predictions(y_preds, y_scores, y_tests, montage, feature_name, segment_length,combiner, run_n, seed)

            print(f'Y_preds shape: {y_preds.shape}')
            print(f'Y_scores shape: {y_scores.shape}')
            print(f'Y_tests shape: {y_tests.shape}')
            
            print(f'y_preds: {y_preds}')
            print(f'y_scores: {y_scores}')
            print(f'y_tests: {y_tests}')
            
            metrics = log_metrics(y_tests, y_preds, y_scores)
            
            # Print summary
            print('###############################')
            print(f'BAC: {metrics[0]:.4f}')
            print(f'BAC80: {metrics[1]:.4f}')
            print(f'AUC: {metrics[2]:.4f}')
            print(f'Score (AUC+BAC80): {metrics[3]:.4f}')
            print(f'Recall: {metrics[4]:.4f}')
            print(f'Precision: {metrics[5]:.4f}')
            print('###############################')
            
            wandb.finish()

if __name__ == "__main__":
    main()
