import os
import wandb.plot
import numpy as np 
import pandas as pd
import itertools 
import secrets
import cupy as cp
from utils.single_set_common import (
    build_subject_loso_splits,
    load_subject_data,
    load_feature_array,
    log_metrics_to_wandb,
    save_predictions_csv,
    setup_environment,
    train_xgb_folds,
)

# Configuration
N_RUNS = 5
N_CUDA = 1
PROJECT_NAME = 'tuh_ss_bg'
FEAT_FOLDER = '/space/gzanardini/tuh_background/split/'
LOG_FOLDER = '/space/gzanardini/tuh'

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10, 20, 60, 120]
feature_names = ['cc', 'cwt', 'dwt', 'gcc', 'gplv', 'plv', 'mst', 'sst', 'spectral', 'utm']
combiners=['mean', 'median', 'std', 'skew', 'kurt']

def load_feature_data(feature_name, montage, segment_length, combiner):
    return load_feature_array(FEAT_FOLDER, feature_name, montage, segment_length, combiner)

def train_and_evaluate(features, labels, subjects, unique_subjects, description, seed):
    split_iterator = []
    for ss, subject in enumerate(unique_subjects):
        print(f'Iteration {ss+1} - Subject: {subject}')
    split_iterator = build_subject_loso_splits(description['subject'], unique_subjects)
    model_params = {
        'n_estimators': 100,
        'max_depth': 7,
        'device': f'cuda:{N_CUDA}',
        'subsample': 0.8,
        'n_jobs': 4,
        'gamma': 0.1,
        'learning_rate': 0.05,
        'as_device_array': cp.array,
        'append_mode': 'extend',
    }
    return train_xgb_folds(features, labels, split_iterator, seed, model_params)

def log_metrics(y_tests, y_preds, y_scores):
    return log_metrics_to_wandb(y_tests, y_preds, y_scores)

def save_predictions(y_preds, y_scores, y_tests, montage, feature_name, segment_length, combiner, run_n, seed):
    output_dir = f'{LOG_FOLDER}/{PROJECT_NAME}/'
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
    setup_environment(N_CUDA)
    description, labels, subjects, unique_subjects, subject_labels = load_subject_data(FEAT_FOLDER)
    
    for montage, feature_name, segment_length, combiner in itertools.product(montages, feature_names, segment_lengths, combiners):
               # if feature_name is gcc and montage is CAR or Laplacian and segment_length is 1: skip
        if feature_name == 'gcc' and montage in ['CAR', 'Laplacian'] and segment_length == 1:
            print(f'Skipping combination: {feature_name}, {montage}, {segment_length}s, {combiner} --- Not valid')
            continue
       
        features = load_feature_data(feature_name, montage, segment_length, combiner)

        for run_n in range(N_RUNS):   
            print(f'Run {run_n} - {montage} - {feature_name} - {segment_length}s - {combiner}')
        
            wandb.init(
                project=PROJECT_NAME,
                name=f'{feature_name}_{montage}_{segment_length}s_{combiner}_run_{run_n}',
                reinit=True,
                dir=LOG_FOLDER
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
                'epochs': False
            })
            
            y_preds, y_scores, y_tests = train_and_evaluate(
                features, labels, subjects, unique_subjects, description, seed
            )

            save_predictions(y_preds, y_scores, y_tests, montage, feature_name, segment_length, combiner, run_n, seed)

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
