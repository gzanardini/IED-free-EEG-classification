import os
import wandb.plot
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import itertools 
import secrets
from sklearn.model_selection import train_test_split
import warnings
import cupy as cp
from cupy.cuda import Device
import random

np.set_printoptions(linewidth=200, precision=4)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
N_RUNS = 5
N_CUDA = 0
SPLIT_RATIO = 0.3
PROJECT_NAME = 'tuh_concat'
WANDB_KEY = '96e9a92e52e807ed253b3872afd1de1bafc3640a'
DATA_FOLDER = '/space/gzanardini/tuh_whole/split'
LOG_FOLDER = '/space/gzanardini/tuh/'
N_JOBS_XGB = 16

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10, 20, 60]
feature_names = ['cc', 'cwt', 'dwt', 'plv', 'mst', 'sst', 'spectral', 'utm', 'gcc', 'gplv']
combiners = ['mean', 'median', 'std', 'skew', 'kurt']

best_parameters = {
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

def setup_environment():
    """Initialize CUDA and wandb."""
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
    return data, montage, segment_length, combiner

def get_train_val_test_indices(description, labels, subject, seed):
    """Get indices for train/validation/test splits for LOSO CV."""
    test_idxs = np.where(description['subject'] == subject)[0]
    subjects = description['subject']
    unique_subjects = np.unique(subjects)
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
        data_path = f'{DATA_FOLDER}/{feature_name}_{montage}_{segment_length}s_{combiner}.npy'
        data = np.load(data_path)
        data = handle_complex_numbers(data)
        
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        # Store as numpy array (no need for GPU for concatenation)
        _feature_data_cache[feature_name] = data
        print(f"Loaded {feature_name}: {data.shape}")

def get_cached_feature_data(feature_name):
    """Get preloaded feature data."""
    return _feature_data_cache[feature_name]

def concatenate_features(feature_combination):
    """Concatenate multiple feature sets along axis 1."""
    feature_arrays = []
    
    for feature_name in feature_combination:
        data = get_cached_feature_data(feature_name)
        feature_arrays.append(data)
    
    # Concatenate along axis 1 (features)
    concatenated_features = np.concatenate(feature_arrays, axis=1)
    
    # Handle any remaining NaN values by replacing with 0
    concatenated_features = np.nan_to_num(concatenated_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return concatenated_features

def train_concatenated_model(feature_combination, train_idxs, val_idxs, test_idxs, y_train, y_val, seed):
    """Train a single model on concatenated features."""
    
    # Concatenate features for this combination
    X_concat = concatenate_features(feature_combination)
    
    # Convert to cupy for GPU processing
    X_concat_gpu = cp.array(X_concat)
    
    # Calculate class weights
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    # Train XGBoost model
    model = XGBClassifier(
        scale_pos_weight=ratio,
        n_jobs=N_JOBS_XGB,
        device=f'cuda:{N_CUDA}',
        n_estimators=100,
        seed=seed,
        max_depth=6,
        subsample=0.9,
        gamma=0.1,
        learning_rate=0.01
    )
    
    model.fit(X_concat_gpu[train_idxs], y_train)
    
    # Generate predictions
    val_probs = model.predict_proba(X_concat_gpu[val_idxs])[:, 1]
    test_probs = model.predict_proba(X_concat_gpu[test_idxs])[:, 1]
    
    # Calculate metrics on validation set
    auc = roc_auc_score(y_val, val_probs)
    bac = balanced_accuracy_score(y_val, val_probs >= 0.5)
    bac80, _, _, _ = calculate_bac(y_val, val_probs, 0.8)
    
    # Find optimal threshold on validation set
    opt_threshold = find_optimal_threshold(y_val, val_probs)
    
    # Apply optimal threshold to test predictions
    test_preds = (test_probs >= opt_threshold).astype(int)
    
    return {
        'model': model,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'test_preds': test_preds,
        'opt_threshold': opt_threshold,
        'auc': auc,
        'bac': bac,
        'bac80': bac80,
        'n_features': X_concat.shape[1]
    }

def save_results(results_df, predictions_df, RUN_NAME, run_n, seed):
    """Save results to CSV files."""
    os.makedirs(f'{LOG_FOLDER}/{PROJECT_NAME}', exist_ok=True)
    results_df.to_csv(f'{LOG_FOLDER}/{PROJECT_NAME}/{RUN_NAME}_run_{run_n}_results_seed_{seed}.csv', index=False)
    predictions_df.to_csv(f'{LOG_FOLDER}/{PROJECT_NAME}/{RUN_NAME}_run_{run_n}_predictions_seed_{seed}.csv', index=False)

def main():
    """Main execution function with concatenated features."""
    setup_environment()
    description, labels, subjects, unique_subjects, subject_labels = load_data()
    
    # Preload all feature data once
    preload_all_feature_data()
    
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
            
            wandb.init(project=PROJECT_NAME, name=RUN_NAME, reinit=True)

            wandb.config.seed = seed
            wandb.config.combination_length = len(combination)
            wandb.config.combination_name = combination_name
        
            print(f'RUN {run_n+1}/{N_RUNS} - Seed: {seed}')
            
            # Initialize arrays to store predictions for this combination
            y_true_all = []
            y_pred_all = []
            y_prob_all = []
            subject_ids = []
            
            # Perform LOSO CV for this combination
            print(f"Running LOSO CV for combination: {combination_name}")
            
            # Iterate through all subjects (LOSO)
            for subject in unique_subjects:
                print(f'Processing subject {subject} with {combination_name}')
                
                # Leave current subject out for testing
                train_idxs, val_idxs, test_idxs = get_train_val_test_indices(
                    description, labels, subject, seed
                )
                
                y_train = labels[train_idxs]
                y_val = labels[val_idxs]
                y_test = labels[test_idxs]
                
                # Train concatenated model for this feature combination
                model_result = train_concatenated_model(
                    combination, train_idxs, val_idxs, test_idxs, y_train, y_val, seed
                )

                wandb.log({
                    'validation/bac80': model_result['bac80'],
                    'validation/auc': model_result['auc'],
                    'validation/bac': model_result['bac'],
                    'validation/opt_threshold': model_result['opt_threshold'],
                    'n_features': model_result['n_features']
                })

                print(f'Model trained with {model_result["n_features"]} concatenated features')
                print(f'Validation AUC: {model_result["auc"]:.4f}, BAC: {model_result["bac"]:.4f}, BAC80: {model_result["bac80"]:.4f}')
                
                # Store predictions for this subject
                y_true_all.extend(y_test)
                y_pred_all.extend(model_result['test_preds'])
                y_prob_all.extend(model_result['test_probs'])
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
            pr_data = [[f, t] for f, t in zip(p, r)]
            pr_table = wandb.Table(data=pr_data, columns=["precision", "recall"])
            pr_line = wandb.plot.line(pr_table, "precision", "recall", title="Precision-Recall Curve")
            

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
                'precision_recall_curve': pr_line
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
                'f1_score': f1
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
