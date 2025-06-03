import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
from concurrent.futures import ThreadPoolExecutor

np.set_printoptions(linewidth=200, precision=4)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
N_RUNS = 10
N_CUDA = 0
SPLIT_RATIO = 0.3
PROJECT_NAME = 'emc_LOSO_nomwu'
RUN_NAME = 'whole'
WANDB_KEY = '96e9a92e52e807ed253b3872afd1de1bafc3640a'
DATA_FOLDER = '/space/gzanardini/emc_dataset/'
LOG_FOLDER = '/space/gzanardini/emc/'
N_JOBS_XGB = 1  # Set to 1 for compatibility with CUDA
NUM_WORKERS = 16  # Number of parallel workers for training

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10, 20, 60]
feature_names = ['cc', 'cwt', 'dwt', 'plv', 'mst', 'sst', 'spectral', 'utm', 'gcc', 'gplv']

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
    """Load the labels data."""
    return np.load(f'{DATA_FOLDER}labels.npy')

def load_feature_data(feature_name, montage, segment_length):
    """Load and preprocess feature data."""
    data = np.load(f'{DATA_FOLDER}{feature_name}_{montage}_{segment_length}s.npy')
    data = handle_complex_numbers(data)
    
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    return cp.array(data)

def get_train_val_test_indices(labels, test_idx, seed):
    """Get indices for train/validation/test splits for LOSO CV."""
    other_idxs = np.delete(np.arange(len(labels)), test_idx)
    train_idxs, val_idxs = train_test_split(
        other_idxs, 
        test_size=SPLIT_RATIO, 
        stratify=labels[other_idxs], 
        random_state=seed
    )
    return train_idxs, val_idxs

def train_single_classifier(feature_name, montage, segment_length, train_idxs, val_idxs, y_train, y_val, seed):
    """Train a single XGBoost classifier."""
    print(f'Feature: {feature_name}, Montage: {montage}, Segment Length: {segment_length}')
    
    data = load_feature_data(feature_name, montage, segment_length)
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
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
    model.fit(data[train_idxs], y_train)
    
    y_prob = model.predict_proba(data[val_idxs])[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    bac=balanced_accuracy_score(y_val,y_prob >= 0.5)
    bac80, fpr, tpr, thresholds = calculate_bac(y_val, y_prob, 0.8)
    score = auc + bac
    
    return auc, bac80, score, model, data

def train_best_classifiers(train_idxs, val_idxs, test_idx, y_train, y_val, seed):
    """Train classifiers for all feature/montage/segment combinations and keep the best."""
    best_classifiers = {}
    run_results = []
    
    for feature_name in feature_names:
        best_score = 0
        best_classifier_data = None
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS, thread_name_prefix='xgb_train') as executor:
            futures = []
            for montage, segment_length in itertools.product(montages, segment_lengths):
                futures.append(executor.submit(
                    train_single_classifier, feature_name, montage, segment_length,
                    train_idxs, val_idxs, y_train, y_val, seed
                ))
            
            for future in futures:
                auc, bac80, score, model, data = future.result()
                
                if score > best_score:
                    print(f'New best score (AUC+BAC80) for {feature_name}: {score:.4f}')
                    best_score = score
                    best_classifier_data = (auc, model, data[val_idxs], data[test_idx], bac80, score)
        
        if best_classifier_data is not None:
            best_classifiers[feature_name] = best_classifier_data
    
    return best_classifiers, run_results

def make_ensemble_predictions(best_classifiers, y_val):
    """Make ensemble predictions using best classifiers."""
    X_train_lr = []
    X_test_lr = []
    
    for feature_name in best_classifiers:
        auc, model, val_data, test_data, bac80, score = best_classifiers[feature_name]
        
        val_prob = model.predict_proba(val_data)[:, 1]
        test_prob = model.predict_proba(test_data)[:, 1]
        
        X_train_lr.append(val_prob)
        X_test_lr.append(test_prob)
    
    X_train_lr = np.array(X_train_lr).T
    X_test_lr = np.array(X_test_lr).T
    
    # Scale features
    scaler = MinMaxScaler(clip=True)
    X_train_lr = scaler.fit_transform(X_train_lr)
    X_test_lr = scaler.transform(X_test_lr)
    
    # Find optimal thresholds
    opt_thresholds = [find_optimal_threshold(y_val, X_train_lr[:, col]) 
                     for col in range(X_train_lr.shape[1])]
    print(f'Optimal thresholds: {opt_thresholds}')
    
    # Apply thresholds
    y_test_preds = np.where(
        np.array([X_test_lr[:, i] > opt_thresholds[i] for i in range(X_test_lr.shape[1])]).T,
        1, 0
    )
    
    y_test_prob = np.mean(y_test_preds, axis=1)
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    
    return y_test_pred, y_test_prob

def log_metrics(y_true, y_pred, y_prob):
    """Calculate and log metrics to wandb."""
    bac = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    bac80, fpr, tpr, thresholds = calculate_bac(y_true, y_prob, 0.8)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    cm = wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, 
                                   class_names=['healthy', 'epileptic'])
    
    roc_data = [[f, t] for f, t in zip(fpr, tpr)]
    roc_table = wandb.Table(data=roc_data, columns=["fpr", "tpr"])
    roc_line = wandb.plot.line(roc_table, "fpr", "tpr", title="ROC Curve")
    
    p, r, t = precision_recall_curve(y_true, y_prob)
    pr_data = [[f, t] for f, t in zip(p, r)]
    pr_table = wandb.Table(data=pr_data, columns=["precision", "recall"])
    pr_line = wandb.plot.line(pr_table, "precision", "recall", title="Precision-Recall Curve")
    
    wandb.log({
        'BAC': bac,
        'AUC': auc,
        'Accuracy': accuracy,
        'BAC80': bac80,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Confusion Matrix': cm,
        'ROC Curve': roc_line,
        'Precision-Recall Curve': pr_line
    })
    
    return bac, auc, accuracy, bac80, precision, recall, f1

def save_results(run_summary, prediction_summary, run_n, seed):
    """Save results to CSV files."""
    os.makedirs(LOG_FOLDER, exist_ok=True)
    run_summary.to_csv(f'{LOG_FOLDER}{RUN_NAME}_run_{run_n}_seed_{seed}.csv', index=False)
    prediction_summary.to_csv(f'{LOG_FOLDER}{RUN_NAME}_run_{run_n}_predictions_seed_{seed}.csv', index=False)

def run_loso_cv(labels, seed):
    """Run Leave-One-Subject-Out cross-validation."""
    run_summary = pd.DataFrame(columns=['montage', 'feature_name', 'segment_length', 'bac', 'bac80', 'auc'])
    prediction_summary = pd.DataFrame(columns=['subject', 'y_pred', 'y_prob', 'y_true'])
    
    for ss in range(len(labels)):
        print(f'Processing subject {ss}')
        
        test_idx = [ss]
        train_idxs, val_idxs = get_train_val_test_indices(labels, test_idx, seed)
        
        y_train = labels[train_idxs]
        y_val = labels[val_idxs]
        y_test = labels[test_idx]
        
        best_classifiers, run_results = train_best_classifiers(
            train_idxs, val_idxs, test_idx, y_train, y_val, seed
        )
        
        # Log individual feature performance
        for feature_name in best_classifiers:
            auc, _, _, _, bac80, score = best_classifiers[feature_name]
            wandb.log({
                f'aucs/{feature_name}': auc,
                f'bac80/{feature_name}': bac80,
                f'score/{feature_name}': score
            }, step=ss)
        
        # Add results to summary
        for result in run_results:
            newline = pd.DataFrame({**result, 'subject': ss}, index=[0])
            run_summary = pd.concat([run_summary, newline], ignore_index=True)
        
        y_test_pred, y_test_prob = make_ensemble_predictions(best_classifiers, y_val)
        
        print(f'Final predictions for fold {ss}: {y_test_pred}')
        print(f'Final probabilities for fold {ss}: {y_test_prob}')
        print(f'Ground truths for fold {ss}: {y_test}')
        
        pred_df = pd.DataFrame({
            'subject': ss,
            'y_pred': y_test_pred,
            'y_prob': y_test_prob,
            'y_true': y_test
        })
        prediction_summary = pd.concat([prediction_summary, pred_df], ignore_index=True)
    
    return run_summary, prediction_summary

def main():
    """Main execution function."""
    setup_environment()
    labels = load_data()
    
    for run_n in range(N_RUNS):
        seed = secrets.randbelow(5000)
        random.seed(seed)
        np.random.seed(seed)
        cp.random.seed(seed)
        
        wandb.init(project=PROJECT_NAME, name=f'{RUN_NAME}_run_{run_n}', reinit=True)
        wandb.config.seed = seed
        
        print(f'RUN {run_n+1} - Seed: {seed}')
        
        run_summary, prediction_summary = run_loso_cv(labels, seed)
        
        # Evaluate results
        y_preds = np.array(prediction_summary['y_pred']).astype(int)
        y_true = np.array(prediction_summary['y_true']).astype(int)
        y_probs = np.array(prediction_summary['y_prob']).astype(float)
        
        bac, auc, accuracy, bac80, precision, recall, f1 = log_metrics(y_true, y_preds, y_probs)
        
        save_results(run_summary, prediction_summary, run_n, seed)
        
        print('###############################')
        print(f'Final BAC: {bac}')
        print(f'Final AUC: {auc}')
        print(f'Final Accuracy: {accuracy}')
        print(f'Final BAC80: {bac80}')
        print(f'Final Precision: {precision}')
        print(f'Final Recall: {recall}')
        print(f'Final F1: {f1}')
        print('###############################')
        print('DONE')
        
        wandb.finish()

if __name__ == "__main__":
    main()