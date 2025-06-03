import os
import wandb.plot
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import itertools 
import secrets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
RUN_NAME = 'epochs'
PROJECT_NAME = 'emc_LOSO_nomwu'
FEAT_FOLDER = '/space/gzanardini/emc_v2/'
WANDB_KEY = '96e9a92e52e807ed253b3872afd1de1bafc3640a'
NUM_WORKERS=8
N_JOBS_XGB = 1  # Set to 1 for compatibility with CUDA

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10]
feature_names = ['cc', 'cwt', 'dwt', 'gcc', 'gplv', 'plv', 'mst', 'sst', 'spectral', 'utm']

def setup_environment():
    """Initialize CUDA and wandb."""
    Device(N_CUDA).use()
    wandb.login(key=WANDB_KEY)

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

def find_optimal_threshold(y_true, y_prob):
    """Find the optimal decision threshold based on the maximum geometric mean score."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmeans = np.sqrt(tpr * (1 - fpr))
    opt_index = np.argmax(gmeans)
    return thresholds[opt_index]

def load_data():
    """Load and prepare the dataset."""
    description = pd.read_csv(f'{FEAT_FOLDER}/description.csv')
    labels = description['epilepsy'].to_numpy()
    subjects = description['subject'].to_numpy()
    unique_subjects = np.unique(description['subject'])
    
    subject_labels = []
    for subj in unique_subjects:
        lbl = labels[subjects == subj][0]
        subject_labels.append([subj, lbl])
    subject_labels = np.array(subject_labels)
    
    return description, labels, subjects, unique_subjects, subject_labels

def load_feature_data(feature_name, montage, segment_length):
    """Load and preprocess feature data."""
    data = np.load(f'{FEAT_FOLDER}{feature_name}_{montage}_{segment_length}s.npy')
    data = handle_complex_numbers(data)
    
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
        
    return cp.array(data)

def get_train_val_test_indices(description, labels, subject, seed):
    """Get indices for train/validation/test splits for LOSO CV."""
    test_idxs = np.where(description['subject'] == subject)
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

def train_single_classifier(feature_name, montage, segment_length, train_idxs, val_idxs, test_idxs, y_train, y_val, seed):
    """Train a single XGBoost classifier.
        Returns:
        - feature_name: Name of the feature used
        - montage: Montage used
        - segment_length: Length of the segments used
        - auc: Area Under the ROC Curve
        - bac80: Balanced Accuracy at 80% sensitivity
        - score: Combined score (AUC + BAC80)
        - model: Trained XGBoost model
        - val_data: Validation data used for predictions
        - test_data: Test data used for predictions
    """
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
    bac80 = calculate_bac(y_val, y_prob, 0.8)[0]
    score = auc + bac80  # Calculate combined score
    
    return feature_name, montage, segment_length, auc, bac80, score, model, data[val_idxs], data[test_idxs]

def train_feature_classifiers(train_idxs, val_idxs, test_idxs, y_train, y_val, seed):
    """Train classifiers for all feature/montage/segment combinations."""
    best_classifiers = {feature: None for feature in feature_names}
    run_results = []
    
    for feature_name in feature_names:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS, thread_name_prefix='xgb_multi') as executor:
            futures = []
            for montage, segment_length in itertools.product(montages, segment_lengths):
                futures.append(executor.submit(
                    train_single_classifier, feature_name, montage, segment_length,
                    train_idxs, val_idxs, test_idxs, y_train, y_val, seed
                ))
            
            for future in futures:
                result = future.result()
                if result is not None:
                    feature_name, montage, segment_length, auc, bac80, score, model, val_data, test_data = result
                    
                    if best_classifiers[feature_name] is None or score > best_classifiers[feature_name][5]:
                        print(f'New best score (AUC+BAC80) for {feature_name}: {score:.4f}')
                        best_classifiers[feature_name] = (auc, model, val_data, test_data, bac80, score)
                    
                    run_results.append({
                        'feature_name': feature_name,
                        'montage': montage,
                        'segment_length': segment_length,
                        'auc': auc,
                        'bac80': bac80,
                        'score': score  # Add score to results
                    })
    
    return best_classifiers, run_results

def make_ensemble_predictions(best_classifiers, y_val):
    """Make ensemble predictions using best classifiers."""
    x_val_preds = []
    x_test_preds = []
    
    for feature_name in best_classifiers:
        if best_classifiers[feature_name] is not None:
            model = best_classifiers[feature_name][1]
            val_data = best_classifiers[feature_name][2]
            test_data = best_classifiers[feature_name][3]
            
            val_prob = model.predict_proba(val_data)[:, 1]
            test_prob = model.predict_proba(test_data)[:, 1]
            
            x_val_preds.append(val_prob)
            x_test_preds.append(test_prob)
    
    x_val_preds = np.array(x_val_preds).T
    x_test_preds = np.array(x_test_preds).T
    
    scaler = MinMaxScaler(clip=True)
    x_val_preds = scaler.fit_transform(x_val_preds)
    x_test_preds = scaler.transform(x_test_preds)
    
    opt_thresholds = [find_optimal_threshold(y_val, x_val_preds[:, col]) 
                     for col in range(x_val_preds.shape[1])]
    
    y_test_preds = np.where(
        np.array([x_test_preds[:, i] > opt_thresholds[i] for i in range(x_test_preds.shape[1])]).T, 
        1, 0
    )
    
    y_test_prob = np.mean(y_test_preds, axis=1)
    y_test_final = np.where(y_test_prob > 0.5, 1, 0)
    
    return y_test_final, y_test_prob

def log_metrics(y_true, y_pred, y_prob, prefix=""):
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
    roc_curve_plot = wandb.plot.line(roc_table, "fpr", "tpr", title=f"{prefix}ROC Curve")
    
    p, r, _ = precision_recall_curve(y_true, y_prob)
    pr_data = [[precision, recall] for precision, recall in zip(p, r)]
    pr_table = wandb.Table(data=pr_data, columns=["precision", "recall"])
    pr_curve_plot = wandb.plot.line(pr_table, "precision", "recall", title=f"{prefix}Precision-Recall Curve")
    
    metrics = {
        f'{prefix}BAC': bac,
        f'{prefix}AUC': auc,
        f'{prefix}Accuracy': accuracy,
        f'{prefix}BAC80': bac80,
        f'{prefix}Precision': precision,
        f'{prefix}Recall': recall,
        f'{prefix}F1': f1,
        f'{prefix}Confusion Matrix': cm,
        f'{prefix}ROC Curve': roc_curve_plot,
        f'{prefix}Precision-Recall Curve': pr_curve_plot
    }
    
    wandb.log(metrics)
    return metrics

def save_results(run_summary, prediction_summary, subject_summary, run_n, seed):
    """Save results to CSV files."""
    log_path = f'/space/gzanardini/emc/{PROJECT_NAME}/'
    os.makedirs(log_path, exist_ok=True)
    
    run_summary.to_csv(f'{log_path}{RUN_NAME}_{run_n}_seed_{seed}.csv', index=False)
    prediction_summary.to_csv(f'{log_path}{RUN_NAME}_{run_n}_predictions_seed_{seed}.csv', index=False)
    subject_summary.to_csv(f'{log_path}{RUN_NAME}_{run_n}_subject_predictions_seed_{seed}.csv', index=False)

def run_loso_cv(description, labels, subjects, unique_subjects, seed):
    """Run Leave-One-Subject-Out cross-validation."""
    run_summary = pd.DataFrame(columns=['subject', 'montage', 'feature_name', 'segment_length', 'bac', 'bac80', 'auc', 'score'])
    prediction_summary = pd.DataFrame(columns=['subject', 'y_pred', 'y_prob', 'y_true'])
    subject_summary = pd.DataFrame(columns=['subject', 'y_pred', 'y_prob', 'y_true'])
    
    for ss, subject in enumerate(unique_subjects):
        print(f'Iteration {ss+1} - Subject: {subject}')
        
        train_idxs, val_idxs, test_idxs = get_train_val_test_indices(description, labels, subject, seed)
        
        y_train = labels[train_idxs].astype(int)
        y_val = labels[val_idxs].astype(int)
        y_test = labels[test_idxs].astype(int)
        
        best_classifiers, run_results = train_feature_classifiers(train_idxs, val_idxs, test_idxs, y_train, y_val, seed)
        
        # Log individual feature performance
        for feature_name in best_classifiers:
            if best_classifiers[feature_name] is not None:
                wandb.log({
                    f'aucs/{feature_name}': best_classifiers[feature_name][0],
                    f'bac80/{feature_name}': best_classifiers[feature_name][4],
                    f'score/{feature_name}': best_classifiers[feature_name][5]  # Log score
                }, step=ss)
        
        # Add results to summary
        for result in run_results:
            newline = pd.DataFrame({**result, 'subject': subject}, index=[0])
            run_summary = pd.concat([run_summary, newline], ignore_index=True)
        
        y_test_preds, y_test_prob = make_ensemble_predictions(best_classifiers, y_val)
        
        print(f'Final predictions for {subject}: {y_test_preds}')
        print(f'Final probabilities for {subject}: {y_test_prob}')
        print(f'Ground truths for {subject}: {y_test}')
        
        pred_df = pd.DataFrame({
            'subject': subject,
            'y_pred': y_test_preds,
            'y_prob': y_test_prob,
            'y_true': y_test
        })
        prediction_summary = pd.concat([prediction_summary, pred_df], ignore_index=True)
        
        # Subject-level aggregation
        subj_label = int(labels[subjects == subject][0])
        y_test_prob_subj = np.mean(y_test_preds)
        y_test_preds_subj = int(y_test_prob_subj > 0.5)
        
        subj_df = pd.DataFrame({
            'subject': [subject],
            'y_pred': [y_test_preds_subj],
            'y_prob': [y_test_prob_subj],
            'y_true': [subj_label]
        })
        subject_summary = pd.concat([subject_summary, subj_df], ignore_index=True)
    
    return run_summary, prediction_summary, subject_summary

def main():
    """Main execution function."""
    setup_environment()
    description, labels, subjects, unique_subjects, subject_labels = load_data()
    
    for run_n in range(N_RUNS):
        seed = secrets.randbelow(5000)
        random.seed(seed)
        np.random.seed(seed)
        cp.random.seed(seed)
        
        wandb.init(project=PROJECT_NAME, name=f'{RUN_NAME}_run_{run_n}', reinit=True)
        wandb.config.seed = seed
        
        print(f'RUN {run_n+1} - Seed: {seed}')
        
        run_summary, prediction_summary, subject_summary = run_loso_cv(
            description, labels, subjects, unique_subjects, seed
        )
        
        # Evaluate results
        y_preds = np.array(prediction_summary['y_pred']).astype(int)
        y_true = np.array(prediction_summary['y_true']).astype(int)
        y_probs = np.array(prediction_summary['y_prob']).astype(float)
        
        log_metrics(y_true, y_preds, y_probs, prefix="Overall Metrics/")
        
        # Subject-level evaluation
        subj_y_preds = np.array(subject_summary['y_pred']).astype(int)
        subj_y_true = np.array(subject_summary['y_true']).astype(int)
        subj_y_probs = np.array(subject_summary['y_prob']).astype(float)
        
        metrics = log_metrics(subj_y_true, subj_y_preds, subj_y_probs, prefix="")
        print('#############################')
        print(f'Final BAC: {metrics["BAC"]:.4f}')
        print(f'Final AUC: {metrics["AUC"]:.4f}')
        print(f'Final Accuracy: {metrics["Accuracy"]:.4f}')
        print(f'Final BAC80: {metrics["BAC80"]:.4f}')
        print(f'Final Precision: {metrics["Precision"]:.4f}')
        print(f'Final Recall: {metrics["Recall"]:.4f}')
        print(f'Final F1: {metrics["F1"]:.4f}')
        print('#############################')

        save_results(run_summary, prediction_summary, subject_summary, run_n, seed)

        wandb.finish()

if __name__ == "__main__":
    main()