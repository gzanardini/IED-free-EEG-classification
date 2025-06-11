import os
import wandb.plot
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import itertools 
import secrets
import cupy as cp
from cupy.cuda import Device

# Configuration
N_RUNS = 3
N_CUDA = 2
N_JOBS_XGB = 4
DATA_FOLDER = '/space/gzanardini/tuh_epoched/split/'
LOG_FOLDER = '/space/gzanardini/tuh/'
PROJECT_NAME = 'tuh_singleset_noied_final'
WANDB_KEY = '96e9a92e52e807ed253b3872afd1de1bafc3640a'

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10]
feature_names = ['cc', 'cwt', 'dwt', 'gcc', 'gplv', 'plv', 'mst', 'sst', 'spectral', 'utm']
combiners = ['mean', 'median', 'std', 'skew', 'kurt']
subject_to_skip = ['aaaaajgj', 'aaaaakcd']


def setup_environment():
    """Initialize CUDA and wandb."""
    Device(N_CUDA).use()
    wandb.login(key=WANDB_KEY)

def calculate_bac(labels, scores, sens_thresh):
    """Calculate balanced accuracy with sensitivity threshold."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid_idxs = np.where(tpr >= sens_thresh)[0]
    if len(valid_idxs) == 0:
        # If no TPR >= sens_thresh, use the last threshold by default
        threshold_sensitivity = thresholds[-1] if len(thresholds) > 0 else 0.5
    else:
        threshold_sensitivity = thresholds[valid_idxs[0]]
    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions, labels=[0 ,1] ).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    bac = ((sensitivity + specificity) / 2)
    return bac, fpr, tpr, thresholds

def handle_complex_numbers(features):
    """Handle complex numbers and infinite values in features."""
    if isinstance(features, pd.DataFrame):
        for column in features.columns:
            if np.iscomplexobj(features[column]):
                # Convert to real part or magnitude
                features[column] = features[column].apply(np.abs)  # or .apply(np.real)
            # Replace inf and -inf with NaN
            features[column].replace([np.inf, -np.inf], np.nan, inplace=True)
    elif isinstance(features, np.ndarray):
        if np.iscomplexobj(features):
            # Convert to real part or magnitude
            features = np.abs(features)  # or np.real(features)
        # Replace inf and -inf with NaN
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

def load_feature_data(feature_name, montage, segment_length, combiner):
    """Load and preprocess feature data."""
    features = np.load(f'{DATA_FOLDER}{feature_name}_{montage}_{segment_length}s_{combiner}.npy')
    print(f'Loading {feature_name} features for montage {montage} and segment length {segment_length} with combiner {combiner}')
    print(f'Features shape: {features.shape}')
    
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    features = handle_complex_numbers(features)
    print(f'Features shape after processing: {features.shape}')
    
    return features

def log_metrics(y_tests, y_preds, y_scores, prefix=""):
    """Calculate and log metrics to wandb."""
    bac = balanced_accuracy_score(y_tests, y_preds)
    bac80, fpr, tpr, thresholds = calculate_bac(y_tests, y_scores, 0.8)
    auc = roc_auc_score(y_tests, y_scores)
    recall = recall_score(y_tests, y_preds)
    precision = precision_score(y_tests, y_preds)
    f1 = f1_score(y_tests, y_preds)
    accuracy = accuracy_score(y_tests, y_preds)
    score = auc + bac80  # Calculate combined score
    
    c_m = wandb.plot.confusion_matrix(y_true=y_tests, preds=y_preds, class_names=['healthy', 'epileptic'])
    
    data_roc = [[f, t] for (f, t) in zip(fpr, tpr)]
    table_roc = wandb.Table(data=data_roc, columns=["fpr", "tpr"])
    roc_line = wandb.plot.line(table_roc, "fpr", "tpr", title=f"{prefix}ROC Curve")
    
    p, r, t = precision_recall_curve(y_tests, y_scores)
    data_pr = [[f, t] for (f, t) in zip(p, r)]
    table_pr = wandb.Table(data=data_pr, columns=["precision", "recall"])
    pr_line = wandb.plot.line(table_pr, "precision", "recall", title=f"{prefix}Precision-Recall Curve")
    
    wandb.log({
        f'{prefix}BAC': bac,
        f'{prefix}BAC80': bac80,
        f'{prefix}AUC': auc,
        f'{prefix}Score': score,  # Log the combined score
        f'{prefix}Recall': recall,
        f'{prefix}Precision': precision,
        f'{prefix}F1': f1,
        f'{prefix}Confusion Matrix': c_m,
        f'{prefix}ROC Curve': roc_line,
        f'{prefix}Precision-Recall Curve': pr_line,
        f'{prefix}Accuracy': accuracy
    })
    
    return bac, bac80, auc, score, recall, precision, f1, accuracy

def save_predictions(y_tests, y_preds, y_scores, montage, feature_name, segment_length, combiner, run_n, seed, subjects=False):
    """Save predictions to CSV file."""
    # Create output directory if it doesn't exist
    output_dir = f'{LOG_FOLDER}{PROJECT_NAME}/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame with the predictions
    predictions_df = pd.DataFrame({
        'y_true': y_tests,
        'y_pred': y_preds,
        'y_prob': y_scores
    })
    
    # Save to CSV
    suffix = '_subjects' if subjects else ''
    filename = f'{output_dir}{feature_name}_{montage}_{segment_length}s_{combiner}_run_{run_n}_seed_{seed}{suffix}.csv'
    predictions_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
    
    return predictions_df

def run_loocv(features, labels, subjects, unique_subjects, seed):
    """Train models and evaluate using Leave-One-Subject-Out CV."""
    y_preds = []
    y_scores = []
    y_tests = []
    
    y_preds_subject = []
    y_scores_subject = []
    y_trues_subject = []
    
    for ss, subject in enumerate(unique_subjects):
        print(f'Iteration {ss+1} - Subject: {subject}')
        test_idxs = np.where(subjects == subject)[0]
        
        other_subjects = [subj_oth for subj_oth in unique_subjects if subj_oth != subject]
        train_idxs = np.where(np.isin(subjects, other_subjects))[0]
        
        y_train = labels[train_idxs].astype(int)
        y_test = labels[test_idxs].astype(int)
        
        ratio = (len(y_train) - sum(y_train)) / sum(y_train)
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=7,
            device=f'cuda:{N_CUDA}',
            seed=seed,
            subsample=0.8,
            scale_pos_weight=ratio,
            n_jobs=N_JOBS_XGB,
            gamma=0.1,
            learning_rate=0.05
        )
        
        model.fit(cp.array(features[train_idxs]), cp.array(labels[train_idxs]))
        
        print('Training data shape:', features[train_idxs].shape)
        print('Test data shape:', features[test_idxs].shape)
        
        y_pred = model.predict(cp.array(features[test_idxs]))
        y_score = model.predict_proba(cp.array(features[test_idxs]))[:, 1]
        
        y_score_subject = np.mean(y_score)
        y_pred_subject = np.where(y_score_subject >= 0.5, 1, 0)
        y_true_subject = int(labels[subjects == subject][0])
        
        print('###################################')
        print(f'Final predictions for {subject}: {y_pred}')
        print(f'Final probabilities for {subject}: {y_score}')
        print(f'Ground truths for {subject}: {y_test}')
        print('###################################')
        print(f'SUBJECT AGGREGATED PREDICTIONS')
        print(f'Final predictions for {subject}: {y_pred_subject}')
        print(f'Final probabilities for {subject}: {y_score_subject}')
        print(f'Ground truths for {subject}: {y_true_subject}')
        print('###################################')
        
        y_preds.extend(y_pred)
        y_scores.extend(y_score)
        y_tests.extend(y_test)
        
        y_preds_subject.append(y_pred_subject)
        y_scores_subject.append(y_score_subject)
        y_trues_subject.append(y_true_subject)
    
    return (np.array(y_preds).flatten(), np.array(y_scores).flatten(), np.array(y_tests).flatten(),
            np.array(y_preds_subject).flatten(), np.array(y_scores_subject).flatten(), np.array(y_trues_subject).flatten())

def main():
    """Main execution function."""
    setup_environment()
    description, labels, subjects, unique_subjects, subject_labels = load_data()

    for subj in subject_to_skip:
        idx = np.where(unique_subjects == subj)[0]
        if len(idx) > 0:
            unique_subjects = np.delete(unique_subjects, idx)
            print(f'Skipping subject {subj} --- CONTAINS IEDs')
    
    for montage, feature_name, segment_length, combiner in itertools.product(montages, feature_names, segment_lengths, combiners):
        features = load_feature_data(feature_name, montage, segment_length, combiner)
        
        for run_n in range(N_RUNS):
            print(f'Run {run_n} - {montage} - {feature_name} - {segment_length}s - {combiner}')
            
            wandb.init(
                project=PROJECT_NAME,
                name=f'{feature_name}_{montage}_{segment_length}s_{combiner}_run_{run_n}',
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
                'combiner': combiner,  # Add combiner to config
                'epochs': True
            })
            
            # Run LOOCV
            y_preds, y_scores, y_tests, y_preds_subject, y_scores_subject, y_trues_subject = run_loocv(
                features, labels, subjects, unique_subjects, seed
            )
            
            # Log sample-level metrics
            sample_metrics = log_metrics(y_tests, y_preds, y_scores, prefix='Samples/')
            
            # Log subject-level metrics
            subject_metrics = log_metrics(y_trues_subject, y_preds_subject, y_scores_subject, prefix='')
            
            # Save predictions to CSV
            save_predictions(y_tests, y_preds, y_scores, montage, feature_name, segment_length, combiner, run_n, seed)
            save_predictions(y_trues_subject, y_preds_subject, y_scores_subject, montage, feature_name, segment_length, combiner, run_n, seed, subjects=True)
            
            # Print summary
            print('###############################')
            print(f'Sample BAC: {sample_metrics[0]:.4f}')
            print(f'Sample BAC80: {sample_metrics[1]:.4f}')
            print(f'Sample AUC: {sample_metrics[2]:.4f}')
            print(f'Sample Score (AUC+BAC80): {sample_metrics[3]:.4f}')
            print(f'Sample Recall: {sample_metrics[4]:.4f}')
            print(f'Sample Precision: {sample_metrics[5]:.4f}')
            print('###############################')
            print(f'Subject BAC: {subject_metrics[0]:.4f}')
            print(f'Subject BAC80: {subject_metrics[1]:.4f}')
            print(f'Subject AUC: {subject_metrics[2]:.4f}')
            print(f'Subject Score (AUC+BAC80): {subject_metrics[3]:.4f}')
            print(f'Subject Recall: {subject_metrics[4]:.4f}')
            print(f'Subject Precision: {subject_metrics[5]:.4f}')
            print('###############################')
            
            wandb.finish()

if __name__ == "__main__":
    main()