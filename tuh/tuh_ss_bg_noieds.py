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
N_CUDA = 1
PROJECT_NAME = 'tuh_ss_bg_noieds'
FEAT_FOLDER = '/space/gzanardini/tuh_background/split/'
WANDB_KEY = '96e9a92e52e807ed253b3872afd1de1bafc3640a'
LOG_FOLDER = '/space/gzanardini/tuh'

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10, 20, 60, 120]
feature_names = ['cc', 'cwt', 'dwt', 'gcc', 'gplv', 'plv', 'mst', 'sst', 'spectral', 'utm']
combiners=['mean', 'median', 'std', 'skew', 'kurt']

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
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions, labels=[0, 1]).ravel()
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

def load_feature_data(feature_name, montage, segment_length, combiner):
    """Load and preprocess feature data."""
    features = np.load(f'{FEAT_FOLDER}{feature_name}_{montage}_{segment_length}s_{combiner}.npy')
    features = handle_complex_numbers(features)
    return features

def train_and_evaluate(features, labels, subjects, unique_subjects, description, seed):
    """Train models and evaluate using Leave-One-Subject-Out CV."""
    y_preds = []
    y_scores = []
    y_tests = []



    for ss, subject in enumerate(unique_subjects):
        print(f'Iteration {ss+1} - Subject: {subject}')
        test_idxs = np.where(description['subject'] == subject)

        other_subjects = [subj_oth for subj_oth in unique_subjects if subj_oth != subject]
        train_subjects = np.array(other_subjects)
        
        train_idxs = np.where(np.isin(description['subject'], train_subjects))[0]
        
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
            n_jobs=4,
            gamma=0.1,
            learning_rate=0.05
        )
        
        model.fit(cp.array(features[train_idxs]), cp.array(labels[train_idxs]))

        print('Training data shape:', features[train_idxs].shape)
        print('Test data shape:', features[test_idxs].shape)

        y_pred = model.predict(cp.array(features[test_idxs]))
        y_score = model.predict_proba(cp.array(features[test_idxs]))[:, 1]

        y_preds.extend(y_pred)
        y_scores.extend(y_score)
        y_tests.extend(y_test)

    return np.array(y_preds).flatten(), np.array(y_scores).flatten(), np.array(y_tests).flatten()

def log_metrics(y_tests, y_preds, y_scores):
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
    roc_line = wandb.plot.line(table_roc, "fpr", "tpr", title="ROC Curve")

    p, r, t = precision_recall_curve(y_tests, y_scores)
    data_pr = [[f, t] for (f, t) in zip(p, r)]
    table_pr = wandb.Table(data=data_pr, columns=["precision", "recall"])
    pr_line = wandb.plot.line(table_pr, "precision", "recall", title="Precision-Recall Curve")

    wandb.log({
        'BAC': bac,
        'BAC80': bac80,
        'AUC': auc,
        'Score': score,  # Log the combined score
        'Recall': recall,
        'Precision': precision,
        'F1': f1,
        'Confusion Matrix': c_m,
        'ROC Curve': roc_line,
        'Precision-Recall Curve': pr_line,
        'Accuracy': accuracy
    })
    
    return bac, bac80, auc, score, recall, precision, f1, accuracy

def save_predictions(y_preds, y_scores, y_tests, montage, feature_name, segment_length, combiner, run_n, seed):
    """Save predictions and scores to CSV."""
    df = pd.DataFrame({
        'y_preds': y_preds,
        'y_scores': y_scores,
        'y_tests': y_tests
    })

    output_dir = f'{LOG_FOLDER}/{PROJECT_NAME}/'
    os.makedirs(output_dir, exist_ok=True)

    filename = f'{output_dir}predictions_{montage}_{feature_name}_{segment_length}s_{combiner}_run_{run_n}_seed_{seed}.csv'
    df.to_csv(filename, index=False)
    print(f'Saved predictions to {filename}')

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