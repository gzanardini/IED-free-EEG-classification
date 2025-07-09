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

np.set_printoptions(linewidth=200, precision=4)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
N_RUNS = 5
N_CUDA = 0
SPLIT_RATIO = 0.3
PROJECT_NAME = 'emc_ensemble_bucket_real'
WANDB_KEY = '96e9a92e52e807ed253b3872afd1de1bafc3640a'
DATA_FOLDER = '/space/gzanardini/emc_whole/split'
LOG_FOLDER = '/space/gzanardini/emc/'
N_JOBS_XGB = 1  # Set to 1 for compatibility with CUDA
NUM_WORKERS = 10  # Number of parallel workers for training
N_PARALLEL_FEATURES = NUM_WORKERS  # Parallel feature training within combination
SCIPY_ARRAY_API=1  # Enable SciPy array API for compatibility with cupy

montages = ['CAR', 'Cz', 'BipolarDB', 'Laplacian']
segment_lengths = [1, 2, 5, 10, 20, 60]
feature_names = ['cc', 'cwt', 'dwt', 'plv', 'mst', 'sst', 'spectral', 'utm', 'gcc', 'gplv']
combiners = ['mean', 'median', 'std', 'skew', 'kurt']

best_parameters = {
    'spectral': ('Cz',          10 ,    'std'),
    'cwt':      ('BipolarDB',   2,      'median'),
    'dwt':      ('Laplacian',   10,     'median'),
    'mst':      ('BipolarDB',   60,     'median'),
    'sst':      ('CAR',         10,     'median'),
    'cc':       ('CAR',         1,      'std'),
    'plv':      ('Laplacian',   60,      'kurt'),
    'gcc':      ('CAR',         60,      'median'),
    'gplv':     ('Laplacian',   2,      'std'),
    'utm':      ('Laplacian',   20,     'std')
}

def train_simplex_logistic(X, y, max_iter=2500):

    d = X.shape[1]
    init_w = np.full(d, 1.0/d)                # uniform start

    # negative log-likelihood  (bias = 0)
    def nll(w):
        logits = X @ w
        ce = -np.sum(y * np.log(expit(logits)) +
                     (1 - y) * np.log(1 - expit(logits)))
        alpha = 1      # α = 1 is uniform prior; α > 1 discourages zeros
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
    return cp.array(data), montage, segment_length, combiner

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
        
        # Convert to cupy array for GPU processing
        _feature_data_cache[feature_name] = cp.array(data)
        print(f"Loaded {feature_name}: {data.shape}")

def get_cached_feature_data(feature_name):
    """Get preloaded feature data."""
    return _feature_data_cache[feature_name]

def train_feature_model_parallel(args):
    """Wrapper function for parallel feature model training."""
    feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed = args
    
    # Set random seeds for this process
    random.seed(seed)
    np.random.seed(seed)
    
    # Get cached data instead of loading
    data = get_cached_feature_data(feature_name)
    
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = XGBClassifier(
        scale_pos_weight=ratio,
        n_jobs=1,  # Keep this as 1 since we're parallelizing at higher level
        device=f'cuda:{N_CUDA}',
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
        device=f'cuda:{N_CUDA}',
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
    """Train models for a combination of features in parallel and create an ensemble."""
    
    # Use parallel training if combination has multiple features
    #         # Prepare arguments for parallel processing
    args_list = [
        (feature_name, train_idxs, val_idxs, test_idxs, y_train, y_val, seed + i)
        for i, feature_name in enumerate(feature_combination)
    ]
    
    # Train feature models in parallel using joblib (thread-based for GPU compatibility)
    feature_results = Parallel(n_jobs=min(N_PARALLEL_FEATURES, len(feature_combination)), backend='threading')(
        delayed(train_feature_model_parallel)(args) for args in args_list
    )
    
    # Sort results to maintain order
    feature_models = sorted(feature_results, key=lambda x: feature_combination.index(x['feature_name']))
    
    # Remove feature_name from results as it's not needed anymore
    for model in feature_models:
        del model['feature_name']
        
    # Stack validation probabilities and train a meta-model
    X_meta_val = np.column_stack([logit(np.clip(model['val_probs'], 0.001, 0.999)) for model in feature_models])
    X_meta_test = np.column_stack([logit(np.clip(model['test_probs'], 0.001, 0.999)) for model in feature_models])

    # --- MinMax scaling on validation logits, apply same scaler to test logits ---
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_meta_val = scaler.fit_transform(X_meta_val)
    X_meta_test = scaler.transform(X_meta_test)

    # Train simplex logistic regression meta-model
    w_simplex = train_simplex_logistic(X_meta_val, y_val)
    meta_model = SimplexLogistic(w_simplex)

    # Generate test predictions
    meta_val_probs = meta_model.predict_proba(X_meta_val)[:, 1]
    meta_test_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_val, meta_val_probs)
    bac = balanced_accuracy_score(y_val, meta_val_probs >= 0.5)
    bac80, _, _, _ = calculate_bac(y_val, meta_val_probs, 0.8)
    
    opt_threshold = find_optimal_threshold(y_val, meta_val_probs)
    meta_test_preds = (meta_test_probs >= opt_threshold).astype(int)
    
    return {
        'feature_models': feature_models,
        'meta_model': meta_model,
        'val_probs': meta_val_probs,
        'test_probs': meta_test_probs,
        'test_preds': meta_test_preds,
        'opt_threshold': opt_threshold,
        'auc': auc,
        'bac': bac,
        'bac80': bac80,
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

            # skip_flag = False
            # for existing_run in wandb.Api().runs(path=PROJECT_NAME):
            #     if existing_run.name == RUN_NAME:
            #         print(f"Run {RUN_NAME} already exists, skipping...")
            #         skip_flag = True
            #         break
            # if skip_flag:
            #     continue

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
                    'validation/opt_threshold': ensemble_result['opt_threshold'],
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