import wandb.plot
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import itertools 
import secrets
from scipy.stats import mannwhitneyu

wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

N_RUNS=20

def calculate_bac(labels, scores, sens_thresh):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold_sensitivity = thresholds[np.where(tpr >= sens_thresh)[0][0]]
    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bac = ((sensitivity + specificity) / 2) * 100
    return bac, fpr, tpr, thresholds

def handle_complex_numbers(features):
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

def mannwhitneyu_test(x, y):
    stat, p = mannwhitneyu(x, y)
    return stat, p

# Load data

description=pd.read_csv('data/photostim_steps.csv')
labels=description['epilepsy'].to_numpy()
subjects=description['subject'].to_numpy()
unique_subjects=np.unique(description['subject'])

cwt_features = np.load('data/feats/cwt_feats.npy')
dwt_features = np.load('data/feats/dwt_feats.npy')
gplv_features = np.load('data/feats/gplv_feats.npy')
mst_features = np.load('data/feats/mst_feats.npy')
spectral_features = np.load('data/feats/spectral_feats.npy')
sst_feats = np.load('data/feats/sst_feats.npy')
utm_feats = np.load('data/feats/utm_feats.npy')
plv_features = np.load('data/feats/plv_feats.npy')

features = [
    cwt_features, dwt_features, gplv_features,
    mst_features, spectral_features, sst_feats, utm_feats, plv_features
]

feature_names = [
    'cwt_features', 'dwt_features', 'gplv_features',
    'mst_features', 'spectral_features', 'sst_feats', 'utm_feats', 'plv_features'
]

# Generate all combinations
all_combinations = []
all_combination_names = []

for r in range(1, len(features) + 1):  # r is the length of combinations
    combinations = itertools.combinations(enumerate(features), r)  # Use enumerate for tracking indices
    for combo in combinations:
        indices, combo_features = zip(*combo)  # Separate indices and feature arrays
        combo_names = [feature_names[i] for i in indices]  # Map indices to names
        combined_array = np.concatenate(combo_features, axis=1)  # Concatenate along axis 1 (adjust as needed)
        
        # Track both the array and the names
        all_combinations.append(combined_array)
        all_combination_names.append(combo_names)

        #print(f"Combination: {combo_names}, Shape: {combined_array.shape}")
        # merge the combo_names to a single string
        combo_name = '_'.join(combo_names)

for i,(combo_name, combination) in enumerate(zip(all_combination_names, all_combinations)):
    print(f'Running combination {combo_name}')
    print('################################################')
    print('################################################')
    print('################################################')

    for i in range(N_RUNS):

        wandb.init(project='xgboost-combos-v2' , name=f'{combo_name}_run_{i}', reinit=True)

        seed=secrets.randbelow(5000)

        y_groundtruths=[]

        
        y_preds=[]
        y_pred_probs=[]

        wandb.config.seed=seed
        wandb.config.combination=combo_name

        print('Are all values finite?')
        combination=handle_complex_numbers(combination)
        print(np.all(np.isfinite(combination)))

        for i,subject in enumerate(unique_subjects):

            test_idxs=np.where(description['subject']==subject)
            train_idxs=np.where(description['subject']!=subject)

            print(f'Iteration {i} - Subject left out: {subject}')
            print(f'Number of train samples: {len(train_idxs[0])}')
            print(f'Number of test samples: {len(test_idxs[0])}')

            x_train=combination[train_idxs]
            x_test=combination[test_idxs]
            y_train=labels[train_idxs].astype(int)
            y_test=labels[test_idxs].astype(int)

            ratio=(len(y_train)-sum(y_train))/sum(y_train)                   #len(np.where(y_train==0)[0])/len(np.where(y_train==1)[0])
            model=XGBClassifier(scale_pos_weight=ratio, n_jobs=4, device='cuda:0', n_estimators=100, seed=seed, max_depth=6, subsample=0.9, gamma=0.1, learning_rate=0.01)
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            y_pred_prob = model.predict_proba(x_test)[:,1]

            y_groundtruths.extend(y_test)
            y_preds.extend(y_pred)
            y_pred_probs.extend(y_pred_prob)

        bac = balanced_accuracy_score(y_groundtruths, y_preds)
        bac80, fpr, tpr, thresholds = calculate_bac(y_groundtruths, y_pred_probs, 0.8)

        c_m = wandb.plot.confusion_matrix(y_true=y_groundtruths, preds=y_preds, class_names=['healthy', 'epileptic'])

        fpr, tpr, thresholds = roc_curve(y_groundtruths, y_pred_probs)
        data = [[f, t] for (f, t) in zip(fpr, tpr)]
        table = wandb.Table(data=data, columns=["fpr", "tpr"])

        roc_line=wandb.plot.line(table, "fpr", "tpr", title="ROC Curve")

        auc = roc_auc_score(y_groundtruths, y_pred_probs)

        p_axis, r_axis, thresholds = precision_recall_curve(y_groundtruths, y_pred_probs)
        data = [[r, p] for (r, p) in zip(r_axis, p_axis)]
        table = wandb.Table(data=data, columns=["recall", "precision"])
        pr_line=wandb.plot.line(table, "recall", "precision", title="Precision-Recall Curve")

        accuracy = accuracy_score(y_groundtruths, y_preds)
        precision=precision_score(y_groundtruths, y_preds)
        recall=recall_score(y_groundtruths, y_preds)
        f1=f1_score(y_groundtruths, y_preds)

        wandb.log({
                    "accuracy": accuracy, 
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confusion_matrix": c_m, 
                    "auc": auc,
                    "bac80": bac80,
                    "bac": bac,
                    "roc_curve": roc_line,
                    "precision_recall_curve": pr_line
                })

        wandb.finish()