from pdb import run
from networkx import dijkstra_predecessor_and_distance
import wandb.plot
from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import itertools 
import secrets
from scipy.stats import mannwhitneyu
import os
from sklearn.model_selection import StratifiedKFold
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import cupy as cp
from cupy.cuda import Device

wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

N_RUNS=10
N_CUDA=2

Device(N_CUDA).use()

def calculate_bac(labels, scores, sens_thresh):
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

montages=['CAR', 'Cz', 'BipolarDB','Laplacian']
segment_lengths=[1, 2, 5,10]
p_values=[1e-3, 5e-4, 1e-4, 1e-5, 1e-6]
feature_names = ['cc', 'cwt', 'dwt', 'gcc', 'gplv','plv','mst','sst','spectral','utm']

description=pd.read_csv('data/photostim_steps.csv')
labels=description['epilepsy'].to_numpy()
subjects=description['subject'].to_numpy()
unique_subjects=np.unique(description['subject'])

subject_labels=[]
for subject in unique_subjects:
    subject_labels.append(labels[subjects==subject][0])
subject_labels=np.array(subject_labels)

ctr=0

run_summary=pd.DataFrame(index=np.arange(5)  ,columns=['subject', 'montage', 'feature_name', 'segment_length', 'significance', 'significant_feats', 'bac', 'bac80', 'auc'])

for run_n in range(N_RUNS):    
    seed=secrets.randbelow(5000)

    y_groundtruths=[]
    y_preds=[]
    y_pred_probs=[]

    wandb.init(project='LOSO_cv_single set', name=f'run_{run_n}', reinit=True)

    for ss, subject in enumerate(unique_subjects):

        test_idxs = np.where(description['subject'] == subject)
        train_idxs = np.where(description['subject'] != subject)

        y_train=labels[train_idxs].astype(int)
        y_test=labels[test_idxs].astype(int)

        best_auc=0
        best_bac=0
        best_bac80=0
        best_fpr=[]
        best_tpr=[]
        best_montage=''
        best_feature_name=''
        best_segment_length=0
        best_significance=0
        best_significant_feats=[]
        best_models=[]

        best_y_preds=[]
        best_y_pred_probs=[]
        best_y_groundtruths=[]

        for montage, feature_name, segment_length, significance in itertools.product(montages, feature_names, segment_lengths, p_values):
            print(f'trying {montage}_{feature_name}_{segment_length}s with p<{significance}')

            feats=np.load(f'featsv3/{feature_name}_features_{montage}_s{segment_length}.npy')
            feats=handle_complex_numbers(feats)

            if len(feats.shape)!=2:
                #print(f'Reshaping from {feats.shape} to {feats.shape[0], feats.shape[1]*feats.shape[2]}')
                feats=feats.reshape(feats.shape[0], feats.shape[1]*feats.shape[2])
    
            significant_feats=[]

            '''arr0=feats[train_idxs][y_train==0]
            arr1=feats[train_idxs][y_train==1]
            ps=[mannwhitneyu_test(arr0[:, col], arr1[:, col])[1] for col in range(feats.shape[1])]'''

            ps=[mannwhitneyu_test(feats[train_idxs][y_train==0][:, col], feats[train_idxs][y_train==1][:, col])[1] for col in range(feats.shape[1])]
        
            significant_feats=[col for col, p_val in enumerate(ps) if p_val<significance]

            feats=cp.array(feats)
            #y_train=cp.array(y_train)

            if len(significant_feats)==0:
                print('No significant features found')
                print('Continuing...')
                continue

            significant_feats=np.array(significant_feats)

            ratio=(len(y_train)-sum(y_train))/sum(y_train)
            model=XGBClassifier(scale_pos_weight=ratio, n_jobs=8, device=f'cuda:{N_CUDA}', n_estimators=100, seed=seed, max_depth=6, subsample=0.9, gamma=0.1, learning_rate=0.01)
            model.fit(feats[train_idxs][:, significant_feats], y_train)

            y_pred=model.predict(feats[test_idxs][:, significant_feats])
            y_pred_prob=model.predict_proba(feats[test_idxs][:, significant_feats])

            bac=balanced_accuracy_score(y_test, y_pred)
            bac80, fpr, tpr, thresholds=calculate_bac(y_test, y_pred_prob[:, 1], 0.8)

            if len(np.unique(y_test)) > 1:
                auc=roc_auc_score(y_test, y_pred_prob[:, 1])
            else:
                auc=0.0

            if auc>best_auc:
                best_auc=auc
                best_bac=bac
                best_bac80=bac80
                best_fpr=fpr
                best_tpr=tpr
                best_montage=montage
                best_feature_name=feature_name
                best_segment_length=segment_length
                best_significance=significance
                best_significant_feats=significant_feats
                best_y_preds=y_pred
                best_y_pred_probs=y_pred_prob
                best_y_groundtruths=y_test
                best_model=model
                print(f'## New best AUC: {best_auc} for subject #{ss} using {montage}_{feature_name}_{segment_length}s with p<{significance}')

                run_summary.loc[ss]=subject, montage, feature_name, segment_length, significance, significant_feats, bac, bac80, auc

                #save the model in a folder with the run number
                if not os.path.exists(f'/space/gzanardini/models/LOSO_{run_n}'):
                    os.makedirs(f'models/{run_n}')
                model.save_model(f'models/{run_n}/{ss}.model')

            '''else:
                print(f'AUC: {auc} for fold {ss} using {montage}_{feature_name}_{segment_length}s with p<{significance}')
            '''
        y_groundtruths.extend(best_y_groundtruths)
        y_preds.extend(best_y_preds)
        y_pred_probs.extend(best_y_pred_probs)

    bac=balanced_accuracy_score(y_groundtruths, y_preds)
    bac80, fpr, tpr, thresholds=calculate_bac(y_groundtruths, y_pred_probs, 0.8)

    c_m=wandb.plot.confusion_matrix(y_true=y_groundtruths, preds=y_preds, class_names=['healthy', 'epileptic'])

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
    
    wandb.log({'summary': wandb.Table(dataframe= run_summary.reset_index())})

    wandb.finish()

    run_summary.to_csv(f'/space/gzanardini/models/LOSO_{run_n}/run_summary_{run_n}.csv')
