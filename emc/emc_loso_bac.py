import os
import wandb.plot
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import itertools 
import secrets
from scipy.stats import mannwhitneyu
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import warnings
import cupy as cp
from cupy.cuda import Device
import random

np.set_printoptions(linewidth=200, precision=4)

#suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set random seed
wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

N_RUNS=10
N_CUDA=2
SPLIT_RATIO=0.3

Device(N_CUDA).use()

def calculate_bac(labels, scores, sens_thresh):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold_sensitivity = thresholds[np.where(tpr >= sens_thresh)[0][0]]
    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bac = ((sensitivity + specificity) / 2)
    return bac, fpr, tpr, thresholds

def handle_complex_numbers(features):
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

def mannwhitneyu_test(x, y):
    stat, p = mannwhitneyu(x, y)
    return stat, p

montages=['CAR', 'Cz', 'BipolarDB','Laplacian']
segment_lengths=[1, 2, 5,10]
p_values=[1e-3, 5e-4, 1e-4, 1e-5, 1e-6]
feature_names = ['cc', 'cwt', 'dwt', 'gcc', 'gplv','plv','mst','sst','spectral','utm']

labels=np.load('/space/gzanardini/emc_dataset/labels.npy')

for run_n in range(N_RUNS):    
    run_summary=pd.DataFrame(columns=['montage', 'feature_name', 'segment_length', 'significance', 'significant_feats', 'bac', 'bac80', 'auc'])
    prediction_summary=pd.DataFrame(columns=['subject', 'y_pred', 'y_prob', 'y_true'])
    
    seed=secrets.randbelow(5000)
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)

    wandb.init(project='emc_nested_LOSO' , name=f'BACweighted_run_{run_n}', reinit=True)
    wandb.config.seed=seed

    print(f'RUN {run_n+1} - Seed: {seed}')

    for ss in range(len(labels)):
        auc_recap = {f'{feature_name}_auc':[] for feature_name in feature_names}
        bac80_recap = {f'{feature_name}_bac80':[] for feature_name in feature_names}

        test_idx=[ss]
        other_idxs=np.delete(np.arange(len(labels)), test_idx)

        train_idxs, val_idxs = train_test_split(other_idxs, test_size=SPLIT_RATIO, stratify=labels[other_idxs], random_state=seed)

        y_train=labels[train_idxs]
        y_val=labels[val_idxs]
        y_test=labels[test_idx]

        best_classifiers = {}
        
        for feature_name in feature_names:
            best_auc = 0
            best_bac80 = 0
            best_model = None
            best_data = None
            best_val_data = None
            
            for montage, segment_length, significance in itertools.product(montages, segment_lengths, p_values):
                
                print(f'Feature: {feature_name}, Montage: {montage}, Segment Length: {segment_length}, Significance: {significance}')
                data=np.load(f'/space/gzanardini/emc_dataset/{feature_name}_{montage}_{segment_length}s.npy')
                data=handle_complex_numbers(data)

                if len(data.shape) >2:
                    data = data.reshape(data.shape[0], -1)
                
                ps=np.array([mannwhitneyu_test(data[train_idxs][y_train==0][:, col], data[train_idxs][y_train==1][:, col])[1] for col in range(data.shape[1])]).squeeze()
                significant_feats = np.where(ps < significance)[0]
                
                if len(significant_feats) == 0:
                    print('No significant features found')
                    continue
                
                data=cp.array(data)
                ratio=(len(y_train)-sum(y_train))/sum(y_train)
                model=XGBClassifier(scale_pos_weight=ratio, n_jobs=8, device=f'cuda:{N_CUDA}', n_estimators=100, seed=seed, max_depth=6, subsample=0.9, gamma=0.1, learning_rate=0.01)
                model.fit(data[train_idxs][:, significant_feats], y_train)
                
                y_prob = model.predict_proba(data[val_idxs][:, significant_feats])[:, 1]
                auc = roc_auc_score(y_val, y_prob)
                bac80= calculate_bac(y_val, y_prob, 0.8)[0]
                
                if bac80 > best_bac80:
                    print(f'New best BAC80 for {feature_name}: {bac80}')
                    best_auc = auc
                    best_model = model
                    best_val_data = data[val_idxs][:, significant_feats]
                    test_data=data[test_idx][:, significant_feats]
                    best_bac80 = bac80
                    best_classifiers[feature_name] = (best_auc, best_model, best_val_data, significant_feats, test_data, best_bac80)    
                         
                # Add data to run_summary DataFrame
                newline=pd.DataFrame({'subject': ss, 'montage': montage, 'feature_name': feature_name, 'segment_length': segment_length, 'significance': significance, 'significant_feats': len(significant_feats), 'bac': None, 'bac80': None, 'auc': auc}, index=[0])           
                run_summary = pd.concat([run_summary, newline], ignore_index=True)

        # for each sample in the validation, make predictions using the best classifiers, and use the prediction as training data for the logistic regression model
        X_train_lr = []
        X_test_lr = []

        for feature_name in best_classifiers:
            model = best_classifiers[feature_name][1]
            val_data = best_classifiers[feature_name][2]
            lab_train = model.predict_proba(val_data)[:,1]
            X_train_lr.append(lab_train)
            test_data = best_classifiers[feature_name][4]
            lab_test = model.predict_proba(test_data)[:,1]
            X_test_lr.append(lab_test)
            wandb.log({f'aucs/{feature_name}': best_classifiers[feature_name][0], f'bac80/{feature_name}': best_classifiers[feature_name][5]}, step=ss)

        X_train_lr = np.array(X_train_lr).T
        X_test_lr = np.array(X_test_lr).T
        
        # use the bac80 of each model as a weight for the ensmbled prections
        weights = [best_classifiers[feature_name][5] for feature_name in best_classifiers]
        weights = np.array(weights)-0.5
        weights = weights/np.sum(weights)
        print(f'Weights: {weights}')

        weights_table=[[f, w] for (f, w) in zip(best_classifiers.keys(), weights)]
        weights_table=wandb.Table(data=weights_table, columns=['feature', 'weight'])
        wandb.log({'weights': weights_table}, step=ss)

        y_test_prob = np.dot(X_test_lr, weights)
        y_test_pred = (y_test_prob >= 0.5).astype(int)

        print(f'Final predictions for fold {ss}: {y_test_pred}')
        print(f'Final probabilities for fold {ss}: {y_test_prob}')
        print(f'Ground truths for fold {ss}: {y_test}')

        prediction_summary = pd.concat([prediction_summary, pd.DataFrame({'subject': ss, 'y_pred': y_test_pred, 'y_prob': y_test_prob, 'y_true': y_test})], ignore_index=True)
        
    print(prediction_summary['y_pred'])
    print(prediction_summary['y_true'])
    print(prediction_summary['y_prob'])

    y_preds_outer=np.array(prediction_summary['y_pred']).astype(int)
    y_true_outer=np.array(prediction_summary['y_true']).astype(int)
    y_probs_outer=np.array(prediction_summary['y_prob']).astype(float)

    final_bac = balanced_accuracy_score(y_true_outer, y_preds_outer)
    final_auc = roc_auc_score(y_true_outer, y_probs_outer)
    final_accuracy = accuracy_score(y_true_outer, y_preds_outer)
    final_bac80, fpr, tpr, thresholds = calculate_bac(y_true_outer, y_probs_outer, 0.8)
    final_precision = precision_score(y_true_outer, y_preds_outer)
    final_recall = recall_score(y_true_outer, y_preds_outer)
    final_f1 = f1_score(y_true_outer, y_preds_outer)

    c_m = wandb.plot.confusion_matrix(y_true=y_true_outer, preds=y_preds_outer, class_names=['healthy', 'epileptic'])

    data_roc = [[f, t] for (f, t) in zip(fpr, tpr)]
    table_roc = wandb.Table(data=data_roc, columns=["fpr", "tpr"])
    roc_line=wandb.plot.line(table_roc, "fpr", "tpr", title="ROC Curve")

    p , r , t = precision_recall_curve(y_true_outer, y_probs_outer)
    data_pr = [[f, t] for (f, t) in zip(p, r)]
    table_pr = wandb.Table(data=data_pr, columns=["precision", "recall"])
    pr_line=wandb.plot.line(table_pr, "precision", "recall", title="Precision-Recall Curve")

    wandb.log({'BAC': final_bac,
                'AUC': final_auc,
                'Accuracy': final_accuracy, 
                'BAC80': final_bac80, 
                'Precision': final_precision, 
                'Recall': final_recall,  
                'F1': final_f1, 
                'Confusion Matrix': c_m, 
                'ROC Curve': roc_line, 
                'Precision-Recall Curve': pr_line})

    if not os.path.exists('/space/gzanardini/emc/'):
        os.mkdir('/space/gzanardini/emc/')

    run_summary.to_csv(f'/space/gzanardini/emc/BAC_weighted_run_{run_n}_seed_{seed}.csv', index=False)    
    prediction_summary.to_csv(f'/space/gzanardini/emc/BAC_weighted_run_{run_n}_predictions_seed_{seed}.csv', index=False)

    print('###############################')
    print(f'Final BAC: {final_bac}')
    print(f'Final AUC: {final_auc}')
    print(f'Final Accuracy: {final_accuracy}')
    print(f'Final BAC80: {final_bac80}')
    print(f'Final Precision: {final_precision}')
    print(f'Final Recall: {final_recall}')
    print(f'Final F1: {final_f1}')
    print('###############################')
    print('DONE')

    wandb.finish()