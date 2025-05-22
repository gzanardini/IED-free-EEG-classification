import os
import wandb.plot
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import itertools 
import secrets
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
import cupy as cp
from cupy.cuda import Device
import random

np.set_printoptions(linewidth=200, precision=4)

#suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set random seed
wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

N_RUNS=1
N_CUDA=1
SPLIT_RATIO=0.3
RUN_NAME='BAC'
PROJECT_NAME='emc_LOSO_subjects'
FEAT_FOLDER='/space/gzanardini/emc_v2/'

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

def find_optimal_threshold(y_true, y_prob):
    """
    Find the optimal decision threshold based on the maximum geometric mean score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmeans = np.sqrt(tpr * (1 - fpr))  # Geometric mean score
    opt_index = np.argmax(gmeans)  # Find threshold with max geometric mean
    return thresholds[opt_index]

montages=['CAR', 'Cz', 'BipolarDB','Laplacian']
segment_lengths=[1, 2, 5,10]
p_values=[1e-3, 5e-4, 1e-4, 1e-5, 1e-6]
feature_names = ['cc', 'cwt', 'dwt', 'gcc', 'gplv','plv','mst','sst','spectral','utm']

description=pd.read_csv(f'{FEAT_FOLDER}/description.csv')
labels=description['epilepsy'].to_numpy()
subjects=description['subject'].to_numpy()
unique_subjects=np.unique(description['subject'])

subject_labels = []
for subj in unique_subjects:
    lbl = labels[subjects == subj][0]
    subject_labels.append([subj, lbl])
subject_labels = np.array(subject_labels)

for run_n in range(N_RUNS):    
    run_summary=pd.DataFrame(columns=['subject', 'montage', 'feature_name', 'segment_length', 'significance', 'significant_feats', 'bac', 'bac80', 'auc'])
    prediction_summary=pd.DataFrame(columns=['subject', 'y_pred', 'y_prob', 'y_true'])
    subject_summary=pd.DataFrame(columns=['subject', 'y_pred', 'y_prob', 'y_true'])
    
    seed=secrets.randbelow(5000)
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)

    wandb.init(project=PROJECT_NAME , name=f'{RUN_NAME}_run_{run_n}', reinit=True)
    wandb.config.seed=seed

    print(f'RUN {run_n+1} - Seed: {seed}')

    for ss, subject in enumerate(unique_subjects):

        print(f'Iteration {ss+1} - Subject: {subject}')
        test_idxs = np.where(description['subject'] == subject)

        other_subjects = [subj_oth for subj_oth in unique_subjects if subj_oth != subject]
        other_subjects_labels = np.array([[subj_oth, labels[subjects == subj_oth][0]] for subj_oth in other_subjects])

        '''stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        train_subjects, val_subjects = list(stratified_kfold.split(other_subjects, other_subjects_labels[:,1]))[0]'''

        train_subjects, val_subjects = train_test_split(other_subjects, test_size=SPLIT_RATIO, stratify=other_subjects_labels[:,1], random_state=seed)

        other_subjects=np.array(other_subjects)

        train_subjects = other_subjects[np.where(np.isin(other_subjects, train_subjects))[0]]
        val_subjects = other_subjects[np.where(np.isin(other_subjects, val_subjects))[0]]
      
        # for the train indices, find in the description the indices of the training subjects
        train_idxs = np.where(np.isin(description['subject'], train_subjects))[0]
        val_idxs = np.where(np.isin(description['subject'], val_subjects))[0]
        
        y_train=labels[train_idxs].astype(int)
        y_val=labels[val_idxs].astype(int)
        y_test=labels[test_idxs].astype(int)
        
        best_classifiers = {}
        
        for feature_name in feature_names:
            best_auc = 0
            best_bac80 = 0
            best_hm = 0
            best_model = None
            best_data = None
            best_val_data = None
            
            for montage, segment_length, significance in itertools.product(montages, segment_lengths, p_values):
                print(f'Feature: {feature_name}, Montage: {montage}, Segment Length: {segment_length}, Significance: {significance}')
                data=np.load(f'{FEAT_FOLDER}{feature_name}_{montage}_{segment_length}s.npy')
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
                
                if auc > best_auc:
                    print(f'New best BAC for {feature_name}: {bac80}')
                    best_auc = auc
                    best_model = model
                    best_val_data = data[val_idxs][:, significant_feats]
                    test_data=data[test_idxs][:, significant_feats]
                    best_bac80 = bac80
                    best_classifiers[feature_name] = (best_auc, best_model, best_val_data, significant_feats, test_data, best_bac80)    

                # Add data to run_summary DataFrame

                newline=pd.DataFrame({'subject': subject, 'montage': montage, 'feature_name': feature_name, 'segment_length': segment_length, 'significance': significance, 'significant_feats': len(significant_feats), 'bac': None, 'bac80': None, 'auc': auc}, index=[0])           
                run_summary = pd.concat([run_summary, newline], ignore_index=True)

        # for each sample in the validation, make predictions using the best classifiers, and use the prediction as training data for the logistic regression model
        x_val_preds = []
        x_test_preds = []
        
        for feature_name in best_classifiers:
            model = best_classifiers[feature_name][1]
            val_data = best_classifiers[feature_name][2]
            lab_train = model.predict_proba(val_data)[:,1]
            x_val_preds.append(lab_train)
            test_data = best_classifiers[feature_name][4]
            lab_test = model.predict_proba(test_data)[:,1]
            x_test_preds.append(lab_test)
            wandb.log({f'aucs/{feature_name}': best_classifiers[feature_name][0], f'bac80/{feature_name}': best_classifiers[feature_name][5]}, step=ss)

        x_val_preds = np.array(x_val_preds).T
        x_test_preds = np.array(x_test_preds).T

        print(x_val_preds.shape)
        print(x_test_preds.shape)

        #scale each column  by (x-min)/(max-min) 
        scaler = MinMaxScaler(clip=True)
        x_val_preds = scaler.fit_transform(x_val_preds)
        x_test_preds = scaler.transform(x_test_preds)
        
        opt_T=[find_optimal_threshold(y_val, x_val_preds[:, col]) for col in range(x_val_preds.shape[1])]
        print(f'Optimal thresholds: {opt_T}')      
        
        #threshold each column with the corresponding threshold
        y_test_preds = np.where(np.array([x_test_preds[:,i] > opt_T[i] for i in range(x_test_preds.shape[1])]).T,1,0)

        #threshold the probabilities in each column of x_test_preds using the corresponding optimal threshold
        print(f'Intermediate predictions for fold {ss}: {y_test_preds}')

        y_test_prob = np.mean(y_test_preds, axis=1)
        y_test_preds = np.where(y_test_prob > 0.5, 1, 0)

        print(f'Final predictions for {subject}: {y_test_preds}')
        print(f'Final probabilities for {subject}: {y_test_prob}')
        print(f'Ground truths for {subject}: {y_test}')

        prediction_summary = pd.concat([prediction_summary, pd.DataFrame({'subject': subject, 'y_pred': y_test_preds, 'y_prob': y_test_prob, 'y_true': y_test})], ignore_index=True)

        # compute some subject_based metrics
        subj_label=int(labels[subjects == subject][0])
        y_test_prob_subj = np.mean(y_test_preds, axis=0)
        y_test_preds_subj = np.where(y_test_prob_subj > 0.5, 1, 0)

        print('###################################')
        print(f'SUBJECT AGGREGATED PREDICTIONS')
        print(f'Final predictions for {subject}: {y_test_preds_subj}')
        print(f'Final probabilities for {subject}: {y_test_prob_subj}')
        print(f'Ground truths for {subject}: {subj_label}')

        subject_summary = pd.concat([subject_summary, pd.DataFrame({'subject': [subject], 'y_pred': [y_test_preds_subj], 'y_prob': [y_test_prob_subj], 'y_true': [subj_label]})], ignore_index=True)

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

    print('###############################')
    print(f'Final BAC: {final_bac}')
    print(f'Final AUC: {final_auc}')
    print(f'Final Accuracy: {final_accuracy}')
    print(f'Final BAC80: {final_bac80}')
    print(f'Final Precision: {final_precision}')
    print(f'Final Recall: {final_recall}')
    print(f'Final F1: {final_f1}')
    print('###############################')

    subj_y_preds = np.array(subject_summary['y_pred']).astype(int)
    subj_y_true = np.array(subject_summary['y_true']).astype(int)
    subj_y_probs = np.array(subject_summary['y_prob']).astype(float)
    subj_final_bac = balanced_accuracy_score(subj_y_true, subj_y_preds)
    subj_final_auc = roc_auc_score(subj_y_true, subj_y_probs)
    subj_final_accuracy = accuracy_score(subj_y_true, subj_y_preds)
    subj_final_bac80, fpr, tpr, thresholds = calculate_bac(subj_y_true, subj_y_probs, 0.8)
    subj_final_precision = precision_score(subj_y_true, subj_y_preds)
    subj_final_recall = recall_score(subj_y_true, subj_y_preds)
    subj_final_f1 = f1_score(subj_y_true, subj_y_preds)

    subj_cm= wandb.plot.confusion_matrix(y_true=subj_y_true, preds=subj_y_preds, class_names=['healthy', 'epileptic'])
    subj_data_roc = [[f, t] for (f, t) in zip(fpr, tpr)]
    subj_table_roc = wandb.Table(data=subj_data_roc, columns=["fpr", "tpr"])
    subj_roc_line=wandb.plot.line(subj_table_roc, "fpr", "tpr", title="ROC Curve")
    subj_p , subj_r , t = precision_recall_curve(subj_y_true, subj_y_probs)
    subj_data_pr = [[f, t] for (f, t) in zip(subj_p, subj_r)]
    subj_table_pr = wandb.Table(data=subj_data_pr, columns=["precision", "recall"])
    subj_pr_line=wandb.plot.line(subj_table_pr, "precision", "recall", title="Precision-Recall Curve")
    wandb.log({'Subject Metrics/BAC': subj_final_bac,
                'Subject Metrics/AUC': subj_final_auc,
                'Subject Metrics/Accuracy': subj_final_accuracy, 
                'Subject Metrics/BAC80': subj_final_bac80, 
                'Subject Metrics/Precision': subj_final_precision, 
                'Subject Metrics/Recall': subj_final_recall,  
                'Subject Metrics/F1': subj_final_f1, 
                'Subject Metrics/Confusion Matrix': subj_cm, 
                'Subject Metrics/ROC Curve': subj_roc_line, 
                'Subject Metrics/Precision-Recall Curve': subj_pr_line})


    LOG_FOLDER='emc'

    if not os.path.exists(f'/space/gzanardini/{LOG_FOLDER}/'):
        os.mkdir('/space/gzanardini/{LOG_FOLDER}/')

    if not os.path.exists(f'/space/gzanardini/{LOG_FOLDER}/{PROJECT_NAME}/'):
        os.mkdir(f'/space/gzanardini/{LOG_FOLDER}/{PROJECT_NAME}/')

    run_summary.to_csv(f'/space/gzanardini/{LOG_FOLDER}/{PROJECT_NAME}/{RUN_NAME}_{run_n}_seed_{seed}.csv', index=False)    
    prediction_summary.to_csv(f'/space/gzanardini/{LOG_FOLDER}/{PROJECT_NAME}/{RUN_NAME}_{run_n}_predictions_seed_{seed}.csv', index=False)
    subject_summary.to_csv(f'/space/gzanardini/{LOG_FOLDER}/{PROJECT_NAME}/{RUN_NAME}_{run_n}_subject_predictions_seed_{seed}.csv', index=False)

    wandb.finish()