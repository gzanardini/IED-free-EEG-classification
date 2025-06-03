from pytest import skip
import wandb.plot
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import itertools 
import secrets
from scipy.stats import mannwhitneyu
import os
from sklearn.model_selection import train_test_split
#warnings.simplefilter(action='ignore', category=FutureWarning)
import cupy as cp
from cupy.cuda import Device

wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

N_RUNS=3
N_CUDA=2
PROJECT_NAME='tuh_LOSO_single_set'
FEAT_FOLDER='/space/gzanardini/tuh_features/'

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
    bac = ((sensitivity + specificity) / 2)
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

montages=['CAR', 'Cz', 'BipolarDB','Laplacian']
segment_lengths=[1, 2, 5,10]
feature_names = ['cc','cwt', 'dwt', 'gcc', 'gplv','plv','mst','sst','spectral','utm']

description=pd.read_csv(f'{FEAT_FOLDER}/description.csv')
labels=description['epilepsy'].to_numpy()
subjects=description['subject'].to_numpy()
unique_subjects=np.unique(description['subject'])

subject_labels = []
for subj in unique_subjects:
    lbl = labels[subjects == subj][0]
    subject_labels.append([subj, lbl])
subject_labels = np.array(subject_labels)

for montage, feature_name, segment_length in itertools.product(montages, feature_names, segment_lengths):

    features=np.load(f'{FEAT_FOLDER}{feature_name}_features_{montage}_s{segment_length}.npy')
    print(f'Loading {feature_name} features for montage {montage} and segment length {segment_length}')
    
    print(f'Features shape: {features.shape}')

    if len(features.shape)>2:
        features=features.reshape(features.shape[0], -1)
    features=handle_complex_numbers(features)

    print(f'Features shape: {features.shape}')

    for run_n in range(N_RUNS):   
        print(f'Run {run_n} - {montage} - {feature_name} - {segment_length}s ')
    
        wandb.init(project=PROJECT_NAME, name=f'{feature_name}_{montage}_{segment_length}s_run_{run_n}', reinit=True)

        seed=secrets.randbelow(5000)
        np.random.seed(seed)
        cp.random.seed(seed)
        wandb.config.seed=seed
        wandb.config.montage=montage
        wandb.config.feature_name=feature_name
        wandb.config.segment_length=segment_length
        wandb.config.subject_metrics=True
        
        y_preds=[]
        y_scores=[]
        y_tests=[]

        y_preds_subject=[]
        y_scores_subject=[]
        y_trues_subject=[]

        ctr=0

        for ss, subject in enumerate(unique_subjects):
            print(f'Iteration {ss+1} - Subject: {subject}')
            test_idxs = np.where(description['subject'] == subject)

            other_subjects = [subj_oth for subj_oth in unique_subjects if subj_oth != subject]
            other_subjects_labels = np.array([[subj_oth, labels[subjects == subj_oth][0]] for subj_oth in other_subjects])
            train_subjects=np.array(other_subjects)

            train_idxs = np.where(np.isin(description['subject'], train_subjects))[0]
            
            y_train=labels[train_idxs].astype(int)
            y_test=labels[test_idxs].astype(int)

            ratio=(len(y_train)-sum(y_train))/sum(y_train) 

            model=XGBClassifier(n_estimators=100, max_depth=7, device=f'cuda:{N_CUDA}', seed=seed, subsample=0.8, scale_pos_weight=ratio, n_jobs=4, gamma=0.1, learning_rate=0.05)
            model.fit(cp.array(features[train_idxs]), cp.array(labels[train_idxs]))

            print('Training data shape:', features[train_idxs].shape)
            print('Test data shape:', features[test_idxs].shape)

            y_pred=model.predict(cp.array(features[test_idxs]))
            y_score=model.predict_proba(cp.array(features[test_idxs]))[:,1]

            y_score_subject=np.mean(y_pred)
            y_pred_subject=np.where(y_score_subject >= 0.5, 1, 0)
            y_true_subject=int(labels[subjects == subject][0])


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



        y_preds=np.array(y_preds).flatten()
        y_scores=np.array(y_scores).flatten()
        y_tests=np.array(y_tests).flatten()

             
        bac=balanced_accuracy_score(y_tests, y_preds)
        bac80, fpr, tpr, thresholds = calculate_bac(y_tests, y_scores, 0.8)
        auc=roc_auc_score(y_tests, y_scores)
        recall=recall_score(y_tests, y_preds)
        precision=precision_score(y_tests, y_preds)
        f1=f1_score(y_tests, y_preds)
        accuracy=accuracy_score(y_tests, y_preds)

        c_m = wandb.plot.confusion_matrix(y_true=y_tests, preds=y_preds, class_names=['healthy', 'epileptic'])

        data_roc = [[f, t] for (f, t) in zip(fpr, tpr)]
        table_roc = wandb.Table(data=data_roc, columns=["fpr", "tpr"])
        roc_line=wandb.plot.line(table_roc, "fpr", "tpr", title="ROC Curve")

        p , r , t = precision_recall_curve(y_tests, y_scores)
        data_pr = [[f, t] for (f, t) in zip(p, r)]
        table_pr = wandb.Table(data=data_pr, columns=["precision", "recall"])
        pr_line=wandb.plot.line(table_pr, "precision", "recall", title="Precision-Recall Curve")

        wandb.log({'BAC': bac,
                     'BAC80': bac80,
                     'AUC': auc,
                     'Recall': recall,
                     'Precision': precision,
                     'F1': f1,
                     'Confusion Matrix': c_m,
                     'ROC Curve': roc_line,
                     'Precision-Recall Curve': pr_line,
                     'Accuracy': accuracy})
        

        # Subject-level metrics
        y_preds_subject = np.array(y_preds_subject).flatten()
        y_scores_subject = np.array(y_scores_subject).flatten()
        y_trues_subject = np.array(y_trues_subject).flatten()
        bac_subject = balanced_accuracy_score(y_trues_subject, y_preds_subject)
        bac80_subject, fpr_subject, tpr_subject, thresholds_subject = calculate_bac(y_trues_subject, y_scores_subject, 0.8)
        auc_subject = roc_auc_score(y_trues_subject, y_scores_subject)
        recall_subject = recall_score(y_trues_subject, y_preds_subject)
        precision_subject = precision_score(y_trues_subject, y_preds_subject)
        f1_subject = f1_score(y_trues_subject, y_preds_subject)
        accuracy_subject = accuracy_score(y_trues_subject, y_preds_subject)

        c_m_subject = wandb.plot.confusion_matrix(y_true=y_trues_subject, preds=y_preds_subject, class_names=['healthy', 'epileptic'])
        data_roc_subject = [[f, t] for (f, t) in zip(fpr_subject, tpr_subject)]
        table_roc_subject = wandb.Table(data=data_roc_subject, columns=["fpr", "tpr"])
        roc_line_subject = wandb.plot.line(table_roc_subject, "fpr", "tpr", title="ROC Curve Subject-Level")
        p_subject, r_subject, t_subject = precision_recall_curve(y_trues_subject, y_scores_subject)
        data_pr_subject = [[f, t] for (f, t) in zip(p_subject, r_subject)]
        table_pr_subject = wandb.Table(data=data_pr_subject, columns=["precision", "recall"])
        pr_line_subject = wandb.plot.line(table_pr_subject, "precision", "recall", title="Precision-Recall Curve Subject-Level")

        wandb.log({'Subjects/BAC': bac_subject,
                        'Subjects/BAC80': bac80_subject,
                        'Subjects/AUC': auc_subject,
                        'Subjects/Recall': recall_subject,
                        'Subjects/Precision': precision_subject,
                        'Subjects/F1': f1_subject,
                        'Subjects/Confusion Matrix': c_m_subject,
                        'Subjects/ROC Curve': roc_line_subject,
                        'Subjects/Precision-Recall Curve': pr_line_subject,
                        'Subjects/Accuracy': accuracy_subject})

        wandb.finish() 