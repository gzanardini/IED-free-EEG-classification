from tkinter import W
import wandb.plot
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import secrets
from scipy.stats import mannwhitneyu

#wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

N_RUNS=5

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

def mannwhitney_test(x, y):
    stat, p = mannwhitneyu(x, y)
    return stat, p

car_feats=np.load('data/feats/data_concatenated_CAR.npy')
cz_feats=np.load('data/feats/data_concatenated_Cz.npy')
bipolar_feats=np.load('data/feats/data_concatenated_BipolarDB.npy')
laplacian_feats=np.load('data/feats/data_concatenated_Laplacian.npy')

feats_list=[car_feats, cz_feats, bipolar_feats, laplacian_feats]
feats_names=['CAR', 'Cz', 'BipolarDB' , 'Laplacian']

description=pd.read_csv('data/feats/aggregation_df.csv')

labels=description['epilepsy'].to_numpy()
subjects=description['subject'].to_numpy()

unique_subjects=np.unique(description['subject'])

                    # print('Are all values finite?')
                    # feats=handle_complex_numbers(feats)
                    # print(np.all(np.isfinite(feats)))

significance_threhsholds=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

for i,(feats, name) in enumerate(zip(feats_list, feats_names)):

    print(f'Running for {name} features')

    for significance in significance_threhsholds:

        for j in range(N_RUNS):

            wandb.init(project=f'xgboost_stats_concat' , name=f'{name}_p<{significance}_run_{j}', reinit=True)

            seed=secrets.randbelow(5000)

            y_groundtruths=[]
            y_preds=[]
            y_pred_probs=[]

            wandb.config.seed=seed
            wandb.significance=significance

            for i,subject in enumerate(unique_subjects):

                test_idxs=np.where(description['subject']==subject)[0].astype(int)
                train_idxs=np.where(description['subject']!=subject)[0].astype(int)

                print(f'Iteration {i} - Subject left out: {subject}')
                print(f'Number of train samples: {len(train_idxs)}')
                print(f'Number of test samples: {len(test_idxs)}')
             
                x_train=feats[train_idxs]
                x_test=feats[test_idxs]
                y_train=labels[train_idxs].astype(int)
                y_test=labels[test_idxs].astype(int)

                # perform Mann-Whitney U test on training data
                significant_feats=[]
                for k in range(x_train.shape[1]):
                    stat, p = mannwhitney_test(x_train[y_train==0,k], x_train[y_train==1,k])
                    if p<significance:
                        significant_feats.append(k)

                if len(significant_feats)==0:
                    print('No significant features found')
                    continue

                print(f'Number of significant features: {len(significant_feats)}')
                wandb.log({'significant_feats': significant_feats})
                
                x_train=x_train[:,significant_feats]
                x_test=x_test[:,significant_feats]

                ratio=(len(y_train)-sum(y_train))/sum(y_train)                   #len(np.where(y_train==0)[0])/len(np.where(y_train==1)[0])
                model=XGBClassifier(scale_pos_weight=ratio, n_jobs=4, seed=seed, device='cuda:1')

                ###model=XGBClassifier(scale_pos_weight=ratio, n_jobs=4, device='cuda:1', n_estimators=500, seed=seed, max_depth=10, subsample=0.9, gamma=0.1, learning_rate=0.05)

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

            wandb.log({"accuracy": accuracy,"precision": precision,"recall": recall,"f1": f1,"confusion_matrix": c_m,"auc": auc,"bac80": bac80,"bac": bac,"roc_curve": roc_line,"precision_recall_curve": pr_line})

            wandb.finish()