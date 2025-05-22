from sklearn import tree
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
import cupy as cp
from cupy.cuda import Device

wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

N_RUNS=5
N_CUDA=2

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

montages=['CAR', 'Cz', 'BipolarDB','Laplacian']
segment_lengths=[1, 2, 5,10]
p_values=[5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6]

description=pd.read_csv('data/photostim_steps.csv')
labels=description['epilepsy'].to_numpy()
subjects=description['subject'].to_numpy()
unique_subjects=np.unique(description['subject'])

ctr=0

for montage in montages:
    for segment_length in segment_lengths:
            print(f'Loading features for {montage} montage and {segment_length} segment length')        
            spectral_features = np.load(f'featsv3/spectral_features_{montage}_s{segment_length}.npy')
            cwt_features = np.load(f'featsv3/cwt_features_{montage}_s{segment_length}.npy')
            dwt_features = np.load(f'featsv3/dwt_features_{montage}_s{segment_length}.npy')
            mst_features = np.load(f'featsv3/mst_features_{montage}_s{segment_length}.npy')
            sst_feats = np.load(f'featsv3/sst_features_{montage}_s{segment_length}.npy')
            utm_feats = np.load(f'featsv3/utm_features_{montage}_s{segment_length}.npy')
            plv_features = np.load(f'featsv3/plv_features_{montage}_s{segment_length}.npy')
            gplv_features = np.load(f'featsv3/gplv_features_{montage}_s{segment_length}.npy')
            cc_features = np.load(f'featsv3/cc_features_{montage}_s{segment_length}.npy')
            gcc_features = np.load(f'featsv3/gcc_features_{montage}_s{segment_length}.npy')

            features = [ spectral_features, cwt_features, dwt_features, mst_features, sst_feats, utm_feats, plv_features, gplv_features, cc_features, gcc_features]
            feature_names = ['spectral', 'cwt', 'dwt', 'mst', 'sst', 'utm', 'plv', 'gplv', 'cc', 'gcc']

            for i, feature in enumerate(features):
                #print(f'Feature {feature_names[i]} has shape {feature.shape}')
                if len(feature.shape)==3:
                    features[i]=features[i].reshape(features[i].shape[0],-1)
                    #print(f'Reshaped to {features[i].shape}')

            # Generate all combinations

            for r in range(1, len(features) + 1):  # r is the length of combinations
                combinations = itertools.combinations(enumerate(features), r)  # Use enumerate for tracking indices
                for combo in combinations:
                    indices, combo_features = zip(*combo)  # Separate indices and feature arrays
                    combo_names = [feature_names[i] for i in indices]  # Map indices to names
                    combined_array = np.concatenate(combo_features, axis=1)  # Concatenate along axis 1 (adjust as needed)
                    
                    # Track both the array and the names

                    #print(f"Combination: {combo_names}, Shape: {combined_array.shape}")
                    # merge the combo_names to a single string
                    combo_name = '_'.join(combo_names)
                    combo_name=f'{montage}_{segment_length}s_{combo_name}'

                    print('################################################')
                    print(f"{ctr+1} Combination: {combo_name}, Shape: {combined_array.shape}")
                    ctr+=1
                    print('################################################')

                    print('Are all values finite?')
                    feats = handle_complex_numbers(combined_array)
                    print(np.all(np.isfinite(feats)))

                    for significance in p_values:
                        for run_n in range(N_RUNS):
                            wandb.init(project='combos-v3_sign_fast' , name=f'{combo_name}_p{significance}_run_{run_n}', reinit=True)
                            
                            skip_outer = False
                            seed=secrets.randbelow(5000)

                            y_groundtruths=[]
                            y_preds=[]
                            y_pred_probs=[]

                            wandb.config.seed=seed
                            wandb.config.combination=combo_name

                            for i, subject in enumerate(unique_subjects):
                                test_idxs = np.where(description['subject'] == subject)
                                train_idxs = np.where(description['subject'] != subject)

                                print(f'Iteration {i} - Subject left out: {subject}')
                                print(f'Number of train samples: {len(train_idxs[0])}')
                                print(f'Number of test samples: {len(test_idxs[0])}')

                                with Device(N_CUDA):
                                    y_train = labels[train_idxs].astype(int)
                                    y_test = labels[test_idxs].astype(int)

                                significant_feats = []
                                arr0 = feats[train_idxs][y_train == 0]
                                arr1 = feats[train_idxs][y_train == 1]
                                ps = [mannwhitneyu_test(arr0[:, col], arr1[:, col])[1] for col in range(feats.shape[1])]
                                significant_feats = [col for col, p_val in enumerate(ps) if p_val < significance]

                                if len(significant_feats) == 0:
                                    print('No significant features found')
                                    skip_outer = True
                                    break

                                significant_feats = np.array(significant_feats)
                                print(f'Number of significant features: {len(significant_feats)}')
                                wandb.log({'significant_feats': significant_feats})

                                ratio = (len(y_train) - sum(y_train)) / sum(y_train)
                                model = XGBClassifier(scale_pos_weight=ratio, n_jobs=8, device=f'cuda:{N_CUDA}',
                                                      n_estimators=100, seed=seed, max_depth=6,
                                                      subsample=0.9, gamma=0.1, learning_rate=0.01)
                                
                                model.fit(feats[train_idxs][:, significant_feats], y_train)
                            
                                y_pred = model.predict(feats[test_idxs][:, significant_feats])
                                y_pred_prob = model.predict_proba(feats[test_idxs][:, significant_feats])[:, 1]

                                y_groundtruths.extend(y_test)
                                y_preds.extend(y_pred)
                                y_pred_probs.extend(y_pred_prob)

                            if y_groundtruths==[] or skip_outer:
                                wandb.finish()
                                continue

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

                    del combined_array
