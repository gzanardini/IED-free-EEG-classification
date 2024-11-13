from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import numpy as np

def calculate_auc_bac(labels, scores):
    auc_score = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold_80_sensitivity = thresholds[np.where(tpr >= 0.80)[0][0]]
    adjusted_predictions = (scores >= threshold_80_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bac = ((sensitivity + specificity) / 2) * 100
    return auc_score, bac, fpr, tpr

def calculate_bac(labels, scores, sens_thresh):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold_sensitivity = thresholds[np.where(tpr >= sens_thresh)[0][0]]
    adjusted_predictions = (scores >= threshold_sensitivity).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, adjusted_predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bac = ((sensitivity + specificity) / 2) * 100
    return bac, fpr, tpr