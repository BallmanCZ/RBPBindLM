from math import sqrt

from sklearn.metrics import roc_auc_score


def calculate_performace(num, y_pred, y_prob, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(num):
        if y_test[index] == 1:
            if y_test[index] == y_pred[index]:
                tp += 1
            else:
                fn += 1
        else:
            if y_test[index] == y_pred[index]:
                tn += 1
            else:
                fp += 1

    acc = float(tp + tn) / num
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 0.0 if (precision + sensitivity) == 0 else 2.0 * precision * sensitivity / (precision + sensitivity)
    auc = roc_auc_score(y_test, y_prob)
    balanced_acc = float(sensitivity + specificity) / 2.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 0.0 if denom == 0 else float(tp * tn - fp * fn) / sqrt(denom)
    return precision, sensitivity, specificity, acc, balanced_acc, auc, mcc, f1
