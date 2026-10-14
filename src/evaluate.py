import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score


def best_threshold(y_true, probs):
    """Pick the probability cutoff with the best F1 score."""
    thresholds = np.arange(0.10, 0.90, 0.05)
    scores = [f1_score(y_true, (probs >= t).astype(int)) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])


def evaluate(model, X_test, y_test, threshold):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_test, probs)
    print(f"Test ROC-AUC: {auc:.3f}")
    print(f"Classification report at threshold {threshold:.2f}:")
    print(classification_report(y_test, preds, digits=3))
    return auc
