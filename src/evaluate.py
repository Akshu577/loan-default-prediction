from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import roc_auc_score


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test,preds))
    auc = roc_auc_score(y_test, probs)
    print("AUC:", auc)
    return accuracy_score(y_test, preds)