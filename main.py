import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.data_generator import generate_data
from src.evaluate import best_threshold, evaluate
from src.preprocess import preprocess
from src.train import train_model

# 1. Generate the synthetic dataset (skipped if it already exists)
generate_data()

# 2. Load and preprocess
df = preprocess(pd.read_csv("data/loan_data.csv"))
X = df.drop("default", axis=1)
y = df["default"]

# 3. Split 60/20/20 into train / validation / test.
# The validation set is used for early stopping and for choosing the threshold,
# so the test set is only touched once, at the very end.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 4. Train
model = train_model(X_train, y_train, X_val, y_val)

# 5. Choose the decision threshold on the validation set
threshold, val_f1 = best_threshold(y_val, model.predict_proba(X_val)[:, 1])
print(f"Best threshold (validation): {threshold:.2f}  F1: {val_f1:.3f}")

# 6. Final evaluation on the test set
auc = evaluate(model, X_test, y_test, threshold)

# 7. Explain the model with SHAP
os.makedirs("images", exist_ok=True)
shap_values = shap.TreeExplainer(model).shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("images/shap_summary.png", dpi=150)
print("Saved SHAP plot to images/shap_summary.png")

# Save the threshold so the app uses the same cutoff
with open("models/threshold.json", "w") as f:
    json.dump({"threshold": threshold}, f)
