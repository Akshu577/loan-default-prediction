from src.data_generator import generate_data
from src.train import train_model
from src.evaluate import evaluate
from src.preprocess import preprocess
import pandas as pd
import shap
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 1. Generate dataset automatically
generate_data()

# 2. Load dataset
df = pd.read_csv("data/loan_data.csv")

# 3. Preprocess 
df = preprocess(df)

# 4. Split data
X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# 5. Train model
model = train_model(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# 6. Evaluate
score = evaluate(model, X_test, y_test)

# STEP 1: Get probabilities
y_probs = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0
best_f1 = 0

for t in thresholds:
    y_pred_temp = (y_probs > t).astype(int)
    f1 = f1_score(y_test, y_pred_temp)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("Best Threshold:", best_threshold)
print("Best F1 Score:", best_f1)
y_pred = (y_probs > best_threshold).astype(int)


print(classification_report(y_test, y_pred))
print("Model Accuracy:", score)