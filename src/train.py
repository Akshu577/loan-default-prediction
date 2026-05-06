from xgboost import XGBClassifier
import joblib
import os

def train_model(X_train, y_train):
    model = XGBClassifier(n_estimators=200, max_depth=5,scale_pos_weight =0.2,learning_rate=0.1)
    model.fit(X_train, y_train)

    #  create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    joblib.dump(model,"models/xgboost_model.pkl")
    return model