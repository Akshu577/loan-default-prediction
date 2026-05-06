def preprocess(df):
    # Example feature engineering
    df["debt_ratio"] = df["debt"] / df["income"]
    df["risk_score"]=df["credit_score"]-(df["late_payments"]*20)
    
    # You can add more cleaning here
    df = df.dropna()
    return df