def preprocess(df):
    df = df.dropna().copy()
    df["debt_ratio"] = df["debt"] / df["income"]
    df["loan_to_income"] = df["loan_amount"] / df["income"]
    df["risk_score"] = df["credit_score"] - (df["late_payments"] * 20)
    return df
