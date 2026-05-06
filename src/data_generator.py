import os
import pandas as pd
import numpy as np


def generate_data(path="data/loan_data.csv"):
    if os.path.exists(path):
        print("Dataset already exists. Skipping generation.")
        return

    print("Generating dataset...")

    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "income": np.random.randint(20000, 120000, n),
        "credit_score": np.random.randint(300, 850, n),
        "loan_amount": np.random.randint(1000, 50000, n),
        "debt": np.random.randint(0, 50000, n),
        "late_payments": np.random.randint(0, 10, n),
        "past_loans": np.random.randint(0, 10, n)
    })

    prob = (
       0.4* (df["credit_score"] < 600).astype(int) +
       0.4* (df["late_payments"] > 3).astype(int) +
        0.2* (df["debt"] > df["income"] * 0.5).astype(int)
    ) 
    df["default"] = (prob + np.random.normal(0, 0.1, n) > 0.4).astype(int)

    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)

    print("Dataset created at:", path)