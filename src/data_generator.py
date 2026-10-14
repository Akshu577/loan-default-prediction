import os

import numpy as np
import pandas as pd


def generate_data(path="data/loan_data.csv", n=5000, seed=42):
    """Create a synthetic loan applicant dataset.

    Default is drawn from a probability (logistic function of the applicant's
    risk factors plus noise), not from a fixed rule, so two applicants with the
    same profile can have different outcomes, like in real lending data.
    """
    if os.path.exists(path):
        print("Dataset already exists. Skipping generation.")
        return

    print("Generating dataset...")
    rng = np.random.default_rng(seed)

    income = rng.lognormal(mean=10.9, sigma=0.45, size=n).clip(15000, 250000).round(-2)
    credit_score = rng.normal(680, 70, n).clip(300, 850).round()
    loan_amount = (income * rng.uniform(0.05, 0.6, n)).clip(1000, 60000).round(-2)
    debt = (income * rng.beta(2, 5, n)).round(-2)
    late_payments = rng.poisson(np.where(credit_score < 620, 3.0, 0.8))
    past_loans = rng.poisson(2.0, n)

    # Log-odds of default: lower credit score, more late payments, higher debt
    # and loan burden increase risk; a history of past loans lowers it slightly.
    logit = (
        -2.2
        - 0.012 * (credit_score - 680)
        + 0.35 * late_payments
        + 2.2 * (debt / income)
        + 1.5 * (loan_amount / income)
        - 0.10 * past_loans
        + rng.normal(0, 0.5, n)
    )
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "debt": debt,
        "late_payments": late_payments,
        "past_loans": past_loans,
        "default": default,
    })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset created at: {path} ({n} rows, {default.mean():.1%} defaults)")
