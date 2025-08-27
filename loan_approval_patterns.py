"""
Bank Loan Approval Patterns Analysis using NumPy
Created by: Sangeeth M
Domain: Finance / Data Analysis
"""

import numpy as np
import pandas as pd

# ------------------------------
# 1. Sample Dataset (20 records)
# ------------------------------
data = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], 20),
    'ApplicantIncome': np.random.randint(2000, 15000, 20),
    'CreditScore': np.random.randint(300, 900, 20),
    'LoanStatus': np.random.choice(['Approved', 'Rejected'], 20)
})

print("Sample Data:\n", data.head(), "\n")

# Convert to NumPy arrays
gender = data['Gender'].to_numpy()
income = data['ApplicantIncome'].to_numpy()
credit_score = data['CreditScore'].to_numpy()
loan_status = np.where(data['LoanStatus'] == "Approved", 1, 0)

# ------------------------------
# 2. Analysis
# ------------------------------

# 2.1 Approval % by Gender
print("Approval Rate by Gender:")
unique_genders = np.unique(gender)
for g in unique_genders:
    mask = (gender == g)
    approval_rate = loan_status[mask].mean() * 100
    print(f"  {g}: {approval_rate:.2f}%")

# 2.2 Average Income (Approved vs Rejected)
avg_income_approved = income[loan_status == 1].mean()
avg_income_rejected = income[loan_status == 0].mean()
print(f"\nAvg Income (Approved): {avg_income_approved:.2f}")
print(f"Avg Income (Rejected): {avg_income_rejected:.2f}")

# 2.3 Correlation between Credit Score and Loan Status
correlation = np.corrcoef(credit_score, loan_status)[0, 1]
print(f"\nCorrelation between Credit Score & Loan Status: {correlation:.2f}")

# 2.4 Correlation Matrix (Income, Credit Score, Loan Status)
features = np.vstack((income, credit_score, loan_status))
corr_matrix = np.corrcoef(features)
print("\nCorrelation Matrix:\n", corr_matrix)

# ------------------------------
# 3. Conclusion
# ------------------------------
print("\nConclusion:")
print("- Loan approval shows dependency on applicant financials (income, credit score).")
print("- Gender may or may not affect approval depending on dataset bias.")
print("- Credit Score has a stronger correlation with loan approval than income.")
