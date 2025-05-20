import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Load the Excel file
df = pd.read_excel("Dataset2Use_Assignment2.xlsx")
df.rename(columns={"ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)": "Status"}, inplace=True)

# Define X (features) and y (target)
X = df.drop(columns=["Status", "ΕΤΟΣ"])  # exclude label and year
y = df["Status"]

# Initialize Stratified K-Fold with 4 splits
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Start the loop over each fold
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n🔁 Fold {fold}")

    # Split the data
    X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]

    # Count original class distribution
    train_counts = X_train['Status'].value_counts()
    test_counts = X_test['Status'].value_counts()

    print(f"Train set - Healthy: {train_counts[1]}, Bankrupt: {train_counts[2]}")
    print(f"Test set  - Healthy: {test_counts[1]}, Bankrupt: {test_counts[2]}")

    # If healthy > 3 * bankrupt in train set, balance to 3:1
    if train_counts[1] > 3 * train_counts[2]:
        bankrupt_df = X_train[X_train["Status"] == 2]
        healthy_df = X_train[X_train["Status"] == 1]

        n_required_healthy = 3 * len(bankrupt_df)

        # Random sample of healthy companies
        healthy_sampled = healthy_df.sample(n=n_required_healthy, random_state=42)

        # Combine balanced data
        X_train_balanced = pd.concat([bankrupt_df, healthy_sampled])
    else:
        X_train_balanced = X_train

    # Reprint class distribution in balanced training set
    balanced_counts = X_train_balanced["Status"].value_counts()
    print(f"Balanced Train set - Healthy: {balanced_counts[1]}, Bankrupt: {balanced_counts[2]}")
