import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load and prepare data
df = pd.read_excel("Dataset2Use_Assignment2.xlsx")
df.rename(columns={"ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)": "Status"}, inplace=True)
features = df.columns[:11]

# Normalize
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df_scaled[features])

X = df_scaled.drop(columns=["Status", "ΕΤΟΣ"])
y = df_scaled["Status"]

# Define classifiers
models = {
    "LDA": LinearDiscriminantAnalysis(),
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "GradientBoosting": GradientBoostingClassifier()
}

# Prepare results
results = []

# Cross-validation
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Balance training set (3:1)
    train_df = X_train.copy()
    train_df["Status"] = y_train

    healthy = train_df[train_df["Status"] == 1]
    bankrupt = train_df[train_df["Status"] == 2]

    healthy_sampled = healthy.sample(n=3 * len(bankrupt), random_state=42)
    train_balanced = pd.concat([bankrupt, healthy_sampled])
    y_train_bal = train_balanced.pop("Status")
    X_train_bal = train_balanced

    for name, model in models.items():
        for dataset_name, Xt, yt, balanced in [
            ("Train", X_train_bal, y_train_bal, "Yes"),
            ("Test", X_test, y_test, "N/A")
        ]:
            model.fit(X_train_bal, y_train_bal)
            y_pred = model.predict(Xt)
            y_prob = model.predict_proba(Xt)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xt)

            tn, fp, fn, tp = confusion_matrix(yt, y_pred).ravel()

            results.append({
                "Classifier": name,
                "Dataset": dataset_name,
                "Fold": fold_idx,
                "Balanced": balanced,
                "Train Size": len(y_train_bal),
                "Bankrupt in Train": sum(y_train_bal == 2),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Accuracy": round(accuracy_score(yt, y_pred), 2),
                "Precision": round(precision_score(yt, y_pred), 2),
                "Recall": round(recall_score(yt, y_pred), 2),
                "F1": round(f1_score(yt, y_pred), 2),
                "AUC": round(roc_auc_score(yt, y_prob), 2)
            })

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("balancedDataOutcomes.csv", index=False)
print("✅ Results saved to 'balancedDataOutcomes.csv'")
