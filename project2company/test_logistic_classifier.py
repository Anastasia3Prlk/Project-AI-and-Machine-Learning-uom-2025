import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel("Dataset2Use_Assignment2.xlsx")
df.rename(columns={"ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)": "Status"}, inplace=True)

# Normalize all numeric columns (except Status & Year)
features_to_scale = df.columns[:11]
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Prepare data
X = df_scaled.drop(columns=["Status", "ΕΤΟΣ"])
y = df_scaled["Status"]

# One fold only
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
train_idx, test_idx = next(iter(skf.split(X, y)))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Balance train set (3:1)
X_train_full = X_train.copy()
X_train_full["Status"] = y_train

healthy = X_train_full[X_train_full["Status"] == 1]
bankrupt = X_train_full[X_train_full["Status"] == 2]
healthy_sampled = healthy.sample(n=3 * len(bankrupt), random_state=42)

X_train_balanced = pd.concat([bankrupt, healthy_sampled])
y_train_balanced = X_train_balanced.pop("Status")

# Fit logistic regression
model = LogisticRegression()
model.fit(X_train_balanced, y_train_balanced)

# Evaluate function
def evaluate(model, X, y, dataset_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    print(f"\n📊 Evaluation on {dataset_name} set:")
    print(f"Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1: {f1:.2f} | AUC: {auc:.2f}")

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Evaluate on both train and test
evaluate(model, X_train_balanced, y_train_balanced, "Train")
evaluate(model, X_test, y_test, "Test")
