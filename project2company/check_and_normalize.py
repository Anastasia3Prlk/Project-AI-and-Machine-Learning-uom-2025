import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the Excel dataset
df = pd.read_excel("Dataset2Use_Assignment2.xlsx")

# Rename the 'Status' column for convenience
df.rename(columns={"ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)": "Status"}, inplace=True)

# -------------------------------
# 1. Check for missing values
# -------------------------------
missing_values = df.isnull().sum()

if missing_values.any():
    print("⚠️ WARNING: Missing values found:")
    print(missing_values[missing_values > 0])
else:
    print("✅ No missing values found in the dataset.")

# -------------------------------
# 2. Normalize numeric data (columns A to K)
# -------------------------------
# Select features to normalize (first 11 columns, excluding Status and Year)
columns_to_normalize = df.columns[:11]

# Create scaler and apply normalization
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

print("\n✅ Data normalization completed using Min-Max scaling.")
print("\nPreview of normalized data:")
print(df_normalized.head())
