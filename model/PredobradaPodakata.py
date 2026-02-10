import pandas as pd
import numpy as np

# Load dataset (use raw string for Windows path)
data = pd.read_csv(r"D:\RUAPStrokeProject\model\healthcare-dataset-stroke-data.csv")

# Quick initial inspection
print("Initial shape:", data.shape)
print(data.head())

# Drop identifier column if present
if 'id' in data.columns:
    data = data.drop(columns=['id'])

data = data.dropna().reset_index(drop=True)

# Define target and features
TARGET = 'stroke'
if TARGET not in data.columns:
    raise KeyError(f"Target column '{TARGET}' not found in dataset")

y = data[TARGET].copy()
X = data.drop(columns=[TARGET])

# Identify categorical columns (object or category dtype)
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns:", cat_cols)

# Create dummy variables for categorical features; keep missing values as-is
X_dummies = pd.get_dummies(X, columns=cat_cols, drop_first=True, dummy_na=False)

# Recombine features and target
df_prepared = pd.concat([X_dummies, y], axis=1)
print("Prepared dataset shape (with outliers):", df_prepared.shape)

# Remove outliers from numeric columns using IQR, but do not drop rows for NaNs
num_cols = X_dummies.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns to check for outliers:", num_cols)

# Compute IQR per numeric column
Q1 = X_dummies[num_cols].quantile(0.25)
Q3 = X_dummies[num_cols].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Build a mask: for each column keep rows where value is NaN OR within [lower, upper]
mask = pd.Series(True, index=X_dummies.index)
for col in num_cols:
    col_vals = X_dummies[col]
    keep_col = col_vals.isna() | ((col_vals >= lower[col]) & (col_vals <= upper[col]))
    mask &= keep_col

# Apply mask
df_no_outliers = df_prepared[mask].copy()
print("Shape after outlier removal (NaNs preserved):", df_no_outliers.shape)

print(df_prepared.head())
print(df_no_outliers.head())

# Save outputs for later use
out_path_base = r"D:\RUAPStrokeProject\stroke_prepared"
df_prepared.to_csv(out_path_base + "_with_outliers.csv", index=False)
df_no_outliers.to_csv(out_path_base + "_no_outliers.csv", index=False)

print("Prepared files saved to D:\\RUAPStrokeProject\\")