import pandas as pd
import numpy as np

# Učitavanje dataset-a
data = pd.read_csv(r"D:\RUAPStrokeProject\model\healthcare-dataset-stroke-data.csv")

# Pregled osnovnih informacija o datasetu
print("Initial shape:", data.shape)
print(data.head())

# Uklanjanje id stupca ako postoji, jer nije koristan za modeliranje
if 'id' in data.columns:
    data = data.drop(columns=['id'])

data = data.dropna().reset_index(drop=True)

# Definiranje ciljne varijable i značajki
TARGET = 'stroke'
if TARGET not in data.columns:
    raise KeyError(f"Target column '{TARGET}' not found in dataset")

y = data[TARGET].copy()
X = data.drop(columns=[TARGET])

# Identificiranje kategorijskih stupaca
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns:", cat_cols)

# Stvaranje dummy varijabli za kategorijske značajke, bez drop_first i dummy_na da sačuvamo sve informacije
X_dummies = pd.get_dummies(X, columns=cat_cols, drop_first=True, dummy_na=False)

# Kombiniranje dummy varijabli s ciljnim stupcem
df_prepared = pd.concat([X_dummies, y], axis=1)
print("Prepared dataset shape (with outliers):", df_prepared.shape)

# Micanje outliersa - zadržavanje NaN vrijednosti
num_cols = X_dummies.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns to check for outliers:", num_cols)

# Račun IQR i definiranje granica za outliers
Q1 = X_dummies[num_cols].quantile(0.25)
Q3 = X_dummies[num_cols].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Stvaranje maske koja zadržava redove koji nisu outliers, ali dopušta NaN vrijednosti
mask = pd.Series(True, index=X_dummies.index)
for col in num_cols:
    col_vals = X_dummies[col]
    keep_col = col_vals.isna() | ((col_vals >= lower[col]) & (col_vals <= upper[col]))
    mask &= keep_col

# Primjena maske
df_no_outliers = df_prepared[mask].copy()
print("Shape after outlier removal (NaNs preserved):", df_no_outliers.shape)

print(df_prepared.head())
print(df_no_outliers.head())

# Spremanje pripremljenih datasetova
out_path_base = r"D:\RUAPStrokeProject\stroke_prepared"
df_prepared.to_csv(out_path_base + "_with_outliers.csv", index=False)
df_no_outliers.to_csv(out_path_base + "_no_outliers.csv", index=False)

print("Prepared files saved to D:\\RUAPStrokeProject\\")