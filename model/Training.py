"""
=============================================================
Stroke Prediction Model - Training and Evaluation
=============================================================
Dataset: Healthcare Stroke Data (Prepared)
Target: Stroke prediction (binary classification)
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Učitavanje pripremljenog dataset-a (sa outliers)
df = pd.read_csv(r"D:\RUAPStrokeProject\stroke_prepared_with_outliers.csv")

print("Dataset informacije:")
print(f"Ukupno primjera: {len(df)}")
print(f"\nDistribucija ciljne klase (stroke):")
print(df['stroke'].value_counts())
print(f"\nOdnos klasa: {df['stroke'].value_counts()[0]} (nema udara) vs {df['stroke'].value_counts()[1]} (udar)")
print(f"Postotak sa moždanim udarom: {(df['stroke'].sum() / len(df) * 100):.2f}%")

# Prepare features and target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Razdvajanje na trening i test setove (80% trening, 20% test, stratificirano po cilju)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrenin set: {len(X_train)} primjera")
print(f"Test set: {len(X_test)} primjera")
print(f"Procenat sa udarom u treningu: {(y_train.sum() / len(y_train) * 100):.2f}%")
print(f"Procenat sa udarom u testu: {(y_test.sum() / len(y_test) * 100):.2f}%")

# Modeli za usporedbu
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier()
}

# Usporedba modela
print("\nUsporedba modela za predviđanje moždanog udara:")
for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")

# Evaluacija modela koji imaju najbolje performanse na test setu - Random Forest
model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"\nAccuracy na test setu (default threshold=0.5): {accuracy:.4f}")
print("Matrica zabune:\n", confusion_matrix(y_test, preds))
print("\nClassification Report (default threshold=0.5):")
print(classification_report(y_test, preds, target_names=['Bez moždanog udara', 'Moždani udar']))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bez moždanog udara', 'Moždani udar'],
            yticklabels=['Bez moždanog udara', 'Moždani udar'])
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.title(f"Matrica zabune - Threshold=0.5")
plt.savefig("output/forest_default_confusion_matrix.png")
plt.close()

# Pronalaženje optimalnog threshold-a za Random Forest model kako bi se poboljšala detekcija moždanog udara (recall)
print("\nOptimizacija threshold vrijednosti...")
probs = model.predict_proba(X_test)[:, 1]
best_threshold = 0.5
best_f1 = 0
from sklearn.metrics import f1_score

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_threshold = (probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimalni threshold: {best_threshold:.2f} (F1-score: {best_f1:.4f})")

# Evaluacija sa optimiziranim threshold-om
preds_optimized = (probs >= best_threshold).astype(int)
print(f"\nAccuracy sa optimiziranim threshold={best_threshold:.2f}: {accuracy_score(y_test, preds_optimized):.4f}")
print("Matrica zabune sa optimiziranim threshold:\n", confusion_matrix(y_test, preds_optimized))
cm = confusion_matrix(y_test, preds_optimized)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bez moždanog udara', 'Moždani udar'],
            yticklabels=['Bez moždanog udara', 'Moždani udar'])
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.title(f"Matrica zabune - Threshold={best_threshold:.2f}")
plt.savefig("output/forest_optimised_confusion_matrix.png")
plt.close()
print("\nClassification Report (optimizirani threshold):")
print(classification_report(y_test, preds_optimized, target_names=['Bez moždanog udara', 'Moždani udar']))

# Treniranje i evaluacija Logistic Regression modela (imao je najbolje performanse u usporedbi modela))
model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluacija modela sa default thresholdom (0.5)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"\nAccuracy na test setu (default threshold=0.5): {accuracy:.4f}")
print("Matrica zabune:\n", confusion_matrix(y_test, preds))
print("\nClassification Report (default threshold=0.5):")
print(classification_report(y_test, preds, target_names=['Bez moždanog udara', 'Moždani udar']))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bez moždanog udara', 'Moždani udar'],
            yticklabels=['Bez moždanog udara', 'Moždani udar'])
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.title(f"Matrica zabune - Threshold={best_threshold:.2f}")
plt.savefig("output/logistic_regression_confusion_matrix.png")
plt.close()

# Pronalaženje optimalnog threshold-a za Logistic Regression model kako bi se poboljšala detekcija moždanog udara (recall)
print("\nOptimizacija threshold vrijednosti...")
probs = model.predict_proba(X_test)[:, 1]
best_threshold = 0.5
best_f1 = 0
from sklearn.metrics import f1_score

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_threshold = (probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimalni threshold: {best_threshold:.2f} (F1-score: {best_f1:.4f})")

# Evaluacija sa optimiziranim threshold-om
preds_optimized = (probs >= best_threshold).astype(int)
print(f"\nAccuracy sa optimiziranim threshold={best_threshold:.2f}: {accuracy_score(y_test, preds_optimized):.4f}")
print("Matrica zabune sa optimiziranim threshold:\n", confusion_matrix(y_test, preds_optimized))
print("\nClassification Report (optimizirani threshold):")
print(classification_report(y_test, preds_optimized, target_names=['Bez moždanog udara', 'Moždani udar']))

cm = confusion_matrix(y_test, preds_optimized)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bez moždanog udara', 'Moždani udar'],
            yticklabels=['Bez moždanog udara', 'Moždani udar'])
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.title(f"Matrica zabune - Threshold={best_threshold:.2f}")
plt.savefig("output/optimised_confusion_matrix.png")
plt.close()

# Račun važnosti značajki (feature importance)
if hasattr(model, "feature_importances_"):
    importances = np.array(model.feature_importances_, dtype=float)
elif hasattr(model, "coef_"):
    coef = np.array(model.coef_, dtype=float)
    if coef.ndim == 1 or coef.shape[0] == 1:
        importances = np.abs(coef.ravel())
    else:
        importances = np.mean(np.abs(coef), axis=0)
else:
    importances = np.zeros(X.shape[1], dtype=float)

feature_names = np.array(X.columns)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Važnost značajki za predviđanje moždanog udara")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
plt.ylabel("Važnost")
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.close()

# Spremanje modela i label encodersa za buduću upotrebu
joblib.dump(model, "model/stroke_prediction_model.pkl")
joblib.dump(label_encoders, "model/stroke_encoders.pkl")
print("\nModel i encoders su spremljeni!")

# Unakrsna validacija (cross-validation) za procjenu stabilnosti modela
cv_scores = cross_val_score(model, X, y, cv=5)
print("\n5-Fold Cross-Validation rezultati:", cv_scores)
print("Prosječna CV accuracy:", cv_scores.mean())

plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), cv_scores, marker='o', linewidth=2, markersize=8)
plt.title("5-Fold Cross Validation Accuracy")
plt.xlabel("Fold broj")
plt.ylabel("Accuracy")
plt.ylim(0.6, 1.0)
plt.grid(True, alpha=0.3)
plt.savefig("output/cross_validation.png")
plt.close()

# Learning curve analiza za procjenu kako model uči s više podataka
print("\nIzračunavanje learning curve...")
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training accuracy", linewidth=2, marker='o')
plt.plot(train_sizes, test_mean, label="Validation accuracy", linewidth=2, marker='s')
plt.xlabel("Broj primjera u treningu")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Stroke Prediction")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("output/learning_curve.png")
plt.close()

# ROC krivulja analiza za procjenu performansi modela u detekciji moždanog udara
print("Izračunavanje ROC curve...")
probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stroke Prediction")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("output/roc_curve.png")
plt.close()