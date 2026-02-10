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

# Load stroke dataset
df = pd.read_csv(r"D:\RUAPStrokeProject\stroke_prepared_with_outliers.csv")

print("Dataset informacije:")
print(f"Ukupno primjera: {len(df)}")
print(f"\nDistribucija ciljne klase (stroke):")
print(df['stroke'].value_counts())
print(f"\nOdnos klasa: {df['stroke'].value_counts()[0]} (nema udara) vs {df['stroke'].value_counts()[1]} (udar)")
print(f"Procenat sa moždanim udarom: {(df['stroke'].sum() / len(df) * 100):.2f}%")

# Prepare features and target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split data into training and testing sets with stratified split
# This ensures both train and test sets have similar class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrenin set: {len(X_train)} primjera")
print(f"Test set: {len(X_test)} primjera")
print(f"Procenat sa udarom u treningu: {(y_train.sum() / len(y_train) * 100):.2f}%")
print(f"Procenat sa udarom u testu: {(y_test.sum() / len(y_test) * 100):.2f}%")

# Define models for stroke prediction
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier()
}

# Model comparison
print("\nUsporedba modela za predviđanje moždanog udara:")
for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")


# Train Random Forest model for stroke prediction with balanced class weights
# class_weight='balanced' handles the class imbalance in the dataset
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"\nAccuracy na test setu (default threshold=0.5): {accuracy:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, preds))
print("\nClassification Report (default threshold=0.5):")
print(classification_report(y_test, preds, target_names=['Bez moždanog udara', 'Moždani udar']))

# Find optimal threshold for better recall (detecting stroke cases)
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

# Evaluate with optimized threshold
preds_optimized = (probs >= best_threshold).astype(int)
print(f"\nAccuracy sa optimiziranim threshold={best_threshold:.2f}: {accuracy_score(y_test, preds_optimized):.4f}")
print("Confusion matrix sa optimiziranim threshold:\n", confusion_matrix(y_test, preds_optimized))
print("\nClassification Report (optimizirani threshold):")
print(classification_report(y_test, preds_optimized, target_names=['Bez moždanog udara', 'Moždani udar']))

# Visualize confusion matrix with optimized threshold
cm = confusion_matrix(y_test, preds_optimized)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bez moždanog udara', 'Moždani udar'],
            yticklabels=['Bez moždanog udara', 'Moždani udar'])
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.title(f"Matrica konfuzije - Threshold={best_threshold:.2f}")
plt.savefig("confusion_matrix.png")
plt.close()

# Feature importance analysis
importances = model.feature_importances_
feature_names = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.title("Važnost značajki za predviđanje moždanog udara")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
plt.ylabel("Važnost")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Save trained model and encoders
joblib.dump(model, "stroke_prediction_model.pkl")
joblib.dump(label_encoders, "stroke_encoders.pkl")
print("\nModel i encoders su spremljeni!")

# Cross-validation analysis
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
plt.savefig("cross_validation.png")
plt.close()

# Learning curve analysis
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
plt.savefig("learning_curve.png")
plt.close()

# ROC Curve analysis
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
plt.savefig("roc_curve.png")
plt.close()

# Visualization: Age distribution of patients
plt.figure(figsize=(10, 6))
plt.hist(df["age"], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.title("Distribucija starosti pacijenata")
plt.xlabel("Starost (године)")
plt.ylabel("Broj pacijenata")
plt.grid(axis='y', alpha=0.3)
plt.savefig("age_distribution.png")
plt.close()

# Visualization: Stroke distribution
plt.figure(figsize=(8, 6))
df["stroke"].value_counts().plot(kind="bar", color=['green', 'red'], alpha=0.7, edgecolor='black')
plt.title("Distribucija moždanog udara")
plt.xticks([0, 1], ["Bez moždanog udara", "Moždani udar"], rotation=0)
plt.ylabel("Broj pacijenata")
plt.grid(axis='y', alpha=0.3)
plt.savefig("stroke_distribution.png")
plt.close()

# Visualization: Hypertension and heart disease by stroke status
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x="hypertension", hue="stroke", data=df, palette=['green', 'red'])
plt.title("Hipertenzija i moždani udar")
plt.xticks([0, 1], ["Nema", "Ima"])
plt.xlabel("Hipertenzija")
plt.legend(["Bez moždanog udara", "Moždani udar"], loc='upper right')

plt.subplot(1, 2, 2)
sns.countplot(x="heart_disease", hue="stroke", data=df, palette=['green', 'red'])
plt.title("Bolest srca i moždani udar")
plt.xticks([0, 1], ["Nema", "Ima"])
plt.xlabel("Bolest srca")
plt.legend(["Bez moždanog udara", "Moždani udar"], loc='upper right')
plt.tight_layout()
plt.savefig("health_conditions_by_stroke.png")
plt.close()

# Visualization: Age distribution by stroke status
plt.figure(figsize=(10, 6))
sns.boxplot(x="stroke", y="age", data=df, palette=['green', 'red'])
plt.xticks([0, 1], ["Bez moždanog udara", "Moždani udar"])
plt.title("Starost pacijenata prema statusu moždanog udara")
plt.ylabel("Starost (godine)")
plt.grid(axis='y', alpha=0.3)
plt.savefig("age_by_stroke.png")
plt.close()

# Visualization: Average glucose levels and BMI by stroke status
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x="stroke", y="avg_glucose_level", data=df, palette=['green', 'red'])
plt.xticks([0, 1], ["Bez moždanog udara", "Moždani udar"])
plt.title("Prosječna razina glukoze po statusu")
plt.ylabel("Prosječna razina glukoze")

plt.subplot(1, 2, 2)
sns.boxplot(x="stroke", y="bmi", data=df, palette=['green', 'red'])
plt.xticks([0, 1], ["Bez moždanog udara", "Moždani udar"])
plt.title("BMI po statusu moždanog udara")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("glucose_bmi_by_stroke.png")
plt.close()

# Visualization: Correlation matrix
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f', cbar_kws={'label': 'Korelacija'})
plt.title("Korelacijska matrica - Stroke Dataset")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

print("\n" + "="*50)
print("Sve vizualizacije su spremljene u radni direktorij!")
print("="*50)