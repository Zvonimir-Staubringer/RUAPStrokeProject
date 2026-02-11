import os
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend to avoid Tk/Tcl dependency
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis, shapiro, chi2_contingency, chisquare

# ensure output directory exists
os.makedirs("EDAoutput", exist_ok=True)

data = pd.read_csv("D:\\RUAPStrokeProject\\model\\healthcare-dataset-stroke-data.csv")

print("Shape of the dataset:", data.shape)

print(data.head())
print(data.info())
print(data.dtypes)
print(data.describe())

print(data.isna().sum())
data.dropna(inplace=True)
print(data.isna().sum())

numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

numerical_features_for_analysis = [col for col in numerical_cols if col not in ['id', 'stroke']]

for col in numerical_features_for_analysis:
    print(f"\n--- Descriptive Statistics for '{col}' ---")
    print(f"Mean: {data[col].mean():.2f}")
    print(f"Median: {data[col].median():.2f}")
    print(f"Standard Deviation: {data[col].std():.2f}")
    print(f"Skewness: {data[col].skew():.2f}")
    print(f"Kurtosis: {data[col].kurtosis():.2f}")

    if col in ('hypertension', 'heart_disease'):
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data, x=col, order=[0, 1], palette='viridis')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.savefig(f"EDAoutput/{col}_countplot.png")
        plt.close()
    else:
        # Histogram
        plt.figure(figsize=(10, 5))
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend([col])
        plt.savefig(f"EDAoutput/{col}_histogram.png")
        plt.close()

        # Box Plot
        plt.figure(figsize=(10, 5))
        sns.boxplot(y=data[col])
        plt.title(f'Box Plot of {col}')
        plt.ylabel(col)
        plt.savefig(f"EDAoutput/{col}_boxplot.png")
        plt.close()

for col in categorical_cols:
    print(f"\n--- Value Counts for '{col}' ---")
    print(data[col].value_counts())

    plt.figure(figsize=(10, 6))
    order = data[col].value_counts().index.tolist()
    sns.countplot(data=data, x=col, order=order, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"EDAoutput/{col}_countplot.png")
    plt.close()

numerical_cols_for_corr = [col for col in numerical_cols if col != 'id']

correlation_matrix = data[numerical_cols_for_corr].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of Numerical Features with Stroke')
plt.savefig("EDAoutput/correlation_matrix.png")
plt.close()

# Analyze stroke patients separately
stroke_patients_df = data[data['stroke'] == 1]
print("Shape of the original DataFrame:", data.shape)
print("Shape of the DataFrame containing only stroke patients:", stroke_patients_df.shape)
print("First 5 rows of stroke_patients_df:")
print(stroke_patients_df.head())

plt.figure(figsize=(10, 6))
sns.kdeplot(data=stroke_patients_df, x='age', fill=True, color='purple')
plt.title('Age Distribution of Stroke Patients')
plt.xlabel('Age')
plt.ylabel('Density')
plt.savefig("EDAoutput/age_distribution_stroke_patients.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=stroke_patients_df, x='bmi', fill=True, color='purple')
plt.title('BMI Distribution of Stroke Patients')
plt.xlabel('BMI')
plt.ylabel('Density')
plt.savefig("EDAoutput/bmi_distribution_stroke_patients.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=stroke_patients_df, x='ever_married', order=stroke_patients_df['ever_married'].value_counts().index.tolist(), palette='viridis')
plt.title('Marital Status of Stroke Patients')
plt.xlabel('Ever Married')
plt.ylabel('Count')
plt.savefig("EDAoutput/ever_married_stroke_patients.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=stroke_patients_df, x='hypertension', order=[0, 1], palette='viridis')
plt.title('Hypertension Status Among Stroke Patients')
plt.xlabel('Hypertension (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Hypertension', 'Hypertension'])
plt.savefig("EDAoutput/hypertension_stroke_patients.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=stroke_patients_df, x='heart_disease', order=[0, 1], palette='viridis')
plt.title('Heart Disease Status Among Stroke Patients')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'])
plt.savefig("EDAoutput/heart_disease_stroke_patients.png")
plt.close()