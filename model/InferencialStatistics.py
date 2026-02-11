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
os.makedirs("IFAoutput", exist_ok=True)

data = pd.read_csv("D:\\RUAPStrokeProject\\model\\healthcare-dataset-stroke-data.csv")

print("Shape of the dataset:", data.shape)

numerical_features_for_inferential_analysis = ['age', 'avg_glucose_level', 'bmi']
print("Numerical features selected for inferential analysis:", numerical_features_for_inferential_analysis)

for feature in numerical_features_for_inferential_analysis:
    # Separate data into stroke and no-stroke groups
    stroke_group = data[data['stroke'] == 1][feature]
    no_stroke_group = data[data['stroke'] == 0][feature]

    # Perform Mann-Whitney U test
    u_statistic, p_value = stats.mannwhitneyu(stroke_group, no_stroke_group, alternative='two-sided')

    print(f"\n--- Mann-Whitney U Test for {feature} ---")
    print(f"U-statistic: {u_statistic:.2f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("The difference between stroke and no-stroke groups is statistically significant (p < 0.05).")
    else:
        print("The difference between stroke and no-stroke groups is not statistically significant (p >= 0.05).")

    # Create violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=data, x='stroke', y=feature, hue='stroke', palette='viridis', legend=False)
    plt.title(f'Distribution of {feature} by Stroke Status')
    plt.xlabel('Stroke Status (0: No Stroke, 1: Stroke)')
    plt.ylabel(feature)
    plt.xticks([0, 1], ['No Stroke', 'Stroke'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"IFAoutput/violin_plot_{feature}.png")
    plt.close()
    
categorical_features_for_inferential_analysis = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
print("Categorical features selected for inferential analysis:", categorical_features_for_inferential_analysis)

for feature in categorical_features_for_inferential_analysis:
    # Create a contingency table
    contingency_table = pd.crosstab(data[feature], data['stroke'])

    # Perform Chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"\n--- Chi-squared Test for '{feature}' vs. 'stroke' ---")
    print(f"Chi2 Statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    if p_value < 0.05:
        print("There is a statistically significant association between", feature, "and stroke (p < 0.05).")
    else:
        print("There is no statistically significant association between", feature, "and stroke (p >= 0.05).")

    # Create a grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=feature, hue='stroke', palette='viridis')
    plt.title(f'Distribution of {feature} by Stroke Status')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Stroke', labels=['No Stroke (0)', 'Stroke (1)']) # Explicitly setting labels for legend
    plt.tight_layout()
    plt.savefig(f"IFAoutput/grouped_bar_chart_{feature}.png")
    plt.close()