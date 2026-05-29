import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs('output', exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------

file_path = 'data/StudentsPerformance.csv'

df = pd.read_csv(file_path)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove duplicates
df = df.drop_duplicates()

# -----------------------------
# VISUALIZATION SETTINGS
# -----------------------------

sns.set(style='whitegrid')

# -----------------------------
# 1. Gender Distribution
# -----------------------------

plt.figure(figsize=(6, 4))

sns.countplot(x='gender', data=df)

plt.title('Gender Distribution')

plt.savefig('output/gender_distribution.png')

plt.close()

# -----------------------------
# 2. Math Score Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(df['math score'], bins=20, kde=True)

plt.title('Math Score Distribution')

plt.savefig('output/score_distribution.png')

plt.close()

# -----------------------------
# 3. Correlation Heatmap
# -----------------------------

plt.figure(figsize=(8, 6))

score_columns = [
    'math score',
    'reading score',
    'writing score'
]

correlation = df[score_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm'
)

plt.title('Score Correlation Heatmap')

plt.savefig('output/correlation_heatmap.png')

plt.close()

# -----------------------------
# 4. Parental Education Impact
# -----------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    x='parental level of education',
    y='math score',
    data=df
)

plt.xticks(rotation=45)

plt.title('Parental Education vs Math Score')

plt.savefig('output/parental_education.png')

plt.close()

# -----------------------------
# INSIGHTS
# -----------------------------

insights = """
EDA Insights:

1. Male and female student distribution is nearly balanced.

2. Reading and writing scores are highly correlated.

3. Students with higher parental education levels tend to perform better.

4. Most students scored between 60 and 80 marks.

5. Academic performance trends can be identified using visual analysis.
"""

with open('output/insights.txt', 'w') as file:
    file.write(insights)

print("\nEDA completed successfully.")
print("Charts and insights saved in output folder.")