import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder automatically
os.makedirs('output', exist_ok=True)

# Load dataset
file_path = 'data/netflix_titles.csv.csv'
df = pd.read_csv(file_path)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove duplicate rows
before_duplicates = df.shape[0]
df = df.drop_duplicates()
after_duplicates = df.shape[0]

print(f"\nDuplicates Removed: {before_duplicates - after_duplicates}")

# Fill missing values
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Not Available')
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna(df['rating'].mode()[0])

# Remove rows where release_year is missing
df = df.dropna(subset=['release_year'])

# -----------------------------
# OUTLIER HANDLING
# -----------------------------

df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

Q1 = df['release_year'].quantile(0.25)
Q3 = df['release_year'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

clean_df = df[
    (df['release_year'] >= lower_bound) &
    (df['release_year'] <= upper_bound)
]

# Save cleaned dataset
clean_df.to_csv('output/cleaned_dataset.csv', index=False)

print("\nCleaned dataset saved successfully.")

# -----------------------------
# DATA VISUALIZATION
# -----------------------------

sns.set(style='whitegrid')

# 1. Movies vs TV Shows
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=clean_df)
plt.title('Movies vs TV Shows on Netflix')
plt.savefig('output/chart1.png')
plt.close()

# 2. Top 10 Countries
plt.figure(figsize=(12, 6))
clean_df['country'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Countries Producing Netflix Content')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('output/chart2.png')
plt.close()

# 3. Content Added Per Year
plt.figure(figsize=(12, 6))
clean_df['release_year'].value_counts().sort_index().tail(20).plot()
plt.title('Netflix Content Release Trend')
plt.xlabel('Year')
plt.ylabel('Number of Releases')
plt.savefig('output/chart3.png')
plt.close()

# 4. Ratings Distribution
plt.figure(figsize=(10, 5))
sns.countplot(
    y='rating',
    data=clean_df,
    order=clean_df['rating'].value_counts().index
)
plt.title('Content Ratings Distribution')
plt.savefig('output/chart4.png')
plt.close()

print("\nVisualizations generated successfully.")