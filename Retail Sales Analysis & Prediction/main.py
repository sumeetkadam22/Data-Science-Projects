import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------------
# Create Output Folder
# -------------------------------------------------

os.makedirs("output", exist_ok=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv("data/Superstore.csv", encoding="latin1")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------------------------
# Data Cleaning
# -------------------------------------------------

df = df.drop_duplicates()

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

print("\nDuplicates Removed Successfully")

# -------------------------------------------------
# Statistical Summary
# -------------------------------------------------

print("\nStatistical Summary:")
print(df.describe())

# -------------------------------------------------
# Sales by Region
# -------------------------------------------------

plt.figure(figsize=(8,5))

region_sales = df.groupby("Region")["Sales"].sum()

region_sales.plot(kind="bar")

plt.title("Sales by Region")
plt.ylabel("Total Sales")
plt.tight_layout()

plt.savefig("output/sales_by_region.png")
plt.close()

# -------------------------------------------------
# Sales by Category
# -------------------------------------------------

plt.figure(figsize=(8,5))

category_sales = df.groupby("Category")["Sales"].sum()

category_sales.plot(kind="bar")

plt.title("Sales by Category")
plt.ylabel("Total Sales")
plt.tight_layout()

plt.savefig("output/category_sales.png")
plt.close()

# -------------------------------------------------
# Monthly Sales Trend
# -------------------------------------------------

df["Month"] = df["Order Date"].dt.to_period("M")

monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(12,6))

monthly_sales.plot()

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig("output/monthly_sales.png")
plt.close()

# -------------------------------------------------
# Machine Learning
# Predict Sales Using Time Index
# -------------------------------------------------

monthly_sales_df = monthly_sales.reset_index()

monthly_sales_df["Month"] = range(len(monthly_sales_df))

X = monthly_sales_df[["Month"]]
y = monthly_sales_df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# -------------------------------------------------
# Actual vs Predicted Plot
# -------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()

plt.savefig("output/prediction_plot.png")
plt.close()

# -------------------------------------------------
# Save Model Results
# -------------------------------------------------

with open("output/model_results.txt", "w") as file:
    file.write("Linear Regression Results\n")
    file.write(f"MAE: {mae:.2f}\n")
    file.write(f"R2 Score: {r2:.2f}\n")

# -------------------------------------------------
# Business Insights
# -------------------------------------------------

insights = f"""
Retail Sales Analysis Insights

1. Total Regions Analyzed: {df['Region'].nunique()}

2. Top Category:
{category_sales.idxmax()}

3. Highest Revenue Region:
{region_sales.idxmax()}

4. Total Sales:
{df['Sales'].sum():,.2f}

5. Machine Learning Model:
Linear Regression

6. R2 Score:
{r2:.2f}

Conclusion:
The project identified important sales patterns and built
a predictive model to estimate future sales performance.
"""

with open("output/insights.txt", "w") as file:
    file.write(insights)

print("\nProject Completed Successfully!")
print("Check the OUTPUT folder for charts and reports.")