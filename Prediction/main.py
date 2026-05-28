import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Create output folder
os.makedirs('output', exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df['Price'] = housing.target

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# -----------------------------
# FEATURES & TARGET
# -----------------------------

X = df.drop('Price', axis=1)
y = df['Price']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# MODELS
# -----------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = {}

# -----------------------------
# TRAIN & EVALUATE
# -----------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }

    print(f"{name} Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

# -----------------------------
# BEST MODEL
# -----------------------------

best_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)

# -----------------------------
# VISUALIZATION 1
# Actual vs Predicted
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.savefig('output/actual_vs_predicted.png')

plt.close()

# -----------------------------
# VISUALIZATION 2
# Feature Importance
# -----------------------------

importance = best_model.feature_importances_

feature_importance = pd.Series(
    importance,
    index=X.columns
).sort_values()

plt.figure(figsize=(10, 6))

feature_importance.plot(kind='barh')

plt.title("Feature Importance")

plt.savefig('output/feature_importance.png')

plt.close()

# -----------------------------
# SAVE RESULTS
# -----------------------------

with open('output/model_scores.txt', 'w') as f:

    for model_name, metrics in results.items():

        f.write(f"{model_name}\n")

        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.2f}\n")

        f.write("\n")

print("\nProject completed successfully.")
print("Outputs saved in output folder.")