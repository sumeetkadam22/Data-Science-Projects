# Retail Store Sales Analysis and Prediction

## Project Overview

This project focuses on analyzing retail sales data and predicting future sales using Machine Learning. The dataset contains information about customer orders, products, categories, regions, and sales performance.

The project demonstrates the complete data science workflow including data cleaning, exploratory data analysis (EDA), visualization, and predictive modeling.

---

## Objectives

- Analyze retail sales trends and patterns.
- Identify top-performing categories and regions.
- Visualize key business metrics.
- Build a machine learning model to predict sales.
- Generate actionable business insights.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Dataset

Dataset: Sample Superstore Dataset

The dataset includes:

- Order Details
- Customer Information
- Product Categories
- Sales
- Profit
- Region

---

## Project Structure

```text
Retail-Sales-Analysis/
│
├── data/
│   └── Superstore.csv
│
├── output/
│   ├── sales_by_region.png
│   ├── category_sales.png
│   ├── monthly_sales.png
│   ├── prediction_plot.png
│   ├── insights.txt
│   └── model_results.txt
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features

### Data Cleaning

- Removed duplicate records.
- Converted date columns into proper datetime format.
- Checked and handled missing values.

### Exploratory Data Analysis (EDA)

- Statistical summary of the dataset.
- Sales distribution analysis.
- Category-wise sales analysis.
- Region-wise sales analysis.
- Monthly sales trend analysis.

### Data Visualization

The project generates the following visualizations:

1. Sales by Region
2. Sales by Category
3. Monthly Sales Trend
4. Actual vs Predicted Sales

### Machine Learning

Algorithm Used:

- Linear Regression

Evaluation Metrics:

- Mean Absolute Error (MAE)
- R² Score

---

## How to Run

### Step 1: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 2: Place Dataset

Create a folder named `data` and place:

```text
Superstore.csv
```

inside it.

### Step 3: Run the Project

```bash
python main.py
```

---

## Output Files

After execution, the following files will be generated inside the `output` folder:

### Visualizations

- sales_by_region.png
- category_sales.png
- monthly_sales.png
- prediction_plot.png

### Reports

- insights.txt
- model_results.txt

---

## Key Findings

- Sales performance varies significantly across regions.
- Certain product categories contribute more revenue than others.
- Monthly sales trends reveal seasonal business patterns.
- Predictive modeling can help estimate future sales performance.

---

## Machine Learning Results

The Linear Regression model was trained and tested on historical sales data.

Performance was evaluated using:

- Mean Absolute Error (MAE)
- R² Score

Results are saved automatically in:

```text
output/model_results.txt
```

---

## Business Insights

The project helps businesses:

- Identify high-performing regions.
- Understand customer purchasing patterns.
- Optimize inventory planning.
- Forecast future sales.

---

## Learning Outcomes

Through this project, the following skills were developed:

- Data Cleaning
- Exploratory Data Analysis
- Data Visualization
- Machine Learning
- Model Evaluation
- Business Insight Generation

---

## Conclusion

This project demonstrates how data science techniques can be applied to real-world retail data to generate insights and support business decision-making. By combining visualization and machine learning, organizations can better understand historical performance and make informed predictions about future sales.