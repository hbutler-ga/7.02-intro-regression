# Module 14 — Introduction to Machine Learning with Scikit-Learn: Regression

**Session Time:** 120 minutes  
**Primary Dataset:** Housing dataset  
**Primary Goal:** Build and evaluate regression models that predict home sale prices.

---

## Module Overview

In this module, students are introduced to the core machine learning workflow using `scikit-learn`.

Rather than trying to cover every major machine learning task at once, this module focuses on **regression** through a concrete question:

> Can we use information about a house to predict its sale price?

Students will begin with a simple linear regression model using one feature, then expand to multiple linear regression with several numeric features. They will also learn how to handle categorical variables by encoding them into numeric form before modeling.

The goal is not to build the most complex model. The goal is to understand the basic workflow:

1. Define the prediction problem
2. Choose features and a target
3. Split the data into training and test sets
4. Train a model
5. Make predictions
6. Evaluate performance
7. Interpret results and limitations

---

## Prerequisites

Students should be comfortable with:

- Python variables, functions, and basic control flow
- Pandas DataFrames
- Selecting columns from a DataFrame
- Basic exploratory data analysis
- Basic plotting with Matplotlib or Seaborn
- Summary statistics
- The idea of correlation

---

## Learning Objectives

By the end of this module, students will be able to:

- Explain machine learning as a process for learning patterns from data
- Distinguish between regression and classification problems
- Identify features `X` and a target variable `y`
- Split data into training and test sets using `train_test_split`
- Fit a simple linear regression model using `scikit-learn`
- Fit a multiple linear regression model using several features
- Encode categorical variables for use in a machine learning model
- Evaluate regression models using MAE, RMSE, and R²
- Interpret regression coefficients at a basic level
- Communicate model performance and limitations responsibly

---

## Session Breakdown

| Segment | Topic | Time |
|---:|---|---:|
| 1 | What is Machine Learning? | 10 min |
| 2 | Regression vs. Classification | 10 min |
| 3 | The Machine Learning Workflow | 10 min |
| 4 | Introduce the Housing Dataset | 10 min |
| 5 | Train/Test Split | 15 min |
| 6 | Simple Linear Regression | 20 min |
| 7 | Model Evaluation Metrics | 20 min |
| 8 | Multiple Linear Regression | 20 min |
| 9 | Encoding Categorical Variables | 20 min |
| 10 | Wrap-Up and Model Limitations | 5 min |

---

## 1. What is Machine Learning?

Machine learning is a field of artificial intelligence where algorithms learn patterns from data and use those patterns to make predictions or decisions.

In traditional programming, we write explicit rules.

```text
Input + Rules → Output
```

In machine learning, we provide examples and allow the algorithm to learn patterns.

```text
Input + Output Examples → Learned Pattern
```

Then we use the learned pattern to make predictions on new data.

```text
New Input + Learned Pattern → Prediction
```

### Example

A traditional rule might say:

> Larger homes are usually more expensive.

A machine learning model learns this relationship from actual housing data.

The model might learn that home sale price tends to increase as living area, overall quality, garage size, and neighborhood desirability increase.

---

## 2. Regression vs. Classification

Most beginner machine learning problems fall into two major supervised learning categories: **regression** and **classification**.

| Task Type | Prediction Type | Example |
|---|---|---|
| Regression | A number | Predicting home sale price |
| Classification | A category | Predicting whether an email is spam or not spam |

### Regression

Regression is used when the target variable is numeric.

Examples:

- Predicting home price
- Predicting salary
- Predicting temperature
- Predicting monthly sales

### Classification

Classification is used when the target variable is categorical.

Examples:

- Spam or not spam
- Churn or not churn
- Approved or denied
- Dog, cat, or bird

### Focus of This Module

This module focuses on **regression**.

The target variable will be:

```text
SalePrice
```

The model will use information about each house to predict its sale price.

---

## 3. Simple Linear Regression

Simple linear regression models the relationship between one input feature and one numeric target.

The general form is:

$$
\hat{y} = b_0 + b_1x
$$

Where:

| Symbol | Meaning |
|---|---|
| $\hat{y}$ | Predicted value |
| $b_0$ | Intercept |
| $b_1$ | Coefficient or slope |
| $x$ | Input feature |

For a housing dataset, a simple linear regression model might look like this:

$$
\widehat{\text{SalePrice}} = b_0 + b_1(\text{Gr Liv Area})
$$

This means the model is using one feature, above-ground living area, to predict sale price.

### Interpretation

If the coefficient for `Gr Liv Area` is positive, the model has learned that larger homes tend to have higher sale prices.

Example:

```text
For each additional square foot of living area, the model predicts that sale price increases by approximately $X.
```

This does **not** prove causation. It describes a relationship learned from the data.

---

## 4. Multiple Linear Regression

Multiple linear regression uses more than one feature to predict a numeric target.

The general form is:

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2 + b_3x_3 + \dots + b_px_p
$$

Where:

| Symbol | Meaning |
|---|---|
| $\hat{y}$ | Predicted value |
| $b_0$ | Intercept |
| $b_1, b_2, \dots, b_p$ | Coefficients |
| $x_1, x_2, \dots, x_p$ | Input features |

For a housing dataset, a multiple linear regression model might look like:

$$
\widehat{\text{SalePrice}} = b_0 + b_1(\text{Gr Liv Area}) + b_2(\text{Overall Qual}) + b_3(\text{Year Built}) + b_4(\text{Garage Cars})
$$

This model has more information than the simple model, so it may make better predictions.

---

## 5. Regression Evaluation Metrics

After training a model, we need to evaluate how well it performs on data it has not seen before.

This module uses three common regression metrics:

1. Mean Absolute Error
2. Root Mean Squared Error
3. R² Score

---

### Mean Absolute Error

Mean Absolute Error, or MAE, measures the average absolute difference between the actual values and predicted values.

$$
MAE = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:

| Symbol | Meaning |
|---|---|
| $y_i$ | Actual value |
| $\hat{y}_i$ | Predicted value |
| $n$ | Number of observations |

In this lesson, MAE answers:

> On average, how many dollars off are the model's predictions?

MAE is easy to explain because it is in the same unit as the target variable.

If the target is sale price, then MAE is measured in dollars.

---

### Root Mean Squared Error

Root Mean Squared Error, or RMSE, also measures prediction error, but it penalizes large errors more heavily.

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

RMSE is also in the same unit as the target variable.

In this lesson, RMSE answers:

> How large are the model's errors when larger mistakes are punished more heavily?

If RMSE is much larger than MAE, that may suggest the model is making some very large prediction errors.

---

### R² Score

R² measures how much variation in the target variable is explained by the model.

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

Where:

| Symbol | Meaning |
|---|---|
| $y_i$ | Actual value |
| $\hat{y}_i$ | Predicted value |
| $\bar{y}$ | Mean of the actual values |

A higher R² generally means the model explains more of the variation in the target.

However, R² should not be interpreted by itself. Always use it with error metrics like MAE and RMSE.

---

## 6. Load the Dataset

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

```python
housing = pd.read_csv("housing.csv")

housing.head()
```

Inspect the dataset:

```python
housing.shape
```

```python
housing.info()
```

```python
housing.describe()
```

---

## 7. Identify Features and Target

In supervised learning, we usually separate the data into:

| Term | Meaning |
|---|---|
| Features | The input columns used to make predictions |
| Target | The output column we want to predict |

In `scikit-learn`, this is commonly represented as:

```python
X = features
y = target
```

For this lesson:

```python
target = "SalePrice"

y = housing[target]
```

---

## 8. Simple Linear Regression Workflow

For the first model, use only one feature:

```python
simple_feature = ["Gr Liv Area"]

X = housing[simple_feature]
y = housing["SalePrice"]
```

The double brackets matter:

```python
X = housing[["Gr Liv Area"]]
```

This keeps `X` as a DataFrame, which is the format expected by `scikit-learn`.

---

### Visualize the Relationship

```python
sns.scatterplot(
    data=housing,
    x="Gr Liv Area",
    y="SalePrice"
)

plt.title("Living Area vs. Sale Price")
plt.xlabel("Above Ground Living Area")
plt.ylabel("Sale Price")
plt.show()
```

This plot helps students see the relationship the model will try to learn.

---

### Split the Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The model learns from the training data.

The model is evaluated on the test data.

This allows us to ask:

> How well does the model perform on data it has never seen before?

---

### Train the Model

```python
simple_model = LinearRegression()

simple_model.fit(X_train, y_train)
```

---

### Make Predictions

```python
simple_preds = simple_model.predict(X_test)

simple_preds[:10]
```

---

### Evaluate the Simple Model

```python
simple_mae = mean_absolute_error(y_test, simple_preds)
simple_rmse = np.sqrt(mean_squared_error(y_test, simple_preds))
simple_r2 = r2_score(y_test, simple_preds)

print("Simple Linear Regression Results")
print(f"MAE: ${simple_mae:,.0f}")
print(f"RMSE: ${simple_rmse:,.0f}")
print(f"R²: {simple_r2:.3f}")
```

---

### Interpret the Coefficient

```python
print(f"Coefficient: {simple_model.coef_[0]:,.2f}")
print(f"Intercept: {simple_model.intercept_:,.2f}")
```

If the coefficient is positive, then the model learned that larger living area is associated with a higher sale price.

A plain-English interpretation:

```text
For each additional square foot of living area, the model predicts that sale price increases by approximately the coefficient amount.
```

---

## 9. Multiple Linear Regression Workflow

Now use several numeric features.

```python
numeric_features = [
    "Gr Liv Area",
    "Overall Qual",
    "Year Built",
    "Garage Cars"
]

X = housing[numeric_features]
y = housing["SalePrice"]
```

Check missing values:

```python
X.isna().sum()
```

For this beginner lesson, use median imputation for missing numeric values:

```python
X = X.fillna(X.median())
```

Split the data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Train the model:

```python
multiple_model = LinearRegression()

multiple_model.fit(X_train, y_train)
```

Make predictions:

```python
multiple_preds = multiple_model.predict(X_test)
```

Evaluate the model:

```python
multiple_mae = mean_absolute_error(y_test, multiple_preds)
multiple_rmse = np.sqrt(mean_squared_error(y_test, multiple_preds))
multiple_r2 = r2_score(y_test, multiple_preds)

print("Multiple Linear Regression Results")
print(f"MAE: ${multiple_mae:,.0f}")
print(f"RMSE: ${multiple_rmse:,.0f}")
print(f"R²: {multiple_r2:.3f}")
```

---

### Compare Simple and Multiple Regression

```python
model_comparison = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression"
    ],
    "MAE": [
        simple_mae,
        multiple_mae
    ],
    "RMSE": [
        simple_rmse,
        multiple_rmse
    ],
    "R2": [
        simple_r2,
        multiple_r2
    ]
})

model_comparison
```

This gives students a clear comparison between the one-feature model and the multi-feature model.

---

### Interpret Multiple Regression Coefficients

```python
coefficients = pd.DataFrame({
    "Feature": numeric_features,
    "Coefficient": multiple_model.coef_
})

coefficients.sort_values("Coefficient", ascending=False)
```

Coefficient interpretation becomes more complicated with multiple features.

A simplified explanation:

> A coefficient estimates how much the prediction changes when that feature increases by one unit, while the other features are held constant.

Important warning:

> Coefficients are easier to interpret when features are measured on similar scales. In this lesson, focus on the direction of the coefficient more than the exact size.

---

## 10. Handling Categorical Variables

Many useful housing features are categorical.

Examples:

- `Neighborhood`
- `House Style`
- `Exterior Type`
- `Sale Condition`

Machine learning models cannot directly use text categories like `"OldTown"` or `"CollgCr"`.

We need to encode those categories as numbers.

---

### Select Numeric and Categorical Features

```python
numeric_features = [
    "Gr Liv Area",
    "Overall Qual",
    "Year Built",
    "Garage Cars"
]

categorical_features = [
    "Neighborhood",
    "House Style"
]

selected_features = numeric_features + categorical_features

X = housing[selected_features]
y = housing["SalePrice"]
```

---

### Handle Missing Values

```python
X = X.copy()

for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_features:
    X[col] = X[col].fillna("Missing")
```

---

### Encode Categorical Variables

```python
X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True
)

X_encoded.head()
```

`pd.get_dummies()` converts categorical columns into multiple numeric columns.

For example, a column like this:

| Neighborhood |
|---|
| OldTown |
| CollgCr |
| NAmes |

May become several columns like this:

| Neighborhood_CollgCr | Neighborhood_NAmes | Neighborhood_OldTown |
|---:|---:|---:|
| 0 | 0 | 1 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |

Each new column contains a `0` or `1`.

---

### Train a Model with Encoded Features

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

encoded_model = LinearRegression()

encoded_model.fit(X_train, y_train)

encoded_preds = encoded_model.predict(X_test)
```

---

### Evaluate the Encoded Model

```python
encoded_mae = mean_absolute_error(y_test, encoded_preds)
encoded_rmse = np.sqrt(mean_squared_error(y_test, encoded_preds))
encoded_r2 = r2_score(y_test, encoded_preds)

print("Linear Regression with Encoded Categorical Variables")
print(f"MAE: ${encoded_mae:,.0f}")
print(f"RMSE: ${encoded_rmse:,.0f}")
print(f"R²: {encoded_r2:.3f}")
```

---

## 11. Final Model Comparison

```python
final_comparison = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Multiple Regression + Categorical Encoding"
    ],
    "Features Used": [
        "1 numeric feature",
        "Multiple numeric features",
        "Numeric + categorical features"
    ],
    "MAE": [
        simple_mae,
        multiple_mae,
        encoded_mae
    ],
    "RMSE": [
        simple_rmse,
        multiple_rmse,
        encoded_rmse
    ],
    "R2": [
        simple_r2,
        multiple_r2,
        encoded_r2
    ]
})

final_comparison
```

Format the table:

```python
final_comparison_formatted = final_comparison.copy()

final_comparison_formatted["MAE"] = final_comparison_formatted["MAE"].map("${:,.0f}".format)
final_comparison_formatted["RMSE"] = final_comparison_formatted["RMSE"].map("${:,.0f}".format)
final_comparison_formatted["R2"] = final_comparison_formatted["R2"].map("{:.3f}".format)

final_comparison_formatted
```

---

## 12. Error Analysis

Model evaluation should not stop with one metric.

Create a DataFrame comparing actual and predicted values:

```python
error_analysis = pd.DataFrame({
    "Actual": y_test,
    "Predicted": encoded_preds
})

error_analysis["Error"] = error_analysis["Actual"] - error_analysis["Predicted"]
error_analysis["Absolute Error"] = error_analysis["Error"].abs()

error_analysis.head()
```

Look at the largest errors:

```python
error_analysis.sort_values("Absolute Error", ascending=False).head(10)
```

This helps students ask:

> Where is the model struggling?

---

### Plot Actual vs. Predicted Values

```python
sns.scatterplot(
    data=error_analysis,
    x="Actual",
    y="Predicted"
)

plt.title("Actual vs. Predicted Sale Prices")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.show()
```

---

### Plot Prediction Errors

```python
sns.histplot(error_analysis["Error"], kde=True)

plt.title("Distribution of Prediction Errors")
plt.xlabel("Actual - Predicted")
plt.ylabel("Count")
plt.show()
```

Interpretation:

- Errors near `0` are better.
- Positive errors mean the model predicted too low.
- Negative errors mean the model predicted too high.
- Large tails suggest the model has some very large prediction errors.

---

## 13. Communicating Model Results

A good machine learning summary should include:

1. What the model predicts
2. What features were used
3. How performance was measured
4. How well the model performed
5. Where the model may struggle
6. What the model should not be used for

Example:

```text
This model predicts home sale prices using numeric and categorical housing features.

The best model used living area, overall quality, year built, garage capacity, neighborhood, and house style.

The model had a Mean Absolute Error of approximately $XX,XXX, meaning its predictions were off by about that many dollars on average.

The RMSE was higher than the MAE, suggesting that some predictions had larger errors.

The model is useful as a first baseline, but it may struggle with unusual homes, luxury homes, or homes with important features not included in the dataset.
```

---

## 14. Optional Extension: Scikit-Learn Pipeline

The code above uses `pd.get_dummies()` because it is beginner-friendly and makes encoding visible.

In professional projects, you will often use a `Pipeline` and `ColumnTransformer` to keep preprocessing and modeling steps together.

This is optional for this module, but it is useful as a preview.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
```

```python
numeric_features = [
    "Gr Liv Area",
    "Overall Qual",
    "Year Built",
    "Garage Cars"
]

categorical_features = [
    "Neighborhood",
    "House Style"
]

selected_features = numeric_features + categorical_features

X = housing[selected_features]
y = housing["SalePrice"]
```

```python
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

pipeline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline_model.fit(X_train, y_train)

pipeline_preds = pipeline_model.predict(X_test)
```

```python
pipeline_mae = mean_absolute_error(y_test, pipeline_preds)
pipeline_rmse = np.sqrt(mean_squared_error(y_test, pipeline_preds))
pipeline_r2 = r2_score(y_test, pipeline_preds)

print("Pipeline Model Results")
print(f"MAE: ${pipeline_mae:,.0f}")
print(f"RMSE: ${pipeline_rmse:,.0f}")
print(f"R²: {pipeline_r2:.3f}")
```

### Teaching Note

Do not lead with this section for beginners.

First, show students what encoding does with `pd.get_dummies()`.

Then explain that `Pipeline` is a cleaner way to package preprocessing and modeling together.

---

## Wrap-Up

In this module, students learned the core machine learning workflow using linear regression.

They built:

1. A simple linear regression model
2. A multiple linear regression model
3. A model that includes encoded categorical variables

They also evaluated each model using:

- MAE
- RMSE
- R²

The most important takeaway is that machine learning is a workflow, not just a model.

Students should leave this module understanding that model building includes:

- Choosing features and a target
- Splitting data properly
- Training only on the training set
- Evaluating on unseen data
- Comparing models
- Communicating limitations clearly

---

## Reflection Questions

1. What is the difference between a feature and a target variable?
2. Why do we split data into training and test sets?
3. What does MAE tell us in the context of housing prices?
4. Why might RMSE be higher than MAE?
5. What does R² tell us about a model?
6. Why did the multiple regression model likely perform better than the simple regression model?
7. Why do categorical variables need to be encoded before modeling?
8. What are some limitations of this housing price model?
9. Would you trust this model to price every home? Why or why not?
10. What would you try next to improve the model?

---

## Lab Preview

For the lab, students should use the same housing dataset or a similar one to practice the full workflow independently.

Suggested lab tasks:

1. Load the housing dataset
2. Select a target variable
3. Build a simple linear regression model
4. Evaluate the simple model
5. Build a multiple linear regression model
6. Evaluate the multiple model
7. Add one or two categorical variables
8. Encode categorical variables
9. Train and evaluate the final model
10. Write a short explanation of the model's performance and limitations

---

## Key Takeaways

- Regression predicts numeric values.
- Linear regression learns a relationship between features and a numeric target.
- Simple linear regression uses one feature.
- Multiple linear regression uses multiple features.
- Categorical variables must be encoded before modeling.
- Train/test split helps estimate performance on unseen data.
- MAE and RMSE measure prediction error.
- R² measures how much variation the model explains.
- Model evaluation should include both metrics and plain-English interpretation.
- A simple, understandable baseline model is often the best place to start.
