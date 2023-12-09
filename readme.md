# Ridge and Lasso Regression Models with NumPy

## Objective

In this project, we will build regression models from scratch using NumPy on sports players, providing flexibility and control over the training process.

---

## Data Description
The dataset contains information about sports players and aims to predict their scores. It comprises around 200 rows and 13 columns.

---

## Tech Stack
- Language: Python
- Libraries: Pandas, NumPy

---

## Approach
1. Import required libraries and read the dataset.
2. Data Pre-processing:
   - Remove missing data points.
   - Drop categorical variables.
   - Check for multicollinearity and remove highly correlated features.
3. Create train and test data by random shuffling.
4. Perform train-test split.
5. Model Building using NumPy:
   - Linear Regression Model
   - Ridge Regression
   - Lasso Regressor
   - Decision Tree Regressor
6. Model Validation:
   - Mean Absolute Error
   - R-squared
  
---

## Modular Code Overview

1. Input folder: Contains the dataset files.
2. Src folder: Contains modularized code for pipeline - data processing, model building, and validation.
3. Output folder: Stores the trained models for future use.
4. Lib folder: Includes reference materials, such as notebooks and presentations.

---
