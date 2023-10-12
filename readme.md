# Regression Models with NumPy

## Introduction
Regression is a type of supervised learning algorithm used to understand the relationship between one or more independent variables and a dependent variable. In this project, we will build regression models from scratch using NumPy, providing flexibility and control over the training process.

## Data Description
The dataset contains information about sports players and aims to predict their scores. It comprises around 200 rows and 13 columns.

## Objective
Our goal is to build multiple regression models using the NumPy module.

## Tech Stack
- Language: Python
- Libraries: Pandas, NumPy

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

## Modular Code Overview
The project's modular code is organized as follows:
1. Input folder: Contains the dataset files.
2. Src folder: Contains modularized code for pipeline - data processing, model building, and validation.
3. Output folder: Stores the trained models for future use.
4. Lib folder: Includes reference materials, such as notebooks and presentations.

---


```
- InputFiles
  |__ EPL_Soccer_MLR_LR.csv
  
- SourceFolder
  |__ Engine.py
  |__ ML_Pipeline
      |__ DataPreparation.py
      |__ _metrics.py
      |__ _label_encoding.py
      |__ _LinearRegression.py
      |__ _LassoRegression.py
      |__ _RidgeRegression.py
      |__ _RegressionTree.py
  
  |__README.md
  
  |__requirements.txt
  
- Lib
  |__ Data_Exploration.ipynb
  |__ Regression_Models.ipynb
  |__ Regression_Tree_numpy.ipynb

- Output
  |__ linear_model.pkl
  |__ lasso_model.pkl
  |__ ridge_model.pkl
  |__ Reg_tree_model.pkl

---
```


## Concepts Explored
Throughout the project, we explored various concepts and techniques, including

- What regression is and its applications.
- Types of regression.
- Linear regression, loss functions, and gradient descent.
- Ridge and Lasso regression.
- Decision trees and their terminologies.
- Model evaluation using MSE and R-squared.
