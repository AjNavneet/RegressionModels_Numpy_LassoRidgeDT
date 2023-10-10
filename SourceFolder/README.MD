# Regression

Regression is a supervised learning algorithm that establishes the relationship between one or more independent variables and a dependent variable. In the regression model, the dependent variable has a continuous numerical value.

## Understanding Project Structure

```plaintext
Regression_Module
|-- InputFiles
    -- EPL_Soccer_MLR_LR
|-- SourceFolder
    |-- ML_Pipeline
        -- DataPreparation.py
        -- metrics.py
        -- LinearRegression.py
        -- LassoRegression.py
        -- RidgeRegression.py
        -- RegressionTree.py
    |-- Engine.py
|-- README.md
|-- requirements.txt
```

## Prerequisites

- Python 3.x

## Installation

To set up the project:

1. Activate Virtual Environment:

   First, install Virtual Environment for Python 3 using this command in your terminal:

   ```shell
   pip install virtualenv
   ```

   In the current working directory (Regression_Module), activate the virtual environment:

   ```shell
   source task_env/bin/activate
   ```

2. Install Dependencies:

   Run the following command in the command line to install the necessary dependencies:

   ```shell
   pip install -r requirements.txt
   ```
