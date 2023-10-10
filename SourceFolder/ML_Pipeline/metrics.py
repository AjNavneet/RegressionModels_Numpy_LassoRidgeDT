import numpy as np

# Mean Squared Error (MSE) measures the average of the squares of the errors between predicted and actual values.
# MSE = (1/n) * Σ(y_true - y_pred) ** 2

def mean_squared_error(y_true, y_pred):
    # Calculate the mean squared error
    return np.mean((y_true - y_pred) ** 2)

# R-squared (R2) Score measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
# It can be calculated by squaring the coefficient of correlation between y_true and y_pred.
# The coefficient of correlation is obtained from the correlation matrix.

def r2_score(y_true, y_pred):
    # Calculate the correlation matrix between y_true and y_pred
    corr_matrix = np.corrcoef(y_true, y_pred)
    
    # Get the correlation coefficient (corr) from the matrix and square it to obtain R2 score
    corr = corr_matrix[0, 1]
    return corr ** 2
