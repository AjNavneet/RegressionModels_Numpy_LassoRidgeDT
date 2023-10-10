import numpy as np
import pandas as pd
from ML_Pipeline.DataPreparation import data_preprocessing
from ML_Pipeline.metrics import mean_squared_error, r2_score

class LinearRegression:
    def __init__(self, lr=0.01, n_iter=1000, csv_path=None):
        # Initialize hyperparameters and paths
        self.lr = lr  # Learning rate
        self.n_iter = n_iter  # Number of iterations
        self.csv_path = csv_path  # Path to CSV file
        self.weights = None  # Model weights
        self.bias = None  # Model bias

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize model parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # Make predictions
            y_pred = self.predict(X)

            # Compute gradients for weights and bias
            dW = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples

            # Update parameters using gradient descent
            self.weights -= self.lr * dW
            self.bias -= self.lr * db

    def predict(self, X):
        # Make predictions
        return np.dot(X, self.weights) + self.bias

    def LR_main(self):
        # Preprocess data and split into train and test sets
        X_train, X_test, y_train, y_test = data_preprocessing(self.csv_path)

        # Fit the linear regression model
        self.fit(X_train, y_train)

        # Make predictions on the test set
        linear_predict = self.predict(X_test)

        # Evaluate the model using mean squared error and R-squared
        print("MSE of Linear Model : ", mean_squared_error(y_test, linear_predict))
        print("R2 Score of Linear Model : ", r2_score(y_test, linear_predict))

if __name__ == '__main__':
    csv_path = '../../InputFiles/EPL_Soccer_MLR_LR.csv'

    # Create and train the linear regression model
    linear_model = LinearRegression(lr=0.00001, n_iter=100, csv_path=csv_path)
    linear_model.LR_main()
