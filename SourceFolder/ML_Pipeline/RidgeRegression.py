import numpy as np
import pandas as pd
from ML_Pipeline.DataPreparation import data_preprocessing
from ML_Pipeline.metrics import mean_squared_error, r2_score

class RidgeRegression:
    def __init__(self, alpha=1, lr=0.01, n_iter=1000, csv_path=None):
        # Initialize hyperparameters
        self.alpha = alpha  # Regularization strength
        self.lr = lr        # Learning rate
        self.n_iter = n_iter  # Number of iterations for training
        self.csv_path = csv_path  # Path to the dataset
        self.weights = None   # Model weights
        self.bias = None      # Model bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize model parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # Make predictions
            y_pred = self.predict(X)

            # Compute gradients
            dW = (-(2 * np.dot(X.T, (y - y_pred))) + (2 * self.alpha * self.weights)) / n_samples
            db = -2 * np.sum(y_pred - y) / n_samples

            # Update model parameters
            self.weights -= self.lr * dW
            self.bias -= self.lr * db

    def predict(self, X):
        # Make predictions using the trained model
        return np.dot(X, self.weights) + self.bias

    def RR_main(self):
        # Splitting data into train and test
        X_train, X_test, y_train, y_test = data_preprocessing(self.csv_path)

        # Fit the model and make predictions
        self.fit(X_train, y_train)
        ridge_predict = self.predict(X_test)

        # Compute and display metrics
        print("MSE of Ridge Model : ", mean_squared_error(y_test, ridge_predict))
        print("R2 Score of Ridge Model : ", r2_score(y_test, ridge_predict))

if __name__ == '__main__':
    csv_path = '../../InputFiles/EPL_Soccer_MLR_LR.csv'

    # Set hyperparameters (learning rate and number of iterations)
    ridge_model = RidgeRegression(alpha=0.03, lr=.00001, n_iter=100, csv_path=csv_path)
    ridge_model.RR_main()
