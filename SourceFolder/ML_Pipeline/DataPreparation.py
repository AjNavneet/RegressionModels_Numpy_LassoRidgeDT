import numpy as np
import pandas as pd

def data_shuffler(X, y, seed=None):
    # Randomly shuffle data in X and y
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    try:
        return X[idx], y[idx]
    except:
        return X.iloc[idx], y.iloc[idx]

def Train_Test_Split(X, y, train_size=0.5, seed=None, shuffle=True):
    # Splitting the X and y into train and test sets
    if shuffle:
        X, y = data_shuffler(X, y, seed)
    idx_split = len(y) - int(len(y) // (1 / (1 - train_size)))
    X_train, X_test = X[:idx_split], X[idx_split:]
    y_train, y_test = y[:idx_split], y[idx_split:]
    return X_train, X_test, y_train, y_test

def data_preprocessing(csv_path=None):
    X_train, X_test, y_train, y_test = None, None, None, None
    if csv_path:
        # Reading CSV as a pandas dataframe
        df = pd.read_csv(csv_path)

        # Dropping null values and removing categorical columns
        df.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)
        df = df.select_dtypes(['number'])

        # Finding correlated features
        X = df.iloc[:, :-1]  # Independent features
        y = df.iloc[:, -1]  # Dependent feature

        print("Original Shape of X : ", X.shape)

        correlated_features = set()
        correlation_matrix = X.corr()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                # Finding positively or negatively correlated
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)

        print("Correlated Features : ", correlated_features)

        # Dropping the correlated features from X
        X.drop(columns=correlated_features, axis=1, inplace=True)

        print("Shape of X after dropping correlated features : ", X.shape)

        X_train, X_test, y_train, y_test = Train_Test_Split(X, y, seed=42, train_size=0.8)

    return X_train, X_test, y_train, y_test
