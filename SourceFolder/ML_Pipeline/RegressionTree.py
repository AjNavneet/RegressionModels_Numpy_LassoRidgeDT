import numpy as np
import pandas as pd
from ML_Pipeline.DataPreparation import data_preprocessing

class RegressionTree:
    def __init__(self, X=None,
                 y=None,
                 min_split=None,
                 maxm_depth=None,
                 depth=None,
                 node_type=None,
                 rule=None):

        self.X = X
        self.y = y

        # Minimum samples required for node splitting
        self.min_split = min_split if min_split else 10
        # Maximum depth of the tree
        self.maxm_depth = maxm_depth if maxm_depth else 5
        # Current depth of the node
        self.depth = depth if depth else 0
        # Type of node (e.g., root, left_node, right_node)
        self.node_type = node_type if node_type else "root"
        # Features in the dataset
        self.features = self.X.columns
        # Rule for splitting nodes (e.g., which feature is used for splitting)
        self.rule = rule if rule else ""

        # Initializing left and right child nodes
        self.left = None
        self.right = None

        # Best splitting attributes
        self.best_feature = None
        self.best_value = None

        # Number of samples
        self.n_samples = len(self.y)

        # Calculating mean of y
        self.y_mean = np.mean(self.y)

        # Calculating the Mean Squared Error (MSE) of the node
        self.mse = self.mse_calculator(self.y, self.y_mean)

        # Calculating the residuals of the node
        self.residual = self.y - self.y_mean

    def mse_calculator(self, y_true, y_pred):
        # Calculate the Mean Squared Error (MSE)
        return np.mean((y_true - y_pred) ** 2)

    def moving_avg_calculator(self, val_array, consecutive_nos):
        # Calculate the moving average of a list of values
        return np.convolve(val_array, np.ones(consecutive_nos), "valid") / consecutive_nos

    def best_split_calculator(self):
        best_feature, best_value = None, None
        df = self.X.copy()
        df["y"] = self.y

        impurity = self.mse

        for feature in self.features:
            X_df = df.dropna().sort_values(feature)
            x_mean = self.moving_avg_calculator(X_df[feature].unique(), 2)

            for value in x_mean:
                left_tree_y = X_df[X_df[feature] < value]["y"].values
                right_tree_y = X_df[X_df[feature] >= value]["y"].values

                left_tree_mean = np.mean(left_tree_y)
                right_tree_mean = np.mean(right_tree_y)
                left_tree_residual = left_tree_y - left_tree_mean
                right_tree_residual = right_tree_y - right_tree_mean

                total_residual = np.concatenate((left_tree_residual, right_tree_residual), axis=None)

                impurity_split = np.mean(total_residual ** 2)

                if (impurity_split < impurity):
                    best_feature = feature
                    best_value = value
                    impurity = impurity_split

        return best_feature, best_value

    def grow_tree_recursive(self):
        df = self.X.copy()
        df["y"] = self.y

        if (self.depth < self.maxm_depth and self.n_samples >= self.min_split):
            current_best_feature, current_best_val = self.best_split_calculator()

            if current_best_feature is not None:
                self.best_feature = current_best_feature
                self.best_value = current_best_val

                left_subtree_df = df[df[current_best_feature] < current_best_val].copy()
                right_subtree_df = df[df[current_best_feature] >= current_best_val].copy()

                left_node = RegressionTree(
                    left_subtree_df[self.features],
                    left_subtree_df["y"].values.tolist(),
                    depth=self.depth + 1,
                    maxm_depth=self.maxm_depth,
                    min_split=self.min_split,
                    node_type="left_node",
                    rule=f"{self.best_feature} <= {round(self.best_value, 4)}"
                )

                self.left = left_node
                self.left.grow_tree_recursive()

                right_node = RegressionTree(
                    right_subtree_df[self.features],
                    right_subtree_df["y"].values.tolist(),
                    depth=self.depth + 1,
                    maxm_depth=self.maxm_depth,
                    min_split=self.min_split,
                    node_type="right_node",
                    rule=f"{self.best_feature} > {round(self.best_value, 4)}"
                )

                self.right = right_node
                self.right.grow_tree_recursive()

    def display_tree_info(self, tree_width=4):
        constant = int(self.depth * tree_width ** 1.5)
        total_spaces = "-" * constant

        if self.node_type.lower() == "root":
            print("ROOT")
        else:
            print(f"|{total_spaces} Rule for splitting :  {self.rule}")

        print(f"{'' * constant} | MSE of the current node : {round(self.mse, 3)}")
        print(f"{'' * constant} | No. of observation in the node : {self.n_samples}")
        print(f"{'' * constant} | Node Prediction : {round(self.y_mean, 3)}")

    def display_tree(self):
        self.display_tree_info()

        if self.left is not None:
            self.left.display_tree()

        if self.right is not None:
            self.right.display_tree()

    def RT_main(self, csv_path=None):
        if (csv_path):
            X_train, X_test, y_train, y_test = data_preprocessing(csv_path)
            root_node = RegressionTree(X_train, y_train, maxm_depth=2, min_split=3)

        self.grow_tree_recursive()
        self.display_tree()


if __name__ == '__main__':
    csv_path = '../../InputFiles/EPL_Soccer_MLR_LR.csv'

    X_train, X_test, y_train, y_test = data_preprocessing(csv_path)
    root_node = RegressionTree(X_train, y_train, maxm_depth=2, min_split=3)
    root_node.RT_main()
