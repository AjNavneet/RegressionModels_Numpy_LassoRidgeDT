from ML_Pipeline.LinearRegression import LinearRegression
from ML_Pipeline.LassoRegression import LassoRegression
from ML_Pipeline.RidgeRegression import RidgeRegression
from ML_Pipeline.RegressionTree import RegressionTree
from ML_Pipeline.DataPreparation import data_preprocessing
import pickle as pkl

class EngineClass:
    def __init__(self, csv_path, model_save_path):
        self.csv_path = csv_path
        self.model_save_path = model_save_path

    def main(self, type_="linear"):
        if type_ == "linear":
            # Create and train a Linear Regression model
            linear_model = LinearRegression(lr=.00001, n_iter=100, csv_path=self.csv_path)
            linear_model.LR_main()
            # Saving the model in the Output Folder
            pkl.dump(linear_model, open(self.model_save_path + "linear_model.pkl", "wb"))

        elif type_ == "lasso":
            # Create and train a Lasso Regression model
            lasso_model = LassoRegression(alpha=0.03, lr=.00001, n_iter=100, csv_path=self.csv_path)
            lasso_model.LR_main()
            # Saving the model in the Output Folder
            pkl.dump(lasso_model, open(self.model_save_path + "lasso_model.pkl", "wb"))

        elif type_ == "ridge":
            # Create and train a Ridge Regression model
            ridge_model = RidgeRegression(alpha=0.03, lr=.00001, n_iter=100, csv_path=self.csv_path)
            ridge_model.RR_main()
            # Saving the model in the Output Folder
            pkl.dump(ridge_model, open(self.model_save_path + "ridge_model.pkl", "wb"))

        elif type_ == "regression_tree":
            # Perform data preprocessing and create a Regression Tree model
            X_train, X_test, y_train, y_test = data_preprocessing(csv_path)
            root_node = RegressionTree(X_train, y_train, maxm_depth=2, min_split=3)
            root_node.RT_main(self.csv_path)
            # Saving the model in the Output Folder
            pkl.dump(root_node, open(self.model_save_path + "Reg_tree.pkl", "wb"))


if __name__ == '__main__':
    csv_path = '../InputFiles/EPL_Soccer_MLR_LR.csv'
    model_save_path = "../OutputFolder/"
    eng_obj = EngineClass(csv_path, model_save_path)
    eng_obj.main("regression_tree")  # Choose the type of model you want to build
