import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from logging import Logger
from housingpackage.logger import configure_logger

model_names = ["forest_model", "grid_search_model", "lin_model", "tree_model"]


def get_path():
    """Function to get current working directory"""
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "housingproject":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def parse_args():
    """Function to get command line arguments"""
    parser = argparse.ArgumentParser()
    print("Function enter")
    print(parser)
    parser.add_argument(
        "--datapath",
        help="path to the datasets ",
        type=str,
        default="data/processed",
    )
    print("Second step")
    parser.add_argument(
        "--modelpath",
        help="path to the model files ",
        type=str,
        default="artifacts",
    )
    parser.add_argument("--log-level", type=str, default="DEBUG")
    print("log print")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument(
        "--log-path", type=str, default=get_path() + "logs/logs.log"
    )
    print("second print")
    return parser.parse_args()


def scoring(X_test, y_test, lin_reg, tree_reg, forest_reg, grid_search):
    """scoring function applied on the test features and test labels using the models generated"""
    lin_predictions = lin_reg.predict(X_test)
    lin_mse = mean_squared_error(y_test, lin_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(y_test, lin_predictions)

    tree_predictions = tree_reg.predict(X_test)
    tree_mse = mean_squared_error(y_test, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(y_test, tree_predictions)

    forest_predictions = forest_reg.predict(X_test)
    forest_mse = mean_squared_error(y_test, forest_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_mae = mean_absolute_error(y_test, forest_predictions)

    grid_search_predictions = grid_search.predict(X_test)
    grid_search_mse = mean_squared_error(y_test, grid_search_predictions)
    grid_search_rmse = np.sqrt(grid_search_mse)
    grid_search_mae = mean_absolute_error(y_test, grid_search_predictions)

    lin_scores = [lin_mae, lin_mse, lin_rmse]
    tree_scores = [tree_mae, tree_mse, tree_rmse]
    forest_scores = [forest_mae, forest_mse, forest_rmse]
    grid_search_scores = [grid_search_mae, grid_search_mse, grid_search_rmse]

    return lin_scores, tree_scores, forest_scores, grid_search_scores


def load_data(in_path):
    """Function to load the test data"""
    prepared = pd.read_csv(in_path + "/test_X.csv")
    lables = pd.read_csv(in_path + "/test_y.csv")
    lables = lables.values.ravel()
    return prepared, lables


def load_models(model_path):
    """Function to load the models"""
    models = []
    for i in model_names:
        with open(model_path + "/models/" + i + ".pkl", "rb") as f:
            models.append(pickle.load(f))
            print(models)
    return models


def score(models, X_test, y_test):
    """scoring function"""
    lin_scores, tree_scores, forest_scores, grid_search_scores = scoring(
        X_test, y_test, models[0], models[1], models[2], models[3]
    )

    return [lin_scores, tree_scores, forest_scores, grid_search_scores]


if __name__ == "__main__":
    args = parse_args()
    print("Function done")
    logger = configure_logger(
        log_level=args.log_level,
        log_file=args.log_path,
        console=not args.no_console_log,
    )
    path_parent = get_path()
    data_path = path_parent + args.datapath
    model_path = path_parent + args.modelpath

    print(model_path)
    X_test, y_test = load_data(data_path)
    logger.debug("Loaded test data")
    print("load test data")
    models = load_models(model_path)
    logger.debug("Loaded Models")
    scores = []
    scores = score(models, X_test, y_test)
    for i in range(len(models)):
        logger.debug(f"{model_names[i]}={scores[i]}")
