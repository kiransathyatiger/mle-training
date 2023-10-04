import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from logging import Logger
from housingpackage.logger import configure_logger
import pytest
import logging
import os
import pickle as pkl
import sys

import numpy as np
import pandas as pd
import pytest


from housingpackage import score, train

linregmodel = "/home/kiran/housingproject/artifacts/models/lin_model.pkl"
linregmodel = pkl.load(open(linregmodel, "rb"))
forestmodel = "/home/kiran/housingproject/artifacts/models/forest_model.pkl"
forestmodel = pkl.load(open(forestmodel, "rb"))
gridsearchmodel = (
    "/home/kiran/housingproject/artifacts/models/grid_search_model.pkl"
)
gridsearchmodel = pkl.load(open(gridsearchmodel, "rb"))
treemodel = "/home/kiran/housingproject/artifacts/models/tree_model.pkl"
treemodel = pkl.load(open(treemodel, "rb"))


def load_data(in_path):
    """Function to load the test data"""
    prepared = pd.read_csv(in_path + "/test_X.csv")
    lables = pd.read_csv(in_path + "/test_y.csv")
    lables = lables.values.ravel()
    return prepared, lables


X, y = load_data("/home/kiran/housingproject/data/processed")
predictions = treemodel.predict(X.iloc[0, :].values.reshape(1, -1))
predictions = np.array(predictions)
print(predictions)


def test_linearmodelpredictions():
    """Function to test working of linear regression model"""
    X, y = load_data("/home/kiran/housingproject/data/processed")
    predictions = linregmodel.predict(X.iloc[0, :].values.reshape(1, -1))
    predictions = np.array(predictions)
    assert round(predictions[0]) == 424327


def test_forestmodelpredictions():
    """Function to test working of linear regression model"""
    X, y = load_data("/home/kiran/housingproject/data/processed")
    predictions = forestmodel.predict(X.iloc[0, :].values.reshape(1, -1))
    predictions = np.array(predictions)
    assert round(predictions[0]) == 487657


def test_gridmodelpredictions():
    """Function to test working of linear regression model"""
    X, y = load_data("/home/kiran/housingproject/data/processed")
    predictions = gridsearchmodel.predict(X.iloc[0, :].values.reshape(1, -1))
    predictions = np.array(predictions)
    assert round(predictions[0]) == 491164


def test_treemodelpredictions():
    """Function to test working of linear regression model"""
    X, y = load_data("/home/kiran/housingproject/data/processed")
    predictions = treemodel.predict(X.iloc[0, :].values.reshape(1, -1))
    predictions = np.array(predictions)
    assert round(predictions[0]) == 394900
