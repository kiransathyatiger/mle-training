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
gridsearchmodel = "/home/kiran/housingproject/artifacts/models/grid_search_model.pkl"
gridsearchmodel = pkl.load(open(gridsearchmodel, "rb"))
treemodel = "/home/kiran/housingproject/artifacts/models/tree_model.pkl"
treemodel = pkl.load(open(treemodel, "rb"))


def dummydata():
    """
    Random data constructor utility for tests
    """
    X = [[12, 14, 13, 20, 45, 50, 60, 70, 21, 12, 41, 4, 34, 14, 2]]
    y = 2022
    return X, y


X, y = dummydata()
X = np.array(X).reshape(1, -1)
y = np.array(y).reshape(1, -1)


X, y = dummydata()
predictions = linregmodel.predict(X)


def test_linearmodelpredictions():
    """Function to test working of linear regression model """
    assert predictions > 0


predictions = forestmodel.predict(X)


def test_forestmodelpredictions():
    """Function to test working of random forest model """
    assert predictions > 0

predictions = treemodel.predict(X)


def test_treemodelpredictions():
    """Function to test working of random forest model """
    assert predictions > 0


predictions=gridsearchmodel.predict(X)
def test_gridsearchpredictions():
    """Function to test working of gridsearch model """
    assert predictions > 0
