from housingpackage.ingest_data import (
    parse_args,
    fetch_housing_data,
    load_housing_data,
    train_test,
)
import os
import pandas as pd

import sys
import pandas as pd
import pickle
import pytest

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data/raw", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

data = load_housing_data(housing_path=HOUSING_PATH)


@pytest.fixture()
def checkdataload():
    """ Function to check if raw file exits"""
    housingdata = pd.read_csv("data/raw/housing/housing.csv")
    return pd.DataFrame(housingdata)


def test_loaded_data(checkdataload):
    assert len(checkdataload) == len(data)


traindata, testdata = train_test(data)


@pytest.fixture()
def checktraindataload():
    """ Function to check if train data is loaded """
    housingdatatrain = pd.read_csv(
        "/home/kiran/housingproject/data/processed/train_X.csv"
    )
    return pd.DataFrame(housingdatatrain)


def test_loaded_traindata(checktraindataload):
    assert len(traindata) == len(checktraindataload)


@pytest.fixture()
def checktestdataload():
    """ Function to check if test data is loaded"""
    housingdatatest = pd.read_csv(
        "/home/kiran/housingproject/data/processed/test_X.csv"
    )
    return pd.DataFrame(housingdatatest)


def test_loaded_testdata(checktestdataload):
    assert len(testdata) == len(checktestdataload)


@pytest.fixture()
def checktestdataload():
    housingdatatest = pd.read_csv(
        "/home/kiran/housingproject/data/processed/test_X.csv"
    )
    return pd.DataFrame(housingdatatest)


def test_loaded_testdata(checktestdataload):
    assert len(testdata) == len(checktestdataload)


def modeload():
    model_names = [
        "forest_model",
        "grid_search_model",
        "lin_model",
        "tree_model",
    ]
    model_path = "/home/kiran/housingproject/artifacts"
    models = []
    for i in model_names:
        with open(model_path + "/models/" + i + ".pkl", "rb") as f:
            models.append(pickle.load(f))
    return models


models = modeload()


def test_model_load():
    """Function to check if all model files are loaded"""
    assert len(models) == 4
