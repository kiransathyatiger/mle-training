from housingpackage import ingest_data as data
import unittest
import os
import pandas as pd

arguments = data.parse_args()
print(arguments)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data/raw", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
path = data.get_path()
print(path)
print(arguments.datapath)


class Testutils(unittest.TestCase):
    def test_parse_args(self):
        self.assertTrue(arguments.datapath == "data/raw/housing")
        self.assertTrue(arguments.dataprocessed == "data/processed")
        self.assertTrue(arguments.log_level == "DEBUG")
        self.assertFalse(arguments.no_console_log)
        self.assertTrue(arguments.log_path == path + "logs/logs.log")

    def test_fetch_data(self):
        print("getting url")
        data.fetch_housing_data(
            housing_url=HOUSING_URL, housing_path=HOUSING_PATH
        )
        print("yes fetching data")

    def test_split(self):
        housing_df = pd.read_csv(f"{path}{arguments.datapath}/housing.csv")
        train_set, test_set = data.train_test(housing_df)
        self.assertFalse(train_set.isna().sum().sum() == 0)
        self.assertFalse(test_set.isna().sum().sum() == 0)
        self.assertTrue(len(train_set) == len(housing_df) * 0.8)
        self.assertTrue(len(test_set) == len(housing_df) * 0.2)


if __name__ == "__main__":
    unittest.main()
