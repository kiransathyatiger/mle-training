from housingpackage import train as traindata
import unittest
import os

arguments = traindata.parse_args()
path = traindata.get_path()
print(arguments)
print(path)


class trainingtest(unittest.TestCase):
    def test_parse_args(self):
        self.assertTrue(arguments.inputpath == "data/processed/")
        self.assertTrue(arguments.outputpath == "artifacts")
        self.assertTrue(arguments.log_level == "DEBUG")
        self.assertFalse(arguments.no_console_log)
        self.assertTrue(arguments.log_path == path + "logs/logs.log")

    def test_load_data(self):
        train_X, train_y = traindata.load_data(path + arguments.inputpath)
        self.assertTrue(len(train_X) == len(train_y))
        self.assertTrue(len(train_y.shape) == 1)

    def test_save_model(self):
        models = traindata.model_names
        for i in models:
            self.assertFalse(
                os.path.isdir(f"{path}{arguments.outputpath}/{i}")
            )


if __name__ == "__main__":
    unittest.main()
