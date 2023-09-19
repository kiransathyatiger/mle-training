import housingpackage.score as score

import unittest
import os

arguments = score.parse_args()
path = score.get_path()


class scoringtest(unittest.TestCase):
    def test_parse_arguments(self):
        self.assertTrue(arguments.datapath == "data/processed")
        self.assertTrue(arguments.modelpath == "artifacts")
        self.assertTrue(arguments.log_level == "DEBUG")
        self.assertFalse(arguments.no_console_log)
        self.assertTrue(arguments.log_path == path + "logs/logs.log")

    def test_load_data(self):
        test_X, test_y = score.load_data(path + arguments.datapath)
        self.assertTrue(len(test_X) == len(test_y))
        self.assertTrue(len(test_y.shape) == 1)

    def test_load_models(self):
        models = score.load_models(path + arguments.modelpath)
        self.assertTrue(len(models) == 4)


if __name__ == "__main__":
    unittest.main()
