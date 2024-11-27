"""
test entropy rate computation
"""
import os
import unittest
from preprocessing.paths import pointwise_features_path, jarLocation
from scripts.algorithms.infotheory import *
import jpype


# initialize jvm for test case
try:
    jpype.startJVM(
        jpype.getDefaultJVMPath(), "-ea",
        "-Djava.class.path=" + jarLocation, convertStrings=True)
except Exception as e:
    initialization_error = e


class TestEntropyRate(unittest.TestCase):
    """
    test entropy rate calculation function
    """
    def setUp(self):
        """
        initialize jvm, set as class property
        """
        assert os.path.isfile(jarLocation)
        if not jpype.isJVMStarted():
            self.skipTest(f"JVM failed to start")


if __name__ == '__main__':
    unittest.main()
