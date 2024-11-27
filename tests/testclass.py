"""
data file paths, modules that are commonly used in the test cases
"""
import os
import sys
import unittest
import pandas as pd
scripts_dir = os.path.join(__file__, '..', '..')
sys.path.insert(1, scripts_dir)
from preprocessing.paths import pointwise_features_path


class PointwiseTestCase(unittest.TestCase):
    """
    test case superclass for pointwise data

    attributes:
    - `full_data`: full pointwise dataframe
    - `sample_patient`: filtered patient id 55 from `full_data`
    - `sample_updrs_speed`: UPDRS-1 of `sample_patient`
    """
    def __init__(self, *args, **kwargs):
        """
        load dataset at initiation
        """
        super(PointwiseTestCase, self).__init__(*args, **kwargs)
        self.load_pointwise_dataset()

    def load_pointwise_dataset(self):
        """
        read pointwise features
        """
        self.full_data = pd.read_csv(pointwise_features_path)
        self.sample_patient = self.full_data[
            self.full_data['patient_id'] == 55]
        self.sample_updrs_speed = self.sample_patient['updrs_1_dt']
        self.sample_updrs_speed = self.sample_updrs_speed[
            self.sample_updrs_speed.notna()
        ]


if __name__ == '__main__':
    # debug
    testcase = PointwiseTestCase()
    print('debug')
