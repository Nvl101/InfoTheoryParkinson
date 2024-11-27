"""
test binning functions
"""
import unittest
from testclass import PointwiseTestCase
from algorithms.discretization import merge_binning
from algorithms.simpleinfotheory import entropyempirical


class TestBinning(PointwiseTestCase):
    """
    test discretizing entropy from binning
    """
    def test_setup(self):
        assert hasattr(self, 'sample_patient')
    def test_merge_binning(self):
        binning = merge_binning(self.sample_updrs_speed)
        entropy_binning = entropyempirical(binning)
        assert entropy_binning > 0, "discrete entropy greater than 0"
    def entropy_rate(self):
        pass


if __name__ == '__main__':
    unittest.main()
