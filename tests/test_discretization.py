"""
test binning functions
"""
import unittest
from testclass import PointwiseTestCase
from algorithms.discretization import merge_bins, quarterly_bins, apply_binning
from algorithms.simpleinfotheory import entropyempirical


class TestBinning(PointwiseTestCase):
    """
    test discretizing entropy from binning
    """
    def test_setup(self):
        assert hasattr(self, 'sample_patient')

    def test_merge_binning(self):
        """
        merge binning

        expected behaviour: close propagation rates merge, otherwise separate
        """
        bins = merge_bins(self.sample_updrs_speed)
        labels = apply_binning(self.sample_updrs_speed, bins)
        entropy_binning = entropyempirical(labels)
        self.assertTrue(
            entropy_binning[0] > 0,
            "discrete entropy shouod not be less or equal 0")

    def test_quarterly_binning(self):
        """
        quarterly binning

        expected behavior: generates required number of bins to approximately
        equally divide the data.
        """
        n_partitions = 8
        full_updrs_speed = self.full_data['updrs_1_dt']
        full_updrs_speed = full_updrs_speed[full_updrs_speed.notna()]
        bins = quarterly_bins(
            full_updrs_speed, partitions=n_partitions)
        # FIXME: bins should increase monotonically
        labels1 = apply_binning(self.sample_updrs_speed, bins)
        labels2 = apply_binning(full_updrs_speed, bins)
        assert len(labels1.unique()) <= len(bins)
        self.assertLessEqual(
            len(labels1.unique()), n_partitions + 1,
            "patient level bins greater than n_partitions")
        self.assertLessEqual(
            len(labels2.unique()), n_partitions + 1,
            "population level bins greater than n_partitions")

    def entropy_rate(self):
        pass


if __name__ == '__main__':
    unittest.main()
