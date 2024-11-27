"""
discretization methods for time series data
"""
import pandas as pd


def merge_binning(data: pd.Series, delta: float = 0.25) -> pd.Series:
    """
    discretize into bins of width 'delta'

    inputs:
    - `data`: time series data
    - `delta`: width of the bins.
    If next bin closer than this value, it will merge.

    outputs:
    - binned_data: pandas series of labelled data
    """
    nunique = data.value_counts().sort_index()
    bin_counts = dict()
    bin_values = []  # list of (lower, upper) in bin
    last_value = None
    for value, count in zip(nunique.index, nunique.values):
        # if value is close enough to previous value, merge
        if last_value is not None and value < last_value + delta:
            bin_counts[last_value] += count
        # otherwise, create new entry in dictionary
        else:
            bin_counts[value] = count
            last_value = value
            bin_values.append()
    binned_data = pd.qcut(data, bin_values)
    return binned_data


def quarterly_binning(data: pd.Series, partitions: int = 8):
    """
    binning that separates into approximately equal counts

    inputs:
    - `data`: pd.Series, raw numerical data
    - `partitions`: number of bins

    outputs:
    - `labels`: pd.Series, categorical bin labels
    """
    # partitioning: define bin values
    # handling zero propagation