"""
discretization methods for time series data
"""
from typing import Iterable, List
import numpy as np
import pandas as pd


delta = 1e-5


def _label_name(n_labels: int) -> List[str]:
    """
    inputs:
    - `n_labels`: number of labels
    outputs:
    - `label_names`: list of label names
    """
    return [f'bin_{i}' for i in range(n_labels)]


def merge_bins(
        data: pd.Series,
        min_diff: float = 0.25,
        handle_zero: bool = True) -> pd.Series:
    """
    discretize into bins of width 'delta'

    inputs:
    - `data`: time series data
    - `min_diff`: width of the bins.
    If next bin closer than this value, it will merge.

    outputs:
    - binned_data: pandas series of labelled data
    """
    nunique = data.value_counts().sort_index()
    bin_counts = dict()
    bin_values = np.array([])  # list of (lower, upper) in bin
    last_value = None
    for value, count in zip(nunique.index, nunique.values):
        # if value is close enough to previous value, merge
        if last_value is not None and value < last_value + min_diff:
            bin_counts[last_value] += count
        # otherwise, create new entry in dictionary
        else:
            bin_counts[value] = count
            last_value = value
            bin_values = np.append(bin_values, value)
    bin_values = np.append(bin_values, np.inf)
    if handle_zero:
        bin_values[0] = delta
        bin_values = np.insert(bin_values, 0, - np.inf)
    bin_values = np.sort(bin_values)
    return bin_values


def merge_binning(data: pd.Series, delta: float = 0.25) -> pd.Series:
    merge_bin_cutoff = merge_bins()
    binned_data = pd.cut(data, merge_bin_cutoff)
    return binned_data


def quarterly_bins(
        data: pd.Series,
        partitions: int = 8,
        handle_zero: bool = True):
    """
    binning that separates into approximately equal counts

    inputs:
    - `data`: pd.Series, raw numerical data
    - `partitions`: number of bins
    - `handle_zero`: whether to put zero values in a separate bin

    outputs:
    - `labels`: pd.Series, categorical bin labels
    """
    # partitioning: define bin values
    bins = []
    # sample and partition non-zero data
    data_copy = data[data > delta].copy() if handle_zero else data.copy()
    sample_data = data_copy.sample(50) if len(data_copy) > 100 \
        else data_copy.copy()
    # use pandas.qcut, but on non-zero values
    if partitions > 2:
        bins = pd.qcut(sample_data, q=partitions - 2, retbins=True)[1]
    else:
        bins = pd.Series([])
    # cover maximum bin
    bins = np.append(bins, np.inf)
    # handling zero bin separately
    if handle_zero:
        bins[0] = delta
        bins = np.insert(bins, 0, -np.inf)
    return bins


def two_state_bins():
    """
    binning for two states (plateau and slope)
    """
    return np.array([-np.inf, delta, np.inf])


def apply_binning(data: pd.Series, bins: Iterable[float]):
    """
    apply binning to pandas Series given bins input

    inputs:
    - `data`: input data
    - `bins`: list of floats representing bins upper bound

    outputs:
    - `labels`: pandas Series of label strings
    """
    # NOTE: consider separate functions to make bins and mark labels
    bin_labels = _label_name(len(bins)-1)
    labels = pd.cut(data, bins, labels=bin_labels)
    return labels
