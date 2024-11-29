"""
transform clinical data for Markov Chain analysis

design:
- isotonic transformation for raw data, to obtain propagation speed
- calculate propagation speed
- for each dimension, label time series data as "plateau" and "slope",
further, separate propagation speed by binning
"""
import numpy as np
import pandas as pd
from algorithms.discretization import 
from basic import load_clinical_data
from clinical_pointwise import pointwise_features
from paths import markov_features_path
pd.set_option('future.no_silent_downcasting', True)


SAVE_DATA = True
propagation_speed_columns = ["updrs_{i}_dt".format(i=i) for i in (1,2,3,4)]
markov_label_format = ["markov_lbl_{i}".format(i=i) for i in (1,2,3,4)]


def markov_label(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    assign labels to propagation speed, for markov chain prediction
    """
    for column in propagation_speed_columns:
        # TODO: write a binning method suitable for Gaussian distribution
        
        column_labels = 

if __name__ == '__main__':
    df_clinical = load_clinical_data()
    df_pointwise = pointwise_features(df_clinical)
    df_markov = markov_label(df_pointwise)
    if SAVE_DATA:
        df_markov.to_csv(markov_features_path)
