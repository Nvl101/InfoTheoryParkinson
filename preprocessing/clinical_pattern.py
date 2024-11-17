'''
Extract features for propagation and plateau patterns

The features can be used to train estimators:
- plateau to slope: (score, duration) -> (delta)
- slope to plateau: (score, delta) -> (duration)
'''

import os
import pandas as pd
from paths import pointwise_features_path, pattern_features_path
from basic import load_clinical_data
from clinical_pointwise import pointwise_features, _delta, calculate_speed
pd.set_option('future.no_silent_downcasting', True)


SAVE_DATA = True
READ_POINTWISE_BUFFER = True  # read pointwise from csv file


def _cumsum(s_input: pd.Series, s_reset: pd.Series) -> pd.Series:
    """
    Cumulative sum with reset on s_reset being True.

    When reset, the value is reset to current line.

    inputs:
    * `s_input`: series of floats, to operate cumulative sum
    * `s_reset`: series of booleans, reset to 0 when condition is True
    """
    cumsum_reset = s_input.cumsum().where(s_reset)
    cumsum_reset.iloc[0] = 0  # initial reset value is 0
    cumsum_reset = cumsum_reset.ffill()
    s_cumsum = s_input.cumsum() - cumsum_reset + s_input
    return s_cumsum


def series_propagation_features(
        updrs: pd.Series, month: pd.Series) -> pd.DataFrame:
    """
    inputs:
        `updrs`: updrs score series

    outputs:
        `features`: pd.DataFrame containing ending score,
        plateau length and delta.
    """
    propagation_speed = calculate_speed(updrs, month)
    # finding plateau length and delta
    # booleans marking whether points are plateaus
    is_plateau = (propagation_speed.bfill() == 0)
    change_flag = (is_plateau.bfill() != is_plateau.bfill().shift(1))\
        .fillna(False)
    change_flag.iloc[0] = False
    # # merge updrs, month, and is_plateau
    # updrs_month_plateau = pd.concat((updrs, month), axis=0)
    # updrs_month_plateau = pd.merge(updrs_month_plateau, is_plateau)
    # find last changing flag
    # cumsum propagation speed for delta, reset at zero
    delta = _cumsum(propagation_speed.fillna(0), change_flag)
    # cumsum plateau duration, reset at zero
    month_delta = _delta(month).bfill()
    duration = _cumsum(month_delta.fillna(0), change_flag)
    # dataframe output, containing score, delta and duration
    df_features = pd.DataFrame({
        'month': month,
        'delta': delta,
        'duration': duration})
    # filter out changing flags, and the last one
    df_features = df_features[change_flag.shift(-1).fillna(True)]
    # then merge back to months, leaving non-changing points blank
    # df_features = pd.merge(df_features, month, how='right', on='month')
    return df_features


def propagation_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    inputs:
    `df_input`: input dataframe grouped by user id
    outputs:
    `df_output`: output dataframe containing (score, delta, duration) features
    """
    df_output = df_input.copy()  # Make a copy of the dataframe
    month = df_output['month']
    for i in (1, 2, 3, 4):
        updrs_i = df_output[f'updrs_{i}_fit']
        df_features = series_propagation_features(updrs_i, month)
        df_features = df_features.rename(columns={
            column: column + f'_{i}'
            for column in df_features.columns if column != 'month'
        })
        df_output = pd.merge(
            df_output, df_features, how='left', on='month')
    return df_output
    # raise NotImplementedError("pending series_propagation_features")


def group_actions(group: pd.DataFrame) -> pd.DataFrame:
    """
    combined group operation on "group by patient id" feature extraction
    """
    # df_return = expand_timeline(group)
    # df_return = updrs_iso_regression(df_return)
    df_return = propagation_features(group)
    return df_return


def pattern_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    extract pattern related features in group-by
    """
    # GROUPBY basd data transformation
    clinical_data_group = df_input.groupby(['patient_id'])
    # group actions:
    # 1. expand time axis to include every sample
    # 2. apply isotonic regression on updrs scores
    # 3. extract propagation features from isotonic regression
    clinical_data_group = clinical_data_group.apply(
        lambda group: group_actions(group))
    # clinical_data_group = clinical_data_group.drop(columns=['patient_id'])
    df_pattern = clinical_data_group.reset_index(drop=True)
    return df_pattern


if __name__ == '__main__':
    if READ_POINTWISE_BUFFER and os.path.isfile(pointwise_features_path):
        df_pointwise = pd.read_csv(pointwise_features_path)
    else:
        df_clinical = load_clinical_data()
        df_pointwise = pointwise_features(df_clinical)
    df_pattern = pattern_features(df_pointwise)
    if SAVE_DATA:
        df_pattern.to_csv(pattern_features_path, index=False)
    print('DEBUG')
