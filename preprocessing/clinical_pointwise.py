"""
Pointwise feature extraction.

The output dataframe contains including:
- isotonic regression
- calculation of propagation speed

Exports pandas.DataFrame
"""
from typing import Union
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from paths import pointwise_features_path
pd.set_option('future.no_silent_downcasting', True)


SAVE_DATA = True


def _delta(s_input: pd.Series):
    """
    calculate delta change of a pd.Series
    """
    s_delta = s_input - s_input.shift(1)
    return s_delta


def expand_timeline(df_input: pd.DataFrame, interval: int = 1) -> pd.DataFrame:
    """
    expand clinical data into monthly samples, where missing updrs will give
    null value. this creates continuous x-axis for fitting.

    inputs:
    * `df_input`: pd.DataFrame, dataframe to be expanded,
        timeline is based on 'month' column
    * `interval`: sampling interval

    outputs:
    * `df_output`: dataframe with expanded timeline
    """
    df_output = df_input.copy()  # Make a copy of the dataframe
    df_output['clinical_visit'] = True
    # expanding the time axis, to cover every month
    continuous_month = np.arange(
        df_output['month'].iloc[0], df_output['month'].iloc[-1] + 1, interval)
    continuous_month = pd.Series(continuous_month, name='month')
    df_output = df_output.merge(continuous_month)
    df_output = pd.merge(df_output, continuous_month, how='right', on='month')
    # after expansion, fill null fields
    # patient id, it's grouped so fill first
    df_output['patient_id'] = df_output['patient_id']\
        .fillna(df_input['patient_id'].iloc[0])\
        .astype(df_input['patient_id'].dtype)
    df_output['clinical_visit'] = df_output['clinical_visit']\
        .fillna(False)
    return df_output


def updrs_iso_regression(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    isotonic regression for dataframe on `updrs_1,2,3,4`, by `month`
    inputs:
    * `df_input`: pd.DataFrame
    outputs:
    * `df_output`: pd.DataFrame
    """
    df_output = df_input.copy()  # Make a copy of the dataframe
    # check input fields
    updrs_columns = [f'updrs_{i}' for i in (1, 2, 3, 4)]
    required_columns = (*(x for x in updrs_columns), 'month')
    for column in required_columns:
        if column not in df_output.columns:
            raise KeyError(f'column missing: {column}')
    # iterate through columns, isotonic regression
    for column in updrs_columns:
        # filter out columns where the updrs dimension is null
        valid_updrs_month = df_output[
            df_output[[column, 'month']].notna().all(axis=1)]
        if not valid_updrs_month.empty:
            regression = IsotonicRegression(out_of_bounds='clip')\
                .fit(valid_updrs_month['month'], valid_updrs_month[column])
            # make predictions on the month values
            df_output[f'{column}_fit'] = regression.predict(df_output['month'])
            # backward fill NaN
        else:
            # if no data at all (e.g. happens in updrs 4), fill with 0
            df_output[f'{column}_fit'] = 0.0
    return df_output


def calculate_speed(updrs: pd.Series, month: Union[pd.Series, float]):
    """
    Calculate delta based on updrs scoring and month
        using (updrs - prev_updrs) / (month - prev_month)

    **Note**: this function outputs null for the first month
    """
    if isinstance(month, pd.Series):
        month_delta = _delta(month)
    elif isinstance(month, float):
        month_delta = pd.Series([month for _ in updrs], index=updrs.index)
    else:
        raise TypeError('month must be pd.Series or float')
    updrs_delta = _delta(updrs)
    updrs_dt = updrs_delta / month_delta
    return updrs_dt


def updrs_calculate_speed(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate updrs propagation speed, on 4 dimensions
    """
    df_output = df_input.copy()
    for i in (1, 2, 3, 4):
        df_output[f'updrs_{i}_dt'] = calculate_speed(
            df_output[f'updrs_{i}_fit'], df_output['month']
        )
    return df_output


def prev_updrs(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Get previous UPDRS values.
    """
    df_output = df_input.copy()
    for i in (1, 2, 3, 4):
        df_output[f'updrs_{i}_prev'] = df_output[f'updrs_{i}_fit'].shift(1)
    return df_output


def group_actions(group: pd.DataFrame) -> pd.DataFrame:
    """
    Group operation to extract pointwise features.
    """
    df_return = expand_timeline(group)
    df_return = updrs_iso_regression(df_return)
    df_return = updrs_calculate_speed(df_return)
    df_return = prev_updrs(df_return)
    return df_return


def pointwise_features(clinical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Obtain pointwise features from clinical pointwise data.
    """
    clinical_data_group = clinical_data.copy()
    clinical_data_group = clinical_data_group.groupby('patient_id')
    clinical_data_group = clinical_data_group.apply(
        lambda group: group_actions(group)
    )
    # clinical_data_group.drop(columns='patient_id', inplace=True)
    df_pointwise = clinical_data_group.reset_index(drop=True)
    return df_pointwise


if __name__ == '__main__':
    from basic import load_clinical_data
    df_clinical = load_clinical_data()
    df_pointwise = pointwise_features(df_clinical)
    if SAVE_DATA:
        df_pointwise.to_csv(pointwise_features_path, index=False)
    print('DEBUG')
