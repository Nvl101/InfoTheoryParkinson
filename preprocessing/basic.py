"""
General data processing:
1. load data from csv file to pandas.dataframe
2. data cleaning and transformation

export dataframes:
* `clicial_data`: processed clinical data
* `peptide_data`: preprocessed peptide data
"""

# TODO: move data loading and cleaning from clinical_pattern.py
import os
import pandas as pd
# path configurations
from paths import clinical_data_path, peptide_data_path


def load_clinical_data(clinical_data_path: str = clinical_data_path):
    '''
    load from csv and preprocess data
    '''
    # load clinical data from csv
    if not os.path.isfile(clinical_data_path):
        raise FileNotFoundError(f"csv file not found: {clinical_data_path}")
    clinical_data_raw = pd.read_csv(clinical_data_path)
    clinical_data = clinical_data_raw.copy()
    # drop visit id
    clinical_data.drop(columns=['visit_id'], inplace=True)
    # rename columns
    clinical_data.rename(
        columns={
            'upd23b_clinical_state_on_medication': 'on_medication',
            'visit_month': 'month'
            }, inplace=True)
    # map on_medication tags
    clinical_data['on_medication'] = clinical_data['on_medication']\
        .map({'On': True, 'Off': False})
    return clinical_data


def load_peptide_data(peptide_data_path: str = peptide_data_path):
    '''
    load peptide from csv and preprocess
    '''
    if not os.path.isfile(peptide_data_path):
        raise FileNotFoundError(f"csv file not found: {peptide_data_path}")
    peptide_data = pd.read_csv(peptide_data_path)
    peptide_data.drop(columns=['visit_id'], inplace=True)
    return peptide_data


if __name__ == '__main__':
    clinical_data = load_clinical_data(clinical_data_path)
    peptide_data = load_peptide_data(peptide_data_path)
    print('debug...')
