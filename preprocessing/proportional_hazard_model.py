'''
Proportional hazard model feature extraction
- extract plateau and propagation features, incl delta and duration

https://en.wikipedia.org/wiki/Proportional_hazards_model

TODO: finish the script poc for feature extraction
after that, integrate into a function
'''
import os
import pandas as pd


# read clinical data
clinical_data_path = R"D:\data\amp-pd-data\train_clinical_data.csv"
if not os.path.isfile(clinical_data_path):
    raise FileNotFoundError(f"csv file not found: {clinical_data_path}")
clinical_data = pd.read_csv(clinical_data_path)
# DATA CLEANING
# rename columns
clinical_data.rename(
    columns={
        'upd23b_clinical_state_on_medication': 'on_medication',
        'visit_month': 'month'
        }, inplace=True)

