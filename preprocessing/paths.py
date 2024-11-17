"""
Configurations for data file paths.

__main__ runs checks directories and files exist.
"""

import os


# path configuration
# if local data directory exists, use that
# otherwise, use google drive
data_dirs = (
    R"D:\data\amp-pd-data",
    R"G:\CRS_CSYS5030\Project\data",
    R"/Volumes/GoogleDrive/My Drive/CRS_CSYS5030/Project/data",
)
jidt_dirs = (
    R"G:/My Drive/CRS_CSYS5030/jidt/",
    R"/Volumes/GoogleDrive/My Drive/CRS_CSYS5030/jidt",
)

# paths for java
for dir in jidt_dirs:
    if os.path.isdir(dir):
        jar_dir = dir
        break
if 'jar_dir' not in locals():
    raise FileNotFoundError('jidt directory not found.')
jarLocation = os.path.join(jar_dir, 'infodynamics.jar')

# paths for dataset files
for dir in data_dirs:
    if os.path.isdir(dir):
        data_dir = dir
        break
if 'data_dir' not in locals():
    raise FileNotFoundError('data directory not found.')
# should contain a ETL folder
etl_dir = os.path.join(data_dir, 'ETL')

clinical_data_filename = "train_clinical_data.csv"
peptide_data_filename = "train_peptides.csv"
protein_data_filename = "train_proteins.csv"
clinical_data_path = os.path.join(data_dir, clinical_data_filename)
peptide_data_path = os.path.join(data_dir, peptide_data_filename)
protein_data_path = os.path.join(data_dir, protein_data_filename)
pointwise_features_path = os.path.join(etl_dir, 'clinical_pointwise.csv')
pattern_features_path = os.path.join(etl_dir, 'clinical_pattern.csv')

if __name__ == '__main__':
    assert os.path.isdir(data_dir), \
        f'data directory {data_dir} does not exist.'
    assert os.path.isfile(clinical_data_path), \
        f'clinical data {clinical_data_path} does not exist'
    assert os.path.isfile(protein_data_path), \
        f'protein data {protein_data_path} does not exist'
    print('paths ok')
