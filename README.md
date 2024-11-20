# Information Theory for Parkinson's Disease Forecasting Solvability

Analyze solvability of forecasting Parkinson's Disease propagation from [AMP-PD dataset](https://kaggle.com/competitions/amp-parkinsons-disease-progression-prediction).

## Introduction

The research will answer following questions:
* How much information does historic UPDRS time series provide about future UPDRS values?
    - Answer: Pairwise MI in historic data provides 64.5%, 65.2%, 84.2% mutual information about next point from previous 2 data points. However the error will be amplified when used to forecast further. AIS peak at $k=1$, providing around $1.24\ {nats}$ in 4 dimensions. But entropy rate and differential entropy is unclear.

* How much information does other dimensions' UPDRS dimensions provide about future UPDRS values?
    - Answer: Very little TE found. It is insignificant compared to information stored in time series, using AIS and pairwise MI.
* How much information can patient profile provide about future UPDRS values?
* How much connection are there between patient profiles and protein and peptide spectrums?

Notebooks and corresponding anlysis:
* `updrs_self_mi.ipynb`: Self-MI between previous and next sample, conditional MI on patient groups, total entropy of UPDRS-1,2,3,4.
* `updrs_self_mi_2.ipynb`: MI across UPDRS dimensions, AIS, patient group clustering.
* `clinical_te.ipynb`: Transfer entropy between UPDRS dimensions.
* `clinical_er.ipynb`: Entropy rate of UPDRS scores. 

## Configuration

### Set up JIDT

Scripts render JIDT classes and algorithms for information dynamics calculatins, including differential entropy, active information storage (AIS), transfer entropy (TE).

JIDT repository by Joseph T. Lizier, 2014: [https://github.com/jlizier/jidt]()

A toolkit written in Java, with user interface for parameter configurations, and code generators.

### Configure dataset and JIDT directories

Configure data and jidt directories using `preprocessing/paths.py`. Replace `data_dirs` with your local directory containing `.csv` files from competition. Replace `jidt_dirs` 

### Obtain dataset and run preprocessing scripts

Due to data privacy rules in AMP-PD, the repository will not contain original data. However, it is obtainable from Kaggle with user registration.

Under `preprocessing` folder are scripts for data preparation and feature extractions:
* `basic.py`: shared data preparation steps
* `clinical_pattern.py`: 

## Future works
* Find better probability density function for differential entropy calculation.
* Find profile-based feature that optimizes synergy in patient group, possibly based on pattern features in time series.
* Connect time series patterns to protein and peptide spectrums.