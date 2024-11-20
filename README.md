# Information Theory for Parkinson's Disease Forecasting Solvability

Analyze solvability of forecasting Parkinson's Disease propagation from [AMP-PD dataset](https://kaggle.com/competitions/amp-parkinsons-disease-progression-prediction).

## Introduction

The research will answer following questions:
* How much information does historic UPDRS time series provide about future UPDRS values?
* How much information does other dimensions' UPDRS dimensions provide about future UPDRS values?
    - Answer: Very little, insignificant compared to AIS and entropy rate.
* How much information can patient profile provide about future UPDRS values?
* How much connection are there between patient profiles and 

Notebooks and corresponding anlysis:
* `updrs_self_mi.ipynb`: Self-MI between previous and next sample, conditional MI on patient groups, total entropy of UPDRS-1,2,3,4.
* `updrs_self_mi_2.ipynb`: MI across UPDRS dimensions, AIS, patient group clustering.
* `clinical_te.ipynb`: Transfer entropy between UPDRS dimensions.
* `clinical_er.ipynb`: Entropy rate of UPDRS scores. 

## Configuration

### Generate dataset

Due to competition rules in AMP-P

## Future works
* Find better probability density function for differential entropy calculation.
* Find profile-based feature that optimizes synergy in patient group, possibly based on pattern features in time series.
* Connect time series patterns to protein and peptide spectrums.