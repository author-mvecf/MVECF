# Mean Variance Efficient Collaborative Filtering for Stock Recommendation


## Introduction
Mean Variance Efficient Collaborative Filtering (MVECF) is a recommender system for stock recommendation that specifically designed to improve the pareto optimality in a trade-off between risk and return (i.e., the mean-variance efficiency) of recommended portfolios. Regularization technique is used to ensure that the recommendation is made based on the user’s current portfolio while increasing diversification effect. MVECF is further restructured to ordinary weighted matrix factorization form and it can be performed by only chaging the target ratings. This modified ratings can be used in the negative sampling in the state-of-the-art graph based ranking models to incoporate MVECF.
This repository contains python codes for MVECF, baseline models (BPR, WMF, and LightGCN), analysis codes and sample data.

## Environment
The codes of MVECF are implemented and tested under the following development environment:
* python=3.9.13
* numpy=1.21.5
* scipy=1.9.3
* scikit-learn=1.1.3
* pandas=1.5.1

Following packages are used for baseline models only.
* tensorflow=2.10.0
* pytorch=1.13.0
* cvxopt=1.2.6

## Datasets
We split each dataset into yearly sub-datasets as shown in Figure. In our experiment, we recommend items to users at the end of each year, so we use holdings snapshot reported in December of year T as the holdings of year T. We define preferred items as the current holdings to avoid recommending the items that were recently sold by the user. Although we have monthly holdings data, portfolios do not change much every month, so we create yearly sub-datasets for more robust experiments. 
![image](https://github.com/author-mvecf/MVECF/assets/132906890/bda0d514-21d1-4db9-9601-98cd3e715dbb)

For MVECF model, we need factor exposures and idiosyncratic risks should be estimated before the recommendation. Hence, we estimate them using returns data in the past 5 years (years T-4 to T). For ex post performance evaluation, we use the next 5 years returns data (years T+1 to T+5). For example, when we are running an experiment for the year 2015 dataset, the user-item interaction data is constructed from the holdings in Dec 2015, the stock mean return and covariance matrix are estimated with the stock returns data from 2011 to 2015, and the stock returns data from 2016 to 2020 is used for ex-post performance evaluation. 

As a result, we get 10 yearly sub-datasets from 2006 to 2015. The number of users and items, and the average number of holdings in resulting 20 yearly sub-datasets (10 for CRSP, 10 for Thomson Reuters) are summarized in Table 1. The user-item interaction data in each dataset is divided into train, test, and validation data at a ratio of 8:1:1.
The sample data is CRSP data for year 2012.

`factor_model_params.pkl` is Fama-French 3 factor model parameters calculated with past 5 years returns data to get mean and covariance matrix of items.
`weekly_ret_data.pkl` is the weekly stock return series of items (including the period of past and future 5 years).
`holdings_data.pkl` is the mutual fund holding snapshot of December 2012.

## How to Run the Code

* MVECF models
MVECF regularization model
```
python script_mvecf_reg.py
```
MVECF restructured to WMF
```
python script_mvecf_wmf.py
```
* Ordinary recommender systems
WMF
```
python script_wmf.py
```
Novelty enhanced BPR
```
python script_bpr_nov.py
```
* Two-phase method
Run after running ordinary recommendation model (ex. wmf).
Set `base_model` as the ordinary RS model name in the script.
```
python script_twophase.py
```

* Incorporating MVECF into Ranking Models.
LightGCN is used for graph based ranking models.
```
python script_mvecf_lgcn.py
```
You can run ordinary LightGCN with the same script by setting

`data = ImplicitCF(train=train, test=test, seed=SEED, factor_params=None)`


After running the models, run `get_result_table.py` to get the analysis result.
