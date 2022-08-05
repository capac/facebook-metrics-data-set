#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
from time import time
from data_preparation import DataPreparation
# Scikit-Learn regression models
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
data_prep = DataPreparation(data_file)
fb_na_tr = data_prep.transform()

# Comment on output columns:
# With the OneHotEncoder class, the number of columns in the transformed matrix
# is 57. The last 12 columns are the performance metrics which are used as labels
# for modeling.


# data modeling
def cv_performance_model(model, threshold=3.0):
    cross_val_scores = []
    # 12 columns for preformance metrics
    num_metrics = fb_na_tr.shape[1]-1
    # range from 56 (included) to 45 (included)
    for col in range(num_metrics, num_metrics-len(data_prep.output_columns), -1):
        clone_model = clone(model)
        # range from 0 (included) to 44 (included), all categorical except for the last one
        X, y = fb_na_tr[:, 0:num_metrics-len(data_prep.output_columns)+1], fb_na_tr[:, col]
        X_thr, y_thr = X[(np.abs(y) < threshold)], y[(np.abs(y) < threshold)]
        scores = cross_val_score(clone_model, X_thr, y_thr, cv=5,
                                 scoring='neg_root_mean_squared_error')
        rmse_mean = -scores.mean()
        rmse_std = scores.std()
        err_perc = 100*rmse_std/rmse_mean
        cross_val_scores.append({'rmse': round(rmse_mean, 6),
                                 'std': round(rmse_std, 6),
                                 'perc': round(err_perc, 6)})
    return cross_val_scores


def performance_model_table(model):
    t0 = time()
    cross_val_scores = cv_performance_model(model)
    data_prep.output_columns.reverse()
    reg_df = pd.DataFrame(cross_val_scores, index=data_prep.output_columns)
    reg_df.sort_values(by='rmse', ascending=True, inplace=True)
    reg_df.reset_index(inplace=True)
    reg_df.rename(columns={'index': 'Performance metric', 'rmse': 'RMSE',
                           'std': 'SD (sigma)', 'perc': 'Error (%)'}, inplace=True)
    print(f'Time elapsed for {name}: {round(time() - t0, 2)} s.')
    return reg_df


# coef_ weights are only available with SVR(kernel='linear')
model_list = {'Support Vector Machine Regressor': SVR(kernel='rbf'),
              'Random Forest Regressor': RandomForestRegressor(random_state=42, n_jobs=-1),
              'Ridge': Ridge(random_state=42),
              }

# model calculation and saving output to file
with open(work_dir / 'stats_output_no_outliers_in_metrics.txt', 'w') as f:
    for name, model in model_list.items():
        results = performance_model_table(model)
        f.writelines(f'Results for {name}: \n{(results)}\n\n')
    f.writelines('\n')
    print('Done!')
