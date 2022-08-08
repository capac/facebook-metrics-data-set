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
from xgboost import XGBRegressor

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
data_prep = DataPreparation(data_file)
fb_na_tr = data_prep.transform()


# data modeling
class DataModeling():
    '''
    With the OneHotEncoder class, the number of columns in the transformed matrix is 57.
    The last 12 columns are the performance metrics which are used as labels for modeling.
    '''
    def __init__(self, data, model, perf_metric_cols, threshold=2.0):
        self.data = data
        self.model = model
        self.threshold = threshold
        # 12 preformance metrics columns, from last to first
        self.perf_metric_cols = perf_metric_cols[::-1]
        # total number of columns
        self.tot_num_cols = self.data.shape[1]
        self.diff_cols = self.tot_num_cols - len(self.perf_metric_cols)

    def _cal_perf_metrics(self):
        cross_val_scores = []
        # range from 56 (included) to 45 (included)
        for col in range(self.tot_num_cols-1, self.diff_cols-1, -1):
            # range from 0 (included) to 44 (included), all categorical except for the last one
            X = self.data[:, 0:self.diff_cols].copy()
            y = self.data[:, col].copy()
            clone_model = clone(self.model)
            # removing outliers
            X_thr = X[(np.abs(y) < self.threshold)]
            y_thr = y[(np.abs(y) < self.threshold)]
            scores = cross_val_score(clone_model, X_thr, y_thr, cv=5,
                                     scoring='neg_root_mean_squared_error')
            rmse_mean = -scores.mean()
            rmse_std = scores.std()
            err_perc = 100*rmse_std/rmse_mean
            cross_val_scores.append({'rmse': round(rmse_mean, 6),
                                     'std': round(rmse_std, 6),
                                     'perc': round(err_perc, 6)})
        return cross_val_scores

    def perf_table(self):
        t0 = time()
        cross_val_scores = self._cal_perf_metrics()
        reg_df = pd.DataFrame(cross_val_scores, index=self.perf_metric_cols)
        reg_df.sort_values(by='rmse', ascending=True, inplace=True)
        reg_df.reset_index(inplace=True)
        reg_df.rename(columns={'index': 'Performance metric', 'rmse': 'RMSE',
                               'std': 'SD (sigma)', 'perc': 'Error (%)'}, inplace=True)
        print(f'Time elapsed for {model.__class__.__name__}: {round(time() - t0, 2)} s.')
        return reg_df


# coef_ weights are only available with SVR(kernel='linear')
model_list = {'Support Vector Machine Regressor': SVR(kernel='rbf', C=0.5),
              'Random Forest Regressor': RandomForestRegressor(n_estimators=200,
                                                               random_state=42,
                                                               n_jobs=-1),
              'Ridge': Ridge(random_state=42),
              'XGBRegressor': XGBRegressor(n_estimators=200,
                                           random_state=42,
                                           eval_metric='rmse',
                                           n_jobs=-1),
              }

# model calculation and saving output to file
with open(work_dir / 'stats_output_no_outliers_in_metrics.txt', 'w') as f:
    t1 = time()
    for name, model in model_list.items():
        data_metrics = DataModeling(fb_na_tr, model, data_prep.output_columns)
        f.writelines(f'Results for {name}: \n{(data_metrics.perf_table())}\n\n')
    f.writelines('\n')
    print(f'Total time elapsed: {round(time() - t1, 2)} s.')
