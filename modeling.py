#! /usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from time import time
from data_preparation import DataPreparation
# Scikit-Learn regression models
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor

home_dir = Path.home()
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/uci-ml-repository/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
renamed_perf_metric_cols = {'Lifetime Post Total Reach': 'LT Post Total Reach',
                            'Lifetime Post Total Impressions': 'LT Post Total Imp',
                            'Lifetime Engaged Users': 'LT Engd Users',
                            'Lifetime Post Consumers': 'LT Post Consumers',
                            'Lifetime Post Consumptions': 'LT Post Consump',
                            'Lifetime Post Impressions by people who have liked your Page':
                            'LT Post Imp + Liked Page',
                            'Lifetime Post reach by people who like your Page':
                            'LT Post Reach + Liked Page',
                            'Lifetime People who have liked your Page and engaged with your post':
                            'LT People + Engd Post',
                            'comment': 'Comment',
                            'like': 'Like',
                            'share': 'Share',
                            'Total Interactions': 'Total Int'
                            }

data_prep = DataPreparation(data_file)
# the NumPy array has categorical and numerical (13) columns
fb_na_tr = data_prep.transform()
fb_mod_df = pd.DataFrame(fb_na_tr[0], columns=fb_na_tr[1])
fb_mod_df.rename(columns=renamed_perf_metric_cols, inplace=True)


# data modeling
class DataModeling():
    '''
    last 12 columns are the performance metrics which are used as labels for modeling
    '''
    def __init__(self, data_frame, model, threshold=2.0):
        self.data_frame = data_frame
        self.model = model
        self.threshold = threshold
        # 12 performance metrics columns
        self.perf_metric_cols = self.data_frame.columns[-12:].tolist()
        # modeling columns: total columns - 12 performance metrics columns
        self.modeling_cols = list(set(self.data_frame.columns.tolist()) - set(self.perf_metric_cols))

    def _cal_perf_metrics(self):
        cross_val_scores = []
        X_copy = self.data_frame[self.modeling_cols].copy()
        y_copy = self.data_frame[self.perf_metric_cols].copy()
        X_copy[X_copy.columns.tolist()[-1]] = np.log(X_copy[X_copy.columns.tolist()[-1]] + 1)
        y_log = np.log(y_copy + 1)
        for col in self.perf_metric_cols:
            clone_model = clone(self.model)
            # removing outliers for performance metrics
            X_thr = X_copy[(np.abs(y_log[col]) < self.threshold)]
            y_thr = y_log[col][(np.abs(y_log[col]) < self.threshold)]
            scores = cross_val_score(clone_model, X_thr, y_thr, cv=10, n_jobs=-1,
                                     scoring='neg_root_mean_squared_error')
            r2_scores = cross_val_score(clone_model, X_thr, y_thr, cv=10,
                                        n_jobs=-1, scoring='r2')
            r2_mean = r2_scores.mean()
            rmse_mean = -scores.mean()
            rmse_std = scores.std()
            err_perc = 100*rmse_std/rmse_mean
            cross_val_scores.append({'rmse': round(rmse_mean, 6),
                                     'std': round(rmse_std, 6),
                                     'perc': round(err_perc, 6),
                                     'r2': round(r2_mean, 6)})
        return cross_val_scores

    def perf_table(self):
        t0 = time()
        cross_val_scores = self._cal_perf_metrics()
        reg_df = pd.DataFrame(cross_val_scores, index=self.perf_metric_cols)
        reg_df.sort_values(by='rmse', ascending=True, inplace=True)
        reg_df.reset_index(inplace=True)
        reg_df.rename(columns={'index': 'Performance metric', 'rmse': 'RMSE',
                               'std': 'SD (sigma)', 'perc': 'Error (%)', 'r2': 'R2'}, inplace=True)
        print(f'Time elapsed for {model.__class__.__name__}: {round(time() - t0, 2)} s.')
        return reg_df


# coef_ weights are only available with SVR(kernel='linear')
model_list = {'SGDRegressor': SGDRegressor(random_state=42),
              'Lasso': Lasso(random_state=42),
              'Support Vector Regressor': SVR(kernel='linear', C=0.5),
              'Ridge': Ridge(random_state=42),
              'Random Forest Regressor': RandomForestRegressor(n_estimators=200,
                                                               random_state=42,
                                                               n_jobs=-1),
              }

# model calculation and saving output to file
with open(work_dir / 'stats_output.txt', 'w') as f:
    t1 = time()
    for name, model in model_list.items():
        data_metrics = DataModeling(fb_mod_df, model)
        f.writelines(f'Results for {name}: \n{(data_metrics.perf_table())}\n\n')
    f.writelines('\n')
    print(f'Total time elapsed: {round(time() - t1, 2)} s.')
