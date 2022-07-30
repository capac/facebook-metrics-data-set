#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
from time import time
from data_preparation import DataPreparation
# Scikit-Learn regression models
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
data_prep = DataPreparation(data_file)
fb_df_tr = data_prep.fit_transform()


# data modeling
def cv_performance_model(model, threshold=3.0):
    cross_val_scores = list()
    # 8 columns for preformance metrics
    for col in range(7, 15):
        X = fb_df_tr
        y = fb_df_tr[:, col]
        X_thr = X[(np.abs(y) < threshold)]
        y_thr = y[(np.abs(y) < threshold)]
        # print(f'{selected_columns[col]}, {model.__class__.__name__} model: {y_thr.shape[0]}')
        scores = cross_val_score(model, X_thr, y_thr, scoring='neg_mean_squared_error', cv=5)
        rmse_scores = np.sqrt(-scores)
        rmse_mean = rmse_scores.mean()
        # print(f'col: {col}, rmse_mean: {rmse_mean}')
        rmse_std = rmse_scores.std()
        err = rmse_std/rmse_mean
        cross_val_scores.append({'rmse': round(rmse_mean, 6),
                                 'std': round(rmse_std, 6),
                                 'perc': round(100*err, 6)})
    return cross_val_scores


def performance_model_table(model):
    t0 = time()
    cross_val_scores = cv_performance_model(model)
    reg_df = pd.DataFrame(cross_val_scores, index=data_prep.output_columns)
    reg_df.sort_values(by='rmse', ascending=True, inplace=True)
    reg_df.reset_index(inplace=True)
    reg_df.rename(columns={'index': 'Performance metric', 'rmse': 'RMSE',
                           'std': 'Standard deviation', 'perc': 'Error (%)'}, inplace=True)
    print(f'Time elapsed for {name}: {round(time() - t0, 2)} s.')
    return reg_df


# coef_ weights are only available with SVR(kernel='linear')
model_list = {'Support Vector Machine Regressor': SVR(kernel='linear'),
              'Ridge': Ridge(),
              'Linear Regression': LinearRegression(),
              }

# model calculation and saving output to file
with open(work_dir / 'stats_output_no_outliers_in_metrics.txt', 'w') as f:
    for name, model in model_list.items():
        results = performance_model_table(model)
        f.writelines(f'Results for {name}: \n{(results)}\n\n')
    f.writelines('\n')
    print('Done!')
