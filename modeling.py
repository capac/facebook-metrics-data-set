#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
from time import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from helper_functions.remove_outliers import RemoveMetricOutliers
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_colwidth', None)

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
fb_df = pd.read_csv(data_file, sep=';')

# column distinction
input_columns = list(fb_df.columns[0:7])
performance_columns = list(fb_df.columns[7:15])

# column data type
numeric_columns = list(set(fb_df.columns[0:7]) - set(fb_df.columns[1:3]))
category_columns = list(fb_df.columns[1:3])

# transformation of category strings to integers
fb_df['Type'] = fb_df['Type'].replace(['Photo', 'Status', 'Link', 'Video'], [1, 2, 3, 4])

# substitution of NAs with median and standardization in samples
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])
fb_df_num_tr = num_pipeline.fit_transform(fb_df[numeric_columns])
fb_df_num = pd.DataFrame(fb_df_num_tr, columns=numeric_columns)
fb_df_prepared = pd.concat([fb_df_num, fb_df[['Type', 'Category']]], axis=1)

# substitution of NAs with median and standardization in labels
fb_df_num_tr = num_pipeline.fit_transform(fb_df[numeric_columns])


# data modeling
def cv_performance_model(model):
    cross_val_scores = list()
    for col in performance_columns:
        X = fb_df_prepared
        y = fb_df[col].values
        rmo = RemoveMetricOutliers(sigma=2.0)
        X_rmo, y_rmo = rmo.transform(X, y)
        y_rmo = y_rmo.ravel()
        scores = cross_val_score(model, X_rmo, y_rmo, scoring='neg_mean_squared_error', cv=5)
        rmse_scores = np.sqrt(-scores)
        rmse_mean = rmse_scores.mean()
        rmse_std = rmse_scores.std()
        cross_val_scores.append({'rmse': round(rmse_mean, 3), 'std': round(rmse_std, 3)})
    return cross_val_scores


def performance_model_table(model):
    t0 = time()
    cross_val_scores = cv_performance_model(model)
    reg_df = pd.DataFrame(cross_val_scores, index=performance_columns)
    reg_df.sort_values(by='rmse', ascending=True, inplace=True)
    reg_df.reset_index(inplace=True)
    reg_df.rename(columns={'index': 'Performance metric', 'rmse': 'RMSE', 'std': 'Standard deviation'}, inplace=True)
    print(f'Time elapsed for {name}: {round(time() - t0, 2)} s.')
    return reg_df


model_list = {'Linear Regression': LinearRegression(),
              'Support Vector Machine Regressor': SVR(kernel='linear', C=1e3),
              'Random Forest Regressor': RandomForestRegressor(random_state=42),
              }


# model calculation and saving output to file
with open(work_dir / 'stats_output_no_outliers_in_metrics.txt', 'w') as f:
    for name, model in model_list.items():
        results = performance_model_table(model)
        f.writelines(f'Results for {name}: \n{(results)}\n\n')
    f.writelines('\n')
    print('Done!')
