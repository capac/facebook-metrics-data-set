#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
from time import time
# Scikit-Learn preprocessing classes
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
# Scikit-Learn regression models
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
fb_df = pd.read_csv(data_file, sep=';', na_values='NaN')

# column distinction
selected_columns = list(fb_df.columns[0:15])

# input column data type
input_columns = selected_columns[0:7]
output_columns = selected_columns[7:15]
# print(f'output_columns: {output_columns}')

numeric_cols = [input_columns[0]] + input_columns[3:6] + output_columns
# print(f'numeric_cols: {numeric_cols}')
cat_onehot_cols = input_columns[1:3] + [input_columns[6]]
# print(f'cat_onehot_cols: {cat_onehot_cols}')

# substitution of NA (just the one in 'Paid') and standardization of data
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent', missing_values=pd.NA)),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    # 6 columns for categorical data
    ('cat_onehot', OneHotEncoder(sparse=False, drop='first'), cat_onehot_cols),
    # 12 columns for numerical data
    ('num', num_pipeline, numeric_cols),
])

# application for feature transformation pipeline
input_fb_df = fb_df[selected_columns].copy()
# input_fb_df.dropna(inplace=True)
fb_df_tr = full_pipeline.fit_transform(input_fb_df)


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
    reg_df = pd.DataFrame(cross_val_scores, index=output_columns)
    pd.set_option('display.precision', 10)
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
