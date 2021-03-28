#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
from time import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

# substitution of NAs with median and standardization
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])
fb_df_num_tr = num_pipeline.fit_transform(fb_df[numeric_columns])
fb_df_num = pd.DataFrame(fb_df_num_tr, columns=numeric_columns)
fb_df_prepared = pd.concat([fb_df_num, fb_df[['Type', 'Category']]], axis=1)


# data modeling
def cv_performance_model(model):
    cross_val_scores = []
    for col in performance_columns:
        X = fb_df_prepared
        y = fb_df[col].values
        scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
        rmse_scores = np.sqrt(-scores)
        rmse_mean = rmse_scores.mean()
#         rmse_std = rmse_scores.std()
#         y_pred = model.predict(X_test)
#         rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
#         results.append(round(rmse_score, 5))
#         cross_val_scores.append((round(rmse_mean, 3), round(rmse_std, 3)))
        cross_val_scores.append(round(rmse_mean, 3))
    return cross_val_scores


def performance_model_table(model):
    t0 = time()
    reg_model = cv_performance_model(model)
    reg_df = pd.Series(reg_model, index=performance_columns)
    reg_df.sort_values(ascending=True, inplace=True)
    reg_df = pd.DataFrame({'Performance metric': reg_df.index, 'RMSE': reg_df.values})
    print(f'Time elapsed for {name}: {round(time() - t0, 2)} s.')
    return reg_df


model_list = {'Linear Regression': LinearRegression(),
              'Support Vector Machine Regressor': SVR(),
              'Random Forest Regressor': RandomForestRegressor(random_state=42),
              }


# model calculation and saving output to file
with open(work_dir / 'stats_output.txt', 'w') as f:
    for name, model in model_list.items():
        results = performance_model_table(model)
        f.writelines(f'Results for {name}: \n{(results)}\n\n')
    f.writelines('\n')
    print('Done!')
