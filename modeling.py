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

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

fb_df = pd.read_csv(data_file, sep=';')

input_columns = list(fb_df.columns[0:7])
performance_columns = list(fb_df.columns[7:15])

numeric_columns = list(set(fb_df.columns[0:7]) - set(fb_df.columns[1:3]))
category_columns = list(fb_df.columns[1:3])

fb_df['Type'] = fb_df['Type'].replace(['Photo', 'Status', 'Link', 'Video'], [1, 2, 3, 4])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

fb_df_num_tr = num_pipeline.fit_transform(fb_df[numeric_columns])
fb_df_num = pd.DataFrame(fb_df_num_tr, columns=numeric_columns)

fb_df_prepared = pd.concat([fb_df_num, fb_df[['Type', 'Category']]], axis=1)


# modeling
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


t0 = time()
lin_svr = cv_performance_model(LinearRegression())
lin_svr_df = pd.Series(lin_svr, index=performance_columns)
print(f'Time elapsed: {round(time() - t0, 2)} s.')
lin_svr_df.sort_values(ascending=True, inplace=True)
lin_svr_df = pd.DataFrame({'Performance metric': lin_svr_df.index, 'RMSE': lin_svr_df.values})
print(lin_svr_df)

t0 = time()
res_svr = cv_performance_model(SVR())
# res_svr.sort()
# res_svr_str = [str(mean)+' +/- '+str(std) for mean, std in res_svr]
print(f'Time elapsed: {round(time() - t0, 2)} s.')
res_svr_df = pd.Series(res_svr, index=performance_columns)
res_svr_df.sort_values(ascending=True, inplace=True)
res_svr_df = pd.DataFrame({'Performance metric': res_svr_df.index, 'RMSE': res_svr_df.values})
print(res_svr_df)

t0 = time()
rfr = cv_performance_model(RandomForestRegressor(random_state=42))
rfr_sr = pd.Series(rfr, index=performance_columns)
print(f'Time elapsed: {round(time() - t0, 2)} s.')
rfr_sr.sort_values(ascending=True, inplace=True)
rfr_df = pd.DataFrame({'Performance metric': rfr_sr.index, 'RMSE': rfr_sr.values})
print(rfr_df)
