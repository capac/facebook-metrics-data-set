#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
# Scikit-Learn preprocessing classes
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro

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

# test calculation and saving output to file
with open(work_dir / 'shapiro-wilk-test-results.txt', 'w') as f:
    for col, name in zip(range(7, 15), output_columns):
        result = shapiro(fb_df_tr[:, col])
        f.writelines(f'''Result for '{name}': \n{(result)}\n\n''')
    f.writelines('\n')
    print('Done!')
