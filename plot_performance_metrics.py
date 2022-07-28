#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
# matplotlib
import matplotlib.pyplot as plt
# Scikit-Learn preprocessing classes
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
fb_df = pd.read_csv(data_file, sep=';', na_values='NaN')

# columns used for analysis
selected_columns = list(fb_df.columns[0:15])
# columns used for modeling
input_columns = selected_columns[0:7]
# performance metric columns
output_columns = selected_columns[7:15]

# columns by data type
numeric_cols = [input_columns[0]] + input_columns[3:6] + output_columns
cat_onehot_cols = input_columns[1:3] + [input_columns[6]]

# substitution of NaN (just the one in 'Paid') and standardization of data
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
fb_na_tr = full_pipeline.fit_transform(input_fb_df)

# plots
fb_df_tr = pd.DataFrame(fb_na_tr[:, 7:15], columns=output_columns)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(13, 7))
# using 'Paired' colormap
cmap = plt.cm.Paired.colors
for ax, clm, col in zip(axes.flat, output_columns, cmap):
    ax.hist(fb_df_tr.loc[:, clm], edgecolor='k', alpha=0.9, color=col)
    ax.set_title(clm)
plt.suptitle('Distribution of standardized performance metrics')
plt.tight_layout()
plt.savefig('plots/performance_metrics_hist.png', dpi=288, bbox_inches='tight')
