#! /usr/bin/env python

import os
from pathlib import Path
import pandas as pd
# import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


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

model_list = {'PCA': PCA(n_components=0.95)}


def pca_explained_variance(model):
    X = fb_df_prepared
    model.fit_transform(X)
    explained_variance_ratio = model.explained_variance_ratio_
    return explained_variance_ratio


# model calculation and saving output to file
with open(work_dir / 'explained_variance_output.txt', 'w') as f:
    for name, model in model_list.items():
        results = pca_explained_variance(model)
        f.writelines(f'Results for {name}: \nExplained variance ratio: {results}\n')
    f.writelines('\n')
    print('Done!')
