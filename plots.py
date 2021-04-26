#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from helper_functions.remove_outliers import RemoveMetricOutliers
from sklearn.svm import SVR

pd.set_option('display.max_colwidth', None)

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

fb_df = pd.read_csv(data_file, sep=';')
# print(f'fb_df.info(): {fb_df.info()}')
# print(f'fb_df.nunique(): {fb_df.nunique()}')
# print(f'fb_df.like.unique(): {fb_df.like.unique()}')
# print(f'fb_df.share.unique(): {fb_df.share.unique()}')

selected_columns = list(fb_df.columns[0:15])
input_columns = list(fb_df.columns[0:7])
performance_columns = list(fb_df.columns[7:15])

numeric_columns = list(set(fb_df.columns[0:6]) - set(fb_df.columns[[1, 2]]))
category_columns = list(fb_df.columns[[1, 2, 6]])
# print(f'numeric_columns: {numeric_columns}')
# print(f'category_columns: {category_columns}')

# fb_df['Type'] = fb_df['Type'].replace(['Photo', 'Status', 'Link', 'Video'], [1, 2, 3, 4])
fb_df['Paid'] = fb_df['Paid'].replace([0.0, 1.0], ['Not Paid', 'Paid'])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, numeric_columns),
    ('cat', OneHotEncoder(), category_columns),
])

# drop NaNs from data frame
selected_fb_df = fb_df[selected_columns].copy()
# print(f'fb_input_df.shape: {selected_fb_df.shape}')
selected_fb_df.dropna(inplace=True)
# print(f'fb_input_df.shape: {selected_fb_df.shape}')


fb_df_num_tr = full_pipeline.fit_transform(selected_fb_df)
# input_columns_df = fb_df[input_columns]
# print(f'input_columns_df.shape: {input_columns_df.shape}')
# print(f'fb_df_num_tr.shape: {fb_df_num_tr.shape}')
cat_one_hot_columns = list(full_pipeline.named_transformers_['cat'].categories_)
# print(f'cat_one_hot_columns: {list(cat_one_hot_columns[0]) + list(cat_one_hot_columns[2])}')
# print(f'cat_one_hot_columns: {cat_one_hot_columns}')
cat_one_hot_columns = [g for f in cat_one_hot_columns for g in list(f)]
attrib_columns = numeric_columns + list(cat_one_hot_columns)
# print(f'attrib_columns: {attrib_columns}')


def feature_importances_plot(model, col, filename):
    sns.set_context("talk")
    X = fb_df_num_tr
    y = selected_fb_df[col].values
    rmo = RemoveMetricOutliers(sigma=2.0)
    X_rmo, y_rmo = rmo.transform(X, y)
    y_rmo = y_rmo.ravel()
    model.fit(X_rmo, y_rmo)
    # std_err = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # print(f'model.coef_: {model.coef_}')
    sorted_features_df = pd.DataFrame({'feature_importance_value': model.coef_[0]},
                                      index=attrib_columns).sort_values(by='feature_importance_value',
                                                                        ascending=False)
    sorted_features_df = sorted_features_df.loc[~(sorted_features_df.index.isin([1, 2, 3]))]
    # print(f'sorted_features_df: {sorted_features_df}')
    _, axes = plt.subplots(figsize=(14, 8))
    sns.barplot(data=sorted_features_df, x=sorted_features_df.index,
                y=sorted_features_df['feature_importance_value'], palette='viridis',
                edgecolor='k', ax=axes)
    axes.set_title('Feature importances', fontsize=18)
    plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=14)
    plt.setp(axes.get_yticklabels(), fontsize=14)
    plt.tight_layout()
    plt.grid(True, linestyle=':')
    plt.savefig('plots/'+filename, dpi=288, bbox_inches='tight')


# Feature importance plot, 'Lifetime People who have liked your Page and engaged with your post'
feature_importances_plot(SVR(kernel='linear', C=1e3),
                         performance_columns[-1],
                         'feature_importances_1.png')
# Feature importance plot, 'Lifetime Post Consumers'
feature_importances_plot(SVR(kernel='linear', C=1e3),
                         performance_columns[3],
                         'feature_importances_2.png')


def count_type_plot(col, filename):
    _, axes = plt.subplots(figsize=(10, 8))
    sns.barplot(data=fb_df, x='Type', y=col, palette='viridis', edgecolor='k', ax=axes)
    plt.tight_layout()
    axes.set_xticklabels(['Photo', 'Status', 'Link', 'Video'])
    # ['Photo', 'Status', 'Link', 'Video'], [1, 2, 3, 4]
    plt.grid(True, linestyle=':')
    plt.savefig('plots/'+filename, dpi=288, bbox_inches='tight')


# Count versus type plot, 'Lifetime People who have liked your Page and engaged with your post'
count_type_plot(performance_columns[-1], 'plot_type_1.png')
# Count versus type plot, ‘Lifetime Post Consumers'
count_type_plot(performance_columns[3], 'plot_type_2.png')
