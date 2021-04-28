#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from helper_functions.remove_outliers import RemoveMetricOutliers
from sklearn.svm import SVR

pd.set_option('display.max_colwidth', None)

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
fb_df = pd.read_csv(data_file, sep=';')

# column distinction
selected_columns = list(fb_df.columns[0:15])
input_columns = list(fb_df.columns[0:7])
performance_columns = list(fb_df.columns[7:15])

# column data type
numeric_cols = [fb_df.columns[0]]
cat_ordenc_cols = list(fb_df.columns[1:7])

# rename the two categorical values in 'Type' and 'Category'
# fb_df['Category'] = fb_df['Category'].replace([1, 2, 3], ['Category-1', 'Category-2', 'Category-3'])
# fb_df['Paid'] = fb_df['Paid'].replace([0.0, 1.0], ['Not Paid', 'Paid'])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('min_max_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, numeric_cols),
    ('cat_labelenc', OrdinalEncoder(), cat_ordenc_cols),
])

# drop NaNs from data frame
selected_fb_df = fb_df[selected_columns].copy()
selected_fb_df.dropna(inplace=True)

# application of feature transformation pipeline
fb_df_num_tr = full_pipeline.fit_transform(selected_fb_df)

attrib_columns = numeric_cols + cat_ordenc_cols
# print(f'attrib_columns: {attrib_columns}')
# print(f'input_columns: {input_columns}')


def feature_importances_plot(model, col, filename):
    sns.set_context("talk")
    X = fb_df_num_tr
    y = selected_fb_df[col].values
    rmo = RemoveMetricOutliers(sigma=2.0)
    X_rmo, y_rmo = rmo.transform(X, y)
    y_rmo = y_rmo.ravel()
    model.fit(X_rmo, y_rmo)
    # print(f'model.scores_: {model.scores_}')
    sorted_features_df = pd.DataFrame({'contributing_features': model.coef_[0]},
                                      index=attrib_columns).sort_values(by='contributing_features',
                                                                        ascending=False)
    # print(f'sorted_features_df:\n{sorted_features_df}')
    # sorted_features_df = sorted_features_df.loc[~(sorted_features_df.index.isin([1, 2, 3]))]
    _, axes = plt.subplots(figsize=(14, 8))
    sns.barplot(data=sorted_features_df, x=sorted_features_df.index,
                y=sorted_features_df['contributing_features'], palette='viridis',
                edgecolor='k', ax=axes)
    plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=14)
    plt.setp(axes.get_yticklabels(), fontsize=14)
    axes.set_xlabel('Input Features', fontsize=14)
    axes.set_ylabel('Contributing Features', fontsize=14)
    axes.set_title('Input Features vs Most Contributing Features in SVR', fontsize=16)
    plt.tight_layout()
    plt.grid(True, linestyle=':')
    plt.savefig('plots/'+filename, dpi=288, bbox_inches='tight')


svr = SVR(kernel='linear', C=1e3)
# Feature importance plot, 'Lifetime People who have liked your Page and engaged with your post'
feature_importances_plot(svr, performance_columns[-1], 'feature_importances_1.png')
# Feature importance plot, 'Lifetime Post Consumers'
feature_importances_plot(svr, performance_columns[3], 'feature_importances_2.png')


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
