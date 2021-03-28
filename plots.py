#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

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


def feature_importances_plot(model, col, filename):
    sns.set_context("talk")
    X = fb_df_prepared
    y = fb_df[col].values
    model.fit(X, y)
    std_err = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    sorted_features_df = pd.DataFrame({'feature_importance_value': model.feature_importances_,
                                      'std_err': std_err},
                                      index=X.columns).sort_values(by='feature_importance_value',
                                                                   ascending=False)
    _, axes = plt.subplots(figsize=(14, 8))
    sns.barplot(data=sorted_features_df, x=sorted_features_df.index, y=sorted_features_df['feature_importance_value'],
                palette='viridis', edgecolor='k', ax=axes, yerr=sorted_features_df['std_err'])
    axes.set_title('Feature importances', fontsize=18)
    plt.tight_layout()
    plt.grid(True, linestyle=':')
    plt.savefig('plots/'+filename, dpi=288, bbox_inches='tight')


# Feature importance plot, 'Lifetime People who have liked your Page and engaged with your post'
feature_importances_plot(RandomForestRegressor(),
                         performance_columns[-1],
                         'feature_importances_1.png')
# Feature importance plot, 'Lifetime Post Consumers'
feature_importances_plot(RandomForestRegressor(),
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
