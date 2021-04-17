#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

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
    sorted_features_df = pd.Series(model.coef_[0], index=X.columns)
    sorted_features_df.sort_values(inplace=True, ascending=False)
    sorted_features_df = sorted_features_df.reset_index()
    sorted_features_df.rename(columns={'index': 'feature_importance', 0: 'feature_importance_value'}, inplace=True)
    _, axes = plt.subplots(figsize=(14, 8))
    sns.barplot(data=sorted_features_df, x=sorted_features_df['feature_importance'],
                y=sorted_features_df['feature_importance_value'],
                palette='viridis', edgecolor='k', ax=axes)
    axes.set_title('Feature importances', fontsize=18)
    axes.set_xlabel('Type')
    axes.set_ylabel('Value')
    plt.tight_layout()
    plt.grid(True, linestyle=':')
    plt.savefig('plots/'+filename, dpi=288, bbox_inches='tight')


# Feature importance plot, 'Lifetime People who have liked your Page and engaged with your post'
feature_importances_plot(SVR(kernel='linear', C=1e3),
                         performance_columns[-1],
                         'feature_importances_SVR_1.png')
# Feature importance plot, 'Lifetime Post Consumers'
feature_importances_plot(SVR(kernel='linear', C=1e3),
                         performance_columns[3],
                         'feature_importances_SVR_2.png')
