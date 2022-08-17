#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import DataPreparation
from sklearn.svm import SVR

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
data_prep = DataPreparation(data_file)
fb_na_tr = data_prep.transform()


def feature_importances_plot(model, col, filename, threshold=2.0):
    sns.set_context("talk")
    # 12 columns for preformance metrics
    diff_cols = fb_na_tr[0].shape[1] - len(data_prep.output_columns)
    X = fb_na_tr[0][:, 0:diff_cols].copy()
    y = fb_na_tr[0][:, col].copy()
    X_thr = X[(np.abs(y) < threshold)]
    y_thr = y[(np.abs(y) < threshold)]
    model.fit(X_thr, y_thr)
    features_df = pd.DataFrame({'contributing_features': model.coef_[0]},
                               index=fb_na_tr[1][0:45])
    sorted_features_df = features_df.sort_values(by='contributing_features', ascending=False)
    _, axes = plt.subplots(figsize=(14, 8))
    sns.barplot(data=sorted_features_df, x=sorted_features_df.index,
                y=sorted_features_df['contributing_features'], palette='Paired',
                edgecolor='k', ax=axes)
    plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45, fontsize=14)
    plt.setp(axes.get_yticklabels(), fontsize=14)
    axes.set_xlabel('Input Features', fontsize=14)
    axes.set_ylabel('Contributing Features', fontsize=14)
    axes.set_title('Input Features vs Most Contributing Features in SVR', fontsize=16)
    plt.tight_layout()
    plt.grid(True, linestyle=':')
    plt.savefig('plots/'+filename, dpi=288, bbox_inches='tight')


svr = SVR(kernel='linear', C=0.5)
# Feature importance plot using 'Lifetime Post reach by people who like your Page' metric
feature_importances_plot(svr, -7, 'feature_importances_1.png')
# Feature importance plot using 'Lifetime Post Total Impressions' metric
feature_importances_plot(svr, -11, 'feature_importances_2.png')
