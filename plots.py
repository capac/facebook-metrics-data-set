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


class FeatureRevelance():
    def __init__(self, data, model, perf_metric, threshold=2.0):
        self.data = data
        self.model = model
        self.perf_metric = perf_metric

        diff_cols = self.data[0].shape[1] - len(data_prep.output_columns)

        X = self.data[0][:, 0:diff_cols].copy()
        y = self.data[0][:, self.perf_metric].copy()
        X_thr = X[(np.abs(y) < threshold)]
        y_thr = y[(np.abs(y) < threshold)]
        model.fit(X_thr, y_thr)
        features_df = pd.DataFrame({'contributing_features': model.coef_[0]},
                                   index=self.data[1][0:diff_cols])
        self.sorted_features_df = features_df.sort_values(by='contributing_features',
                                                          ascending=False)
        # self.sorted_features_df.nlargest(top, columns='contributing_features')

    def feature_importances_plot(self, filename):
        # plot figure
        sns.set_context("talk")
        _, axes = plt.subplots(figsize=(14, 8))
        sns.barplot(data=self.sorted_features_df, x=self.sorted_features_df.index,
                    y=self.sorted_features_df['contributing_features'], palette='Paired',
                    edgecolor='k', ax=axes)
        plt.setp(axes.get_xticklabels(), ha="right", rotation_mode="anchor",
                 rotation=45, fontsize=14)
        plt.setp(axes.get_yticklabels(), fontsize=14)
        axes.set_xlabel('Input Features', fontsize=14)
        axes.set_ylabel('Contributing Features', fontsize=14)
        axes.set_title('Input Features vs Most Contributing Features in SVR', fontsize=16)
        plt.tight_layout()
        plt.grid(True, linestyle=':')
        plt.savefig('plots/'+filename, dpi=288, bbox_inches='tight')


svr = SVR(kernel='linear', C=0.5)
# Feature importance plot using 'Lifetime Post reach by people who like your Page' metric
rel_feats_7 = FeatureRevelance(fb_na_tr, svr, -7)
rel_feats_7.feature_importances_plot('feature_importances_1.png')
# Feature importance plot using 'Lifetime Post Total Impressions' metric
rel_feats_11 = FeatureRevelance(fb_na_tr, svr, -11)
rel_feats_11.feature_importances_plot('feature_importances_2.png')
