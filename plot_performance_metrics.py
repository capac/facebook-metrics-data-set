#! /usr/bin/env python3

from pathlib import Path
import pandas as pd
# matplotlib
import matplotlib.pyplot as plt
from data_preparation import DataPreparation

home_dir = Path.home()
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
data_prep = DataPreparation(data_file)
fb_df_tr = data_prep.transform()
columns = ['Lifetime Post Total Reach', 'Lifetime Post Total Impressions',
           'Lifetime Engaged Users', 'Lifetime Post Consumers',
           'Lifetime Post Consumptions',
           'Lifetime Post Impressions by people who have liked your Page',
           'Lifetime Post reach by people who like your Page',
           'Lifetime People who have liked your Page and engaged with your post',
           'comment', 'like', 'share', 'Total Interactions']

# plots
fb_df_tr = pd.DataFrame(fb_df_tr[:, 7:19], columns=columns)
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(13, 7))
# using 'Paired' colormap
cmap = plt.cm.Paired.colors
for ax, clm, col in zip(axes.flat, columns, cmap):
    ax.hist(fb_df_tr.loc[:, clm], edgecolor='k', alpha=0.9, color=col)
    ax.set_title(clm.capitalize())
plt.suptitle('Distribution of standardized performance metrics')
plt.tight_layout()
plt.savefig('plots/performance_metrics_hist.png', dpi=288, bbox_inches='tight')
