#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
# matplotlib
import matplotlib.pyplot as plt
from data_preparation import DataPreparation

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
data_prep = DataPreparation(data_file)
fb_df_tr = data_prep.transform()

# plots
fb_df_tr = pd.DataFrame(fb_df_tr[:, 7:15], columns=data_prep.output_columns)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(13, 7))
# using 'Paired' colormap
cmap = plt.cm.Paired.colors
for ax, clm, col in zip(axes.flat, data_prep.output_columns, cmap):
    ax.hist(fb_df_tr.loc[:, clm], edgecolor='k', alpha=0.9, color=col)
    ax.set_title(clm)
plt.suptitle('Distribution of standardized performance metrics')
plt.tight_layout()
plt.savefig('plots/performance_metrics_hist.png', dpi=288, bbox_inches='tight')
