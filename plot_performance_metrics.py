#! /usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
from data_preparation import DataPreparation

home_dir = Path.home()
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/uci-ml-repository/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
data_prep = DataPreparation(data_file)
fb_na_tr = data_prep.transform()

# standardized performance metrics plot
fb_mod_df = pd.DataFrame(fb_na_tr[0], columns=fb_na_tr[1])
fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(8, 18), layout='constrained')
# using 'tab20' colormap
cmap = plt.cm.tab20.colors
for ax, clm, col in zip(axes.flat, fb_mod_df.columns.tolist()[-13:], cmap):
    ax.hist(fb_mod_df[clm], edgecolor='k', alpha=0.9, color=col, bins=60,)
    ax.set_title(clm.capitalize())
plt.suptitle("Standardized performance metrics and 'Page total likes'")
plt.savefig('plots/performance_metrics_hist.png', dpi=288, bbox_inches='tight')


# standardized logarithmic performance metrics and 'Page total likes'
fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(8, 18), layout='constrained')
# using 'tab20' colormap
cmap = plt.cm.tab20.colors
for ax, clm, col in zip(axes.flat, fb_mod_df.columns.tolist()[-13:], cmap):
    ax.hist(np.log(fb_mod_df[clm]+1), edgecolor='k', alpha=0.9, color=col, bins=60,)
    ax.set_title(clm.capitalize())
plt.suptitle("Standardized logarithmic performance metrics and 'Page total likes'")
plt.savefig('plots/performance_metrics_log_hist.png', dpi=288, bbox_inches='tight')


# standardized logarithmic performance metrics plot and different layout
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 8), layout='constrained')
# using 'tab20' colormap
cmap = plt.cm.tab20.colors
for ax, clm, col in zip(axes.flat, fb_mod_df.columns.tolist()[-12:], cmap):
    ax.hist(np.log(fb_mod_df.loc[:, clm]+1), edgecolor='k', alpha=0.9, color=col, bins=60,)
    ax.set_title(clm.capitalize())
plt.suptitle("Logarithmic standardized performance metrics")
plt.savefig('plots/performance_metrics_log_hist_2.png', dpi=288, bbox_inches='tight')
