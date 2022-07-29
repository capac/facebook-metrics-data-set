#! /usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

home = os.environ['HOME']
home_dir = Path(home)
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

fb_df = pd.read_csv(data_file, sep=';', na_values='NaN')

input_columns = list(fb_df.columns[0:7])
performance_columns = list(fb_df.columns[7:15])

numeric_columns = list(set(fb_df.columns[0:7]) - set(fb_df.columns[1:3]))
category_columns = list(fb_df.columns[1:3])

fb_df['Type'] = fb_df['Type'].replace(['Photo', 'Status', 'Link', 'Video'], [1, 2, 3, 4])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# fb_df_num_tr = num_pipeline.fit_transform(fb_df[numeric_columns])
# fb_df_num = pd.DataFrame(fb_df_num_tr, columns=numeric_columns)

fb_df_num = fb_df[numeric_columns]
fb_df_prepared = pd.concat([fb_df_num, fb_df[['Type', 'Category']]], axis=1)

label_font_size = 14
g = sns.PairGrid(fb_df_prepared, hue='Type', palette='tab10', height=3.0, aspect=1.2)
g = g.map_diag(plt.hist)
# g = g.map_offdiag(sns.regplot, scatter=False)
g = g.map_offdiag(plt.scatter, s=60, alpha=0.6)
g = g.add_legend()
g._legend.set_title('Type', prop={'size': label_font_size})
for txt, lb in zip(g._legend.texts, ['Photo', 'Status', 'Link', 'Video']):
    txt.set_text(lb)
    txt.set_fontsize(label_font_size)

xlabels, ylabels = [], []

for ax in g.axes[-1, :]:
    xlabel = ax.xaxis.get_label_text()
    xlabels.append(xlabel)
for ax in g.axes[:, 0]:
    ylabel = ax.yaxis.get_label_text()
    ylabels.append(ylabel)
for i in range(len(xlabels)):
    for j in range(len(ylabels)):
        g.axes[j, i].xaxis.set_label_text(xlabels[i])
        g.axes[j, i].xaxis.label.set_size(label_font_size)
        g.axes[j, i].tick_params(axis='x', which='major', labelsize=label_font_size)
        g.axes[j, i].yaxis.set_label_text(ylabels[j])
        g.axes[j, i].yaxis.label.set_size(label_font_size)
        g.axes[j, i].tick_params(axis='y', which='major', labelsize=label_font_size)

plt.tight_layout(rect=(0, 0, 0.92, 1))
plt.savefig(work_dir / 'plots/pairplot.png', dpi=288, bbox_inches='tight')
