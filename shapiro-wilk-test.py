#! /usr/bin/env python3

from pathlib import Path
from data_preparation import DataPreparation
from scipy.stats import shapiro

home_dir = Path.home()
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
data_prep = DataPreparation(data_file)
fb_df_tr = data_prep.transform()

# test calculation and saving output to file
with open(work_dir / 'shapiro-wilk-test-results.txt', 'w') as f:
    for col, name in zip(range(7, 15), data_prep.output_columns):
        result = shapiro(fb_df_tr[:, col])
        f.writelines(f'''Result for '{name}': \n{(result)}\n\n''')
    f.writelines('\n')
    print('Done!')
