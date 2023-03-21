#! /usr/bin/env python3

from pathlib import Path
from data_preparation import DataPreparation
from scipy.stats import shapiro
import pandas as pd
import numpy as np

home_dir = Path.home()
work_dir = home_dir / 'Programming/Python/machine-learning-exercises/uci-ml-repository/facebook-metrics-data-set'
data_file = work_dir / 'data/dataset_Facebook.csv'

# data preparation
data_prep = DataPreparation(data_file)
fb_na_tr = data_prep.transform()
fb_mod_df = pd.DataFrame(fb_na_tr[0], columns=fb_na_tr[1])

# test calculation and saving output to file
# https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
with open(work_dir / 'shapiro-wilk-test-results.txt', 'w') as f:
    for col_name in fb_na_tr[1].tolist()[-12:]:
        result = shapiro(np.log(fb_mod_df[col_name]+1))
        f.writelines(f'''Result for '{col_name.capitalize()}': \n{(result)}\n\n''')
    f.writelines('\n')
    print('Done!')
