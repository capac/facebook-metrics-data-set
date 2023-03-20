#! /usr/bin/env python3

import pandas as pd
# Scikit-Learn preprocessing classes
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer


class DataPreparation():
    '''data preparation class'''

    def __init__(self, data_file, columns=None):
        self.data_file = data_file
        self.fb_df = pd.read_csv(self.data_file, sep=',', na_values='NaN',
                                 dtype={'Category': 'object', 'Paid': 'object',
                                        'Post Month': 'object', 'Post Weekday': 'object',
                                        'Post Hour': 'object'})

        # shortened column names
        self.fb_df.rename(columns=columns, inplace=True)

        # fill NaN of 'Paid' column with mode
        self.fb_df.fillna(value=self.fb_df['Paid'].mode().values[0], inplace=True)
        # fill NaN of 'like' column with median
        self.fb_df.fillna(value=self.fb_df['Like'].median(), inplace=True)
        # fill NaN of 'share' column with median
        self.fb_df.fillna(value=self.fb_df['Share'].median(), inplace=True)

        # input columns (7)
        self.input_columns = self.fb_df.columns[0:7].tolist()
        # output columns (12)
        self.output_columns = self.fb_df.columns[7:19].tolist()

        # numerical columns: page total likes (1) and all performance metrics (12)
        self.numeric_cols = [self.input_columns[0]] + self.output_columns
        # categorical columns: type, category, post month, post weekday, post hour, paid (6)
        self.cat_cols = self.input_columns[1:]

    def transform(self):
        # standardization of data
        num_pipeline = make_pipeline(StandardScaler())
        cat_pipeline = make_pipeline(OneHotEncoder(sparse_output=False, drop='first'))

        full_pipeline = ColumnTransformer([
            # 6 columns of categorical data
            ('cat', cat_pipeline, self.cat_cols),
            # 13 columns of numerical data
            ('num', num_pipeline, self.numeric_cols),
        ], verbose_feature_names_out=False)
        # pipeline for feature transformations; returns NumPy
        # array which will be transformed into a dataframe
        return full_pipeline.fit_transform(self.fb_df), full_pipeline.get_feature_names_out()
