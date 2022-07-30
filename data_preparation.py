#! /usr/bin/env python3

import pandas as pd
# Scikit-Learn preprocessing classes
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class DataPreparation():
    '''data preparation class'''

    def __init__(self, data_file):
        self.data_file = data_file
        self.fb_df = pd.read_csv(self.data_file, sep=';', na_values='NaN')

        # column distinction
        self.selected_columns = list(self.fb_df.columns[0:15])
        # input column data type
        self.input_columns = self.selected_columns[0:7]
        self.output_columns = self.selected_columns[7:15]
        # numerical and categorical columns
        self.numeric_cols = [self.input_columns[0]] + self.input_columns[3:6] + self.output_columns
        self.cat_onehot_cols = self.input_columns[1:3] + [self.input_columns[6]]

    def transform(self):
        # substitution of NA (just for the one in 'Paid') and standardization of data
        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent', missing_values=pd.NA)),
            ('std_scaler', StandardScaler()),
        ])
        full_pipeline = ColumnTransformer([
            # 6 columns for categorical data
            ('cat_onehot', OneHotEncoder(sparse=False, drop='first'), self.cat_onehot_cols),
            # 12 columns for numerical data
            ('num', num_pipeline, self.numeric_cols),
        ])
        # application for feature transformation pipeline
        input_fb_df = self.fb_df[self.selected_columns].copy()
        return full_pipeline.fit_transform(input_fb_df)
