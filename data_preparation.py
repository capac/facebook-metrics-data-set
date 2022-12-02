#! /usr/bin/env python3

import pandas as pd
# Scikit-Learn preprocessing classes
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataPreparation():
    '''data preparation class'''

    def __init__(self, data_file):
        self.data_file = data_file
        self.fb_df = pd.read_csv(self.data_file, sep=';', na_values='NaN')

        # shortened column names
        self.fb_df.rename(columns={
                          'Lifetime Post Total Reach':
                          'LT Post Total Reach',
                          'Lifetime Post Total Impressions':
                          'LT Post Total Imp',
                          'Lifetime Engaged Users':
                          'LT Engd Users',
                          'Lifetime Post Consumers':
                          'LT Post Consumers',
                          'Lifetime Post Consumptions':
                          'LT Post Consump',
                          'Lifetime Post Impressions by people who have liked your Page':
                          'LT Post Imp + Liked Page',
                          'Lifetime Post reach by people who like your Page':
                          'LT Post Reach + Liked Page',
                          'Lifetime People who have liked your Page and engaged with your post':
                          'LT People + Engd Post',
                          'comment':
                          'Comment',
                          'like':
                          'Like',
                          'share':
                          'Share',
                          'Total Interactions':
                          'Total Int'
                          }, inplace=True)

        # fill NaN of 'Paid' column with mode
        self.fb_df.fillna(value=self.fb_df['Paid'].mode().values[0], inplace=True)
        # fill NaN of 'like' column with median
        self.fb_df.fillna(value=self.fb_df['Like'].median(), inplace=True)
        # fill NaN of 'share' column with median
        self.fb_df.fillna(value=self.fb_df['Share'].median(), inplace=True)

        # input columns
        self.input_columns = self.fb_df.columns[0:7].tolist()
        # output columns
        self.output_columns = self.fb_df.columns[7:19].tolist()

        # numerical columns: all posts and performance metrics
        self.numeric_cols = [self.input_columns[0]] + self.output_columns
        # categorical columns: type, category, hour, month, day, paid
        self.cat_cols = self.input_columns[1:7]

    def transform(self):
        # standardization of data
        num_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])
        full_pipeline = ColumnTransformer([
            # 6 columns for categorical data
            ('ord_enc', OrdinalEncoder(), self.cat_cols),
            # 12 columns for numerical data
            ('num', num_pipeline, self.numeric_cols),
        ])
        # application for feature transformation pipeline
        input_fb_df = self.fb_df.copy()
        return full_pipeline.fit_transform(input_fb_df)
