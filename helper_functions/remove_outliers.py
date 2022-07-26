#!/ usr/bin/env python

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# https://stackoverflow.com/questions/52346725/can-i-add-outlier-detection-and-removal-to-scikit-learn-pipeline
class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        '''
        Removes outliers in performance metrics by standardizing samples
        using StandardScaler. A sigma threshold is set for outlier removal.

        Keyword Args:
            sigma (float): The threshold for excluding samples such that the absolute
            value of the sample is less than the sigma threshold.

        Returns:
            ndarray: subsampled data
        '''
        self.threshold = kwargs.pop('sigma', 2.0)

    def transform(self, X, y):
        '''
        Uses StandardScaler to standardize y label values and retain only
        those that are within the sigma value threshold.

        Returns:
            ndarray: subsampled data
        '''
        y = y.reshape(-1, 1)
        ss = StandardScaler()
        y_tr = ss.fit_transform(y)
        return (X[(np.abs(y_tr) <= self.threshold).reshape(X.shape[0])],
                y[(np.abs(y_tr) <= self.threshold).reshape(X.shape[0])])
