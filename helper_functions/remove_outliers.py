#!/ usr/bin/env python

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class OutlierExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        '''
        Create a transformer to remove outliers. A threshold is set for selection
        criteria, and further arguments are passed to the LocalOutlierFactor class

        Keyword Args:
            neg_conf_val (float): The threshold for excluding samples with a lower
               negative outlier factor.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        '''

        self.threshold = kwargs.pop('neg_conf_val', -10.0)

        self.kwargs = kwargs

    def transform(self, X, y):
        '''
        Uses LocalOutlierFactor class to subselect data based on some threshold

        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        '''
        X = np.asarray(X)
        y = np.asarray(y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(X)
        return (X[lcf.negative_outlier_factor_ > self.threshold, :],
                y[lcf.negative_outlier_factor_ > self.threshold])

    def fit(self, *args, **kwargs):
        return self


class RemoveMetricOutliers(BaseEstimator, TransformerMixin):
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

        self.kwargs = kwargs

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, y):
        '''
        Uses StandardScaler to standardize y label values and retain only
        those that are within the sigma value threshold.

        Returns:
            ndarray: subsampled data
        '''
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        ss = StandardScaler(**self.kwargs)
        y_tr = ss.fit_transform(y)
        return (X[(np.abs(y_tr) < self.threshold).reshape(X.shape[0])],
                y[(np.abs(y_tr) < self.threshold).reshape(X.shape[0])])
