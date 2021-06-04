from __future__ import division

import logging

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression


logger = logging.getLogger(__name__)

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# from sklearn.utils.multiclass import unique_labels
# from sklearn.metrics import euclidean_distances


class TempROMEO(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    Examples
    --------
    >>> from romeo.romeo import TempROMEO
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TempROMEO()
    >>> estimator.fit(X, y)
    >>> TempROMEO(demo_param='demo_param')
    """
    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64)


class ROMEO:
    def __init__(self, model=None):

        if model is None:
            span = LinearRegression(n_jobs=-1)

        self.trained_model = None

    def predict(self, data: pd.DataFrame, formula: str):

        if type(formula) != str:
            raise TypeError('formula must be a string, e.g., "DV = IV1 + IV2" .')

        if self.trained_model is None:
            raise NotFittedError(
                """The instance is not fitted yet. Call 'fit' before using this method"""
            )


