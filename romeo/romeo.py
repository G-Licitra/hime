from __future__ import division

import logging

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression


logger = logging.getLogger(__name__)


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
