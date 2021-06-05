import numpy as np
import casadi as ca
import pandas as pd

class LinearMixedRegression:

    def __init__(self, fit_intercept = True,
                 normalize = False,
                 copy_X = True,
                 positive = False):
        """Constructor. It runs every instance is created"""

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive


    def fit(self, X, y, sample_weight=None):
        """docstring"""

        # TODO: self._preprocess_data
        # X, y, X_offset, y_offset, X_scale = self._preprocess_data(
        #    X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
        #    copy=self.copy_X, sample_weight=sample_weight,
        #    return_mean=True)

        pass

    def predict(self, X:pd.DataFrame):
        """y = a*x + b"""
        pass




# executable for stand-alone program
if __name__ == "__main__":
    pass