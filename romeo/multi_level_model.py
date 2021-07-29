import numpy as np
import casadi as ca
import pandas as pd
import os
import contextlib
from scipy import stats

# from sklearn.base import BaseEstimator#, ClassifierMixin, TransformerMixin
from linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class LinearMixedRegression(LinearRegression):

    def __init__(self, fit_intercept = True,
                 normalize = False,
                 copy_X = True,
                 positive = False):
        """Constructor. It runs every instance is created"""

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive

    def fit(self, X, y, sample_weight=None, verbose=True):
        """docstring"""

        # TODO: change the fit method to match the LMM framework

        if isinstance(X, pd.DataFrame):
            column_names = X.columns.tolist()
        else:
            column_names = list(range(X.shape[1]))

        self.predictors = X
        self.target = y

        if self.copy_X:
            X = X.copy()

        X, y = check_X_y(X, y, accept_sparse=True)

        if self.fit_intercept:
            # # add 1 col using broadcasting
            # X["intercept"] = 1
            # # move intercept col at the first column
            # cols = X.columns.tolist()
            # cols = cols[-1:] + cols[:-1]
            # X = X[cols]
            X = np.c_[np.ones(X.shape[0]), X]

        (N, ntheta) = X.shape
        theta = ca.SX.sym("theta", ntheta)

        # create residual
        e = y - ca.mtimes(X, theta)

        # create optimization problem (x: optimization parameter, f: cost function)
        nlp = {"x": theta, "f": 0.5 * ca.dot(e, e)}

        # solve opt
        solver = ca.nlpsol("ols", "ipopt", nlp)
        if verbose:
            sol = solver(x0=np.zeros(ntheta))
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                sol = solver(x0=np.zeros(ntheta))

        self.is_fitted_ = True

        if self.fit_intercept:
            self.intercept_ = sol["x"].full().ravel()[0]
            self.coef_ = sol["x"].full().ravel()[1:]
            summary_index = ["intercept"] + column_names
        else:
            self.intercept_ = 0
            self.coef_ = sol["x"].full().ravel()
            summary_index = column_names

        self.params = sol["x"].full().ravel()

        tmp_x = self.intercept_ * X
        cov_mat = np.linalg.inv(np.matmul(tmp_x.transpose(1, 0), tmp_x))
        self.bse = np.sqrt(np.diag(cov_mat))
        self.tvalues = sol["x"].full().ravel() / self.bse
        self.df_resid = X.shape[0] - X.shape[1]
        self.df_model = X.shape[1] - 1
        self.pvalues = stats.t.sf(np.abs(self.tvalues), self.df_resid) * 2
        self.summary_ = (pd.DataFrame(data={"coef": sol["x"].full().ravel(),
                                            "std_err": self.bse,
                                            "t": self.tvalues,
                                            "P>|t|": self.pvalues},
                                      index=summary_index)
                         .join(pd.DataFrame(self.conf_int(), columns=["[0.025", "0.975]"],
                                            index=summary_index))
                         )

        self.resid = self.resid()
        self.ssr = self.ssr()
        self.uncentered_tss = self.uncentered_tss()
        self.rsquared = self.rsquared()
        self.nobs = X.shape[0]
        self.rsquared_adj = self.rsquared_adj()
        self.ess = self.ess()
        self.mse_model = self.mse_model()
        self.mse_resid = self.mse_resid()
        self.mse_total = self.mse_total()
        self.fvalue = self.fvalue()
        self.f_pvalue = self.f_pvalue()
        self.llf = self.loglike()
        self.aic = self.aic()
        self.bic = self.bic()

        self.fit_evaluation_ = (pd.DataFrame(data={
            "r_squared": self.rsquared,
            "r_squared_adj": self.rsquared_adj,
            "f_statistic": self.fvalue,
            "f_statistic_pvalue": self.f_pvalue,
            "log_likelihood": self.llf,
            "AIC": self.aic,
            "BIC": self.bic,
        },
            index=["model_evaluation"])
                                )

        return self

    def predict(self, X:pd.DataFrame):
        """y = a*x + b"""
        pass




# executable for stand-alone program
if __name__ == "__main__":
    pass