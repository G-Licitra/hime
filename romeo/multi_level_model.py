import numpy as np
import casadi as ca
import pandas as pd
import os
import contextlib
from scipy import stats

# from sklearn.base import BaseEstimator#, ClassifierMixin, TransformerMixin
from .linear_model import LinearRegression
from .logistic_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .utils import sigmoid

flatten = lambda l: [item for sublist in l for item in sublist]

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

    def fit(self, X, y, sample_weight=None, verbose=True, kwargs=dict()):
        """docstring"""

        # TODO: change the fit method to match the LMM framework

        if isinstance(X, pd.DataFrame):
            column_names = X.columns.tolist()
        else:
            column_names = list(range(X.shape[1]))

        grouping_var = kwargs["grouping_var"]
        self.u = X[grouping_var].sort_values().unique()

        z_list = list()
        for cluster_group in self.u:
            tmp = X.loc[X[grouping_var] == cluster_group, grouping_var]
            z_list.append(
                np.pad(tmp, (tmp.index.tolist()[0], X[grouping_var].shape[0] - tmp.index.tolist()[-1] + 1)))

        possible_Z_df = pd.DataFrame(np.vstack(z_list).T)
        binary_df = np.where(possible_Z_df > 0, 1, possible_Z_df)
        self.Z = binary_df[0: X.shape[0]]

        fixed_effect = kwargs["fixed_effect"]
        X = X[[fixed_effect]]

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

        ntheta_fixed = X.shape[1]  # (time,)
        ntheta_random = len(self.u)  # (amount of pigs groups)
        theta_fixed = ca.SX.sym("theta_fixed", ntheta_fixed)
        theta_random = ca.SX.sym("theta_random", ntheta_random)

        model_lmm = ca.mtimes(X, theta_fixed) + ca.mtimes(self.Z, theta_random)

        # create residual
        e = y.reshape(-1, 1) - model_lmm

        # create optimization problem (x: optimization parameter, f: cost function)
        nlp = {"x": ca.vertcat(theta_fixed, theta_random), "f": 0.5 * ca.dot(e, e), }

        # solve opt
        solver = ca.nlpsol("ols", "ipopt", nlp)

        if verbose:
            sol = solver(x0=np.zeros(ntheta_fixed + ntheta_random))
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                sol = solver(x0=np.zeros(ntheta_fixed + ntheta_random))

        self.is_fitted_ = True

        if self.fit_intercept:
            self.intercept_ = sol["x"].full().ravel()[0]
            self.coef_ = sol["x"].full().ravel()[1:]
            summary_index = ["intercept"] + column_names[:-1] + self.u.tolist()
        else:
            self.intercept_ = 0
            self.coef_ = sol["x"].full().ravel()
            summary_index = column_names

        self.params = sol["x"].full().ravel()

        # TODO: Take the intercept into account when calculating the fitted values
        theta_est = sol["x"]
        results = flatten(theta_est.toarray().tolist())
        Zu = np.dot(self.Z, results[2:])
        XB = X * results[1]
        self.fitted_values = pd.DataFrame(XB)[1].add(pd.Series(Zu), axis=0)

        # tmp_x = self.intercept_ * X
        # cov_mat = np.linalg.inv(np.matmul(tmp_x.transpose(1, 0), tmp_x))
        # self.bse = np.sqrt(np.diag(cov_mat))
        # self.tvalues = sol["x"].full().ravel() / self.bse
        # self.df_resid = X.shape[0] - X.shape[1]
        # self.df_model = X.shape[1] - 1
        # self.pvalues = stats.t.sf(np.abs(self.tvalues), self.df_resid) * 2
        # self.summary_ = (pd.DataFrame(data={"coef": sol["x"].full().ravel(),
        #                                     "std_err": self.bse,
        #                                     "t": self.tvalues,
        #                                     "P>|t|": self.pvalues},
        #                               index=summary_index)
        #                  .join(pd.DataFrame(self.conf_int(), columns=["[0.025", "0.975]"],
        #                                     index=summary_index))
        #                  )
        self.summary_ = (pd.DataFrame(data={"coef": sol["x"].full().ravel(),
                                            # "std_err": self.bse,
                                            # "t": self.tvalues,
                                            # "P>|t|": self.pvalues
                                            },
                                      index=summary_index
                                      )
                         )
        #
        # self.resid = self.resid()
        # self.ssr = self.ssr()
        # self.uncentered_tss = self.uncentered_tss()
        # self.rsquared = self.rsquared()
        # self.nobs = X.shape[0]
        # self.rsquared_adj = self.rsquared_adj()
        # self.ess = self.ess()
        # self.mse_model = self.mse_model()
        # self.mse_resid = self.mse_resid()
        # self.mse_total = self.mse_total()
        # self.fvalue = self.fvalue()
        # self.f_pvalue = self.f_pvalue()
        # self.llf = self.loglike()
        # self.aic = self.aic()
        # self.bic = self.bic()
        #
        # self.fit_evaluation_ = (pd.DataFrame(data={
        #     "r_squared": self.rsquared,
        #     "r_squared_adj": self.rsquared_adj,
        #     "f_statistic": self.fvalue,
        #     "f_statistic_pvalue": self.f_pvalue,
        #     "log_likelihood": self.llf,
        #     "AIC": self.aic,
        #     "BIC": self.bic,
        # },
        #     index=["model_evaluation"])
        #                         )

        return self

    def predict(self, X:pd.DataFrame):
        """y = a*x + b"""
        pass

class LogisticMixedRegression(LogisticRegression):

    def __init__(self, fit_intercept = True,
                 normalize = False,
                 copy_X = True,
                 positive = False):
        """Constructor. It runs every instance is created"""

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive

    def fit(self, X, y, sample_weight=None, verbose=True, kwargs=dict()):
        """docstring"""

        # TODO: change the fit method to match the LMM framework

        if isinstance(X, pd.DataFrame):
            column_names = X.columns.tolist()
        else:
            column_names = list(range(X.shape[1]))

        grouping_var = kwargs["grouping_var"]
        self.u = X[grouping_var].sort_values().unique()

        z_list = list()
        for cluster_group in self.u:
            tmp = X.loc[X[grouping_var] == cluster_group, grouping_var]
            z_list.append(
                np.pad(tmp, (tmp.index.tolist()[0], X[grouping_var].shape[0] - tmp.index.tolist()[-1] + 1)))

        possible_Z_df = pd.DataFrame(np.vstack(z_list).T)
        binary_df = np.where(possible_Z_df > 0, 1, possible_Z_df)
        self.Z = binary_df[0: X.shape[0]]

        fixed_effect = kwargs["fixed_effect"]
        X = X[[fixed_effect]]

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

        ntheta_fixed = X.shape[1]  # (time,)
        ntheta_random = len(self.u)  # (amount of pigs groups)
        theta_fixed = ca.SX.sym("theta_fixed", ntheta_fixed)
        theta_random = ca.SX.sym("theta_random", ntheta_random)

        model_lmm = ca.mtimes(X, theta_fixed) + ca.mtimes(self.Z, theta_random)

        (N, ntheta) = X.shape

        A = sigmoid(model_lmm)  # compute activation

        # might be 1 / 2*N
        cost = -1 / N * ca.sum1(y * ca.log(A) + (1 - y) * ca.log(1 - A))

        # create residual
        # e = y.reshape(-1, 1) - model_lmm

        # create optimization problem (x: optimization parameter, f: cost function)
        # nlp = {"x": ca.vertcat(theta_fixed, theta_random), "f": 0.5 * ca.dot(e, e), }

        # create optimization problem (x: optimization parameter, f: cost function)
        nlp = {"x": ca.vertcat(theta_fixed, theta_random), "f": cost, }

        # solve opt
        solver = ca.nlpsol("ols", "ipopt", nlp)

        if verbose:
            sol = solver(x0=np.zeros(ntheta_fixed + ntheta_random))
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                sol = solver(x0=np.zeros(ntheta_fixed + ntheta_random))

        self.is_fitted_ = True

        if self.fit_intercept:
            self.intercept_ = sol["x"].full().ravel()[0]
            self.coef_ = sol["x"].full().ravel()[1:]
            summary_index = ["intercept"] + column_names[:-1] + self.u.tolist()
        else:
            self.intercept_ = 0
            self.coef_ = sol["x"].full().ravel()
            summary_index = column_names

        self.params = sol["x"].full().ravel()

        # TODO: Take the intercept into account when calculating the fitted values
        theta_est = sol["x"]
        results = flatten(theta_est.toarray().tolist())
        Zu = np.dot(self.Z, results[2:])
        XB = X * results[1]
        self.fitted_values = pd.DataFrame(XB)[1].add(pd.Series(Zu), axis=0)

        # tmp_x = self.intercept_ * X
        # cov_mat = np.linalg.inv(np.matmul(tmp_x.transpose(1, 0), tmp_x))
        # self.bse = np.sqrt(np.diag(cov_mat))
        # self.tvalues = sol["x"].full().ravel() / self.bse
        # self.df_resid = X.shape[0] - X.shape[1]
        # self.df_model = X.shape[1] - 1
        # self.pvalues = stats.t.sf(np.abs(self.tvalues), self.df_resid) * 2
        # self.summary_ = (pd.DataFrame(data={"coef": sol["x"].full().ravel(),
        #                                     "std_err": self.bse,
        #                                     "t": self.tvalues,
        #                                     "P>|t|": self.pvalues},
        #                               index=summary_index)
        #                  .join(pd.DataFrame(self.conf_int(), columns=["[0.025", "0.975]"],
        #                                     index=summary_index))
        #                  )
        self.summary_ = (pd.DataFrame(data={"coef": sol["x"].full().ravel(),
                                            # "std_err": self.bse,
                                            # "t": self.tvalues,
                                            # "P>|t|": self.pvalues
                                            },
                                      index=summary_index
                                      )
                         )
        #
        # self.resid = self.resid()
        # self.ssr = self.ssr()
        # self.uncentered_tss = self.uncentered_tss()
        # self.rsquared = self.rsquared()
        # self.nobs = X.shape[0]
        # self.rsquared_adj = self.rsquared_adj()
        # self.ess = self.ess()
        # self.mse_model = self.mse_model()
        # self.mse_resid = self.mse_resid()
        # self.mse_total = self.mse_total()
        # self.fvalue = self.fvalue()
        # self.f_pvalue = self.f_pvalue()
        # self.llf = self.loglike()
        # self.aic = self.aic()
        # self.bic = self.bic()
        #
        # self.fit_evaluation_ = (pd.DataFrame(data={
        #     "r_squared": self.rsquared,
        #     "r_squared_adj": self.rsquared_adj,
        #     "f_statistic": self.fvalue,
        #     "f_statistic_pvalue": self.f_pvalue,
        #     "log_likelihood": self.llf,
        #     "AIC": self.aic,
        #     "BIC": self.bic,
        # },
        #     index=["model_evaluation"])
        #                         )

        return self

    def predict(self, X:pd.DataFrame):
        """y = a*x + b"""
        pass

