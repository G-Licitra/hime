import numpy as np
import casadi as ca
import pandas as pd
import os
import contextlib
from scipy import stats

from sklearn.base import BaseEstimator#, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LinearRegression(BaseEstimator):

    def __init__(self, fit_intercept = True,
                 normalize = False,
                 copy_X = True,
                 positive = False):
        """Constructor. It runs every instance is created"""

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive
        self.is_fitted_ = False


    def fit(self, X, y, sample_weight=None, verbose=True):
        """docstring"""

        # TODO: self._preprocess_data
        # X, y, X_offset, y_offset, X_scale = self._preprocess_data(
        #    X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
        #    copy=self.copy_X, sample_weight=sample_weight,
        #    return_mean=True)

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
            X = np.c_[np.ones(X.shape[0]), X]

        (N, ntheta) = X.shape
        theta = ca.SX.sym("theta", ntheta)

        # create residual
        e = y - ca.mtimes(X, theta)


        # constrains on coefficient >=0
        lbx = np.zeros(ntheta) if self.positive else -np.inf*np.ones(ntheta)

        # create optimization problem (x: optimization parameter, f: cost function)
        nlp = {"x": theta, "f": 0.5 * ca.dot(e, e)}

        # fit model
        solver = ca.nlpsol("ols", "ipopt", nlp)
        if verbose:
            sol = solver(x0=np.zeros(ntheta), lbx=lbx)
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                sol = solver(x0=np.zeros(ntheta), lbx=lbx)

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

        index_names = None

        if isinstance(X, pd.DataFrame):
            index_names = X.index.tolist()

        X = check_array(X, accept_sparse=True)

        check_is_fitted(self, 'is_fitted_')

        # y_pred = X.dot(self.coef_.filter(items=feature_list, axis=0))
        y_pred = X.dot(self.coef_)

        if self.fit_intercept:
            y_pred += self.intercept_

        if index_names:
            y_pred = pd.DataFrame(y_pred,
                                  index=index_names,
                                  columns=["y_pred"])

        return y_pred

    def conf_int(self, alpha=.05, cols=None):
        """
        Construct confidence interval for the fitted parameters.
        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval. The default
            `alpha` = .05 returns a 95% confidence interval.
        cols : array_like, optional
            Specifies which confidence intervals to return.
        Returns
        -------
        array_like
            Each row contains [lower, upper] limits of the confidence interval
            for the corresponding parameter. The first column contains all
            lower, the second column contains all upper limits.
        Notes
        -----
        The confidence interval is based on the standard normal distribution
        if self.use_t is False. If self.use_t is True, then uses a Student's t
        with self.df_resid_inference (or self.df_resid if df_resid_inference is
        not defined) degrees of freedom.
        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> results.conf_int()
        array([[-5496529.48322745, -1467987.78596704],
               [    -177.02903529,      207.15277984],
               [      -0.1115811 ,        0.03994274],
               [      -3.12506664,       -0.91539297],
               [      -1.5179487 ,       -0.54850503],
               [      -0.56251721,        0.460309  ],
               [     798.7875153 ,     2859.51541392]])
        >>> results.conf_int(cols=(2,3))
        array([[-0.1115811 ,  0.03994274],
               [-3.12506664, -0.91539297]])
        """

        def lzip(*args, **kwargs):
            return list(zip(*args, **kwargs))

        bse = self.bse

        # if self.use_t:
        #     dist = stats.t
        #     df_resid = getattr(self, 'df_resid_inference', self.df_resid)
        #     q = dist.ppf(1 - alpha / 2, df_resid)
        # else:
        #     dist = stats.norm
        #     q = dist.ppf(1 - alpha / 2)

        dist = stats.t
        df_resid = self.df_resid
        q = dist.ppf(1 - alpha / 2, df_resid)

        # dist = stats.norm
        # q = dist.ppf(1 - alpha / 2)

        params = self.params
        lower = params - q * bse
        upper = params + q * bse
        if cols is not None:
            cols = np.asarray(cols)
            lower = lower[cols]
            upper = upper[cols]
        return np.asarray(lzip(lower, upper))

    def resid(self):
        """The residuals of the model."""
        if isinstance(self.predictors, pd.DataFrame):
            residuals = pd.Series(self.target.values - self.predict(self.predictors.values),
                                  index=self.target.index)
        else:
            residuals = self.target - self.predict(self.predictors)
        return residuals

    def ssr(self):
        """Sum of squared residuals."""
        resid = self.resid
        return np.dot(resid, resid)

    def uncentered_tss(self):
        """
        Uncentered sum of squares.
        The sum of the squared values of the (whitened) endogenous response
        variable.
        """
        endog = self.target
        return np.dot(endog, endog)

    def rsquared(self):
        """
        R-squared of the model.
        This is defined here as 1 - `ssr`/`centered_tss` if the constant is
        included in the model and 1 - `ssr`/`uncentered_tss` if the constant is
        omitted.
        """
        return 1 - self.ssr / self.uncentered_tss

    def rsquared_adj(self):
        """
        Adjusted R-squared.
        This is defined here as 1 - (`nobs`-1)/`df_resid` * (1-`rsquared`)
        if a constant is included and 1 - `nobs`/`df_resid` * (1-`rsquared`) if
        no constant is included.
        """
        return 1 - (np.divide(self.nobs, self.df_resid) * (1 - self.rsquared))

    def ess(self):
        """
        The explained sum of squares.
        If a constant is present, the centered total sum of squares minus the
        sum of squared residuals. If there is no constant, the uncentered total
        sum of squares is used.
        """
        return self.uncentered_tss - self.ssr

    def mse_model(self):
        """
        Mean squared error the model.
        The explained sum of squares divided by the model degrees of freedom.
        """
        return self.ess / self.df_model

    def mse_resid(self):
        """
        Mean squared error of the residuals.
        The sum of squared residuals divided by the residual degrees of
        freedom.
        """
        return self.ssr / self.df_resid

    def mse_total(self):
        """
        Total mean squared error.
        The uncentered total sum of squares divided by the number of
        observations.
        """
        return self.uncentered_tss / (self.df_resid + self.df_model)

    def fvalue(self):
        """
        F-statistic of the fully specified model.
        Calculated as the mean squared error of the model divided by the mean
        squared error of the residuals.
        """
        return self.mse_model / self.mse_resid

    def f_pvalue(self):
        """The p-value of the F-statistic."""
        return stats.f.sf(self.fvalue, self.df_model, self.df_resid)

    def loglike(self):
        r"""
        Compute the value of the Gaussian log-likelihood function at params.
        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `endog`.
        Parameters
        ----------
        params : array_like
            The model parameters.
        Returns
        -------
        float
            The value of the log-likelihood function for a GLS Model.
        Notes
        -----
        The log-likelihood function for the normal distribution is
        .. math:: -\frac{n}{2}\log\left(\left(Y-\hat{Y}\right)^{\prime}
                   \left(Y-\hat{Y}\right)\right)
                  -\frac{n}{2}\left(1+\log\left(\frac{2\pi}{n}\right)\right)
                  -\frac{1}{2}\log\left(\left|\Sigma\right|\right)
        Y and Y-hat are whitened.
        """
        # TODO: combine this with OLS/WLS loglike and add _det_sigma argument
        nobs2 = self.nobs / 2.0
        # TODO: not sure if should include intercept here (currently not)
        SSR = np.sum((self.target - np.dot(self.predictors, self.coef_)) ** 2, axis=0)
        llf = -np.log(SSR) * nobs2  # concentrated likelihood
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2  # with likelihood constant
        return llf

    def aic(self):
        r"""
        Akaike's information criteria.
        For a model with a constant :math:`-2llf + 2(df\_model + 1)`. For a
        model without a constant :math:`-2llf + 2(df\_model)`.
        """
        # TODO: need to think about dealing with the constant throughout here (if intercept requested)
        return -2 * self.llf + 2 * (self.df_model# + self.k_constant
                                    )

    def bic(self):
        r"""
        Bayes' information criteria.
        For a model with a constant :math:`-2llf + \log(n)(df\_model+1)`.
        For a model without a constant :math:`-2llf + \log(n)(df\_model)`.
        """
        return (-2 * self.llf + np.log(self.nobs) * (self.df_model
                                                     # + self.k_constant
                                                     ))


class LassoRegression(LinearRegression):

    def __init__(self, fit_intercept=True,
                 normalize=False,
                 copy_X=True,
                 positive=False,
                 alpha=1):
        """Constructor. It runs every instance is created"""

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.positive = positive
        self.is_fitted_ = False
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None, verbose=True):
        """docstring"""

        # TODO: self._preprocess_data
        # X, y, X_offset, y_offset, X_scale = self._preprocess_data(
        #    X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
        #    copy=self.copy_X, sample_weight=sample_weight,
        #    return_mean=True)

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

        # constrains on coefficient >=0
        lbx = np.zeros(ntheta) if self.positive else -np.inf*np.ones(ntheta)

        # create optimization problem (x: optimization parameter, f: cost function)
        nlp = {"x": theta, "f": 0.5*ca.dot(e, e) + 0.5*self.alpha*ca.sum1(ca.fabs(theta))}

        # solve opt
        solver = ca.nlpsol("ols", "ipopt", nlp)
        if verbose:
            sol = solver(x0=np.zeros(ntheta), lbx=lbx)
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                sol = solver(x0=np.zeros(ntheta), lbx=lbx)

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


class RidgeRegression(LinearRegression):

    def fit(self, X, y, sample_weight=None, verbose=True, alpha=1):
        """docstring"""

        # TODO: self._preprocess_data
        # X, y, X_offset, y_offset, X_scale = self._preprocess_data(
        #    X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
        #    copy=self.copy_X, sample_weight=sample_weight,
        #    return_mean=True)

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
            X = np.c_[np.ones(X.shape[0]), X]

        (N, ntheta) = X.shape
        theta = ca.SX.sym("theta", ntheta)

        # create residual
        e = y - ca.mtimes(X, theta)

        # constrains on coefficient >=0
        lbx = np.zeros(ntheta) if self.positive else -np.inf*np.ones(ntheta)

        # create optimization problem (x: optimization parameter, f: cost function)
        nlp = {"x": theta, "f": 0.5*ca.dot(e, e) + 0.5*alpha*ca.dot(theta, theta)}

        # solve opt
        solver = ca.nlpsol("ols", "ipopt", nlp)
        if verbose:
            sol = solver(x0=np.zeros(ntheta), lbx=lbx)
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                sol = solver(x0=np.zeros(ntheta), lbx=lbx)

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

# executable for stand-alone program
if __name__ == "__main__":

    from sklearn.datasets import make_regression
    import sklearn.linear_model as sk

    np.random.seed(42)

    bias = 5
    (X, y, theta0) = make_regression(n_samples=50,
                                         n_features=2,
                                         n_informative=2,
                                         n_targets=1,
                                         bias=bias,
                                         noise=0.0,
                                         shuffle=True,
                                         coef=True,
                                         random_state=42)

    X = pd.DataFrame(data=X, columns=["IV1", "IV2"])
    y = pd.Series(data=y, name="DV")

    # fit & predict using sklearn
    sk_reg = sk.LinearRegression().fit(X, y)
    sk_y_pred = sk_reg.predict(X)

    # fit & predict using hime
    rm_reg = LinearRegression().fit(X, y)
    rm_y_pred = rm_reg.predict(X)

    temp = pd.DataFrame(data={"sk": sk_y_pred, "rm": rm_y_pred.values.ravel()})
    temp = temp.assign(**{"diff": lambda x: x["sk"]-x["rm"]})

    # Check coef and intercept
    np.testing.assert_allclose(theta0, rm_reg.coef_.loc[["IV1", "IV2"]].values.ravel(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(bias, rm_reg.coef_.loc[["intercept"]].values.ravel(), rtol=1e-10, atol=1e-10)

    # assert prediction
    np.testing.assert_allclose(sk_y_pred, rm_y_pred.values.ravel(), rtol=1e-10, atol=1e-10)