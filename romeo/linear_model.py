import numpy as np
import casadi as ca
import pandas as pd

class LinearRegression:

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

        if self.copy_X:
            X = X.copy()

        if self.fit_intercept:
            # add 1 col using broadcasting
            X["intercept"] = 1
            # move intercept col at the first column
            cols = X.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            X = X[cols]


        (N, ntheta) = X.shape
        theta = ca.SX.sym("theta", ntheta)

        # create residual
        e = y.values - ca.mtimes(X.values, theta)

        # create optimization problem (x: optimization parameter, f: cost function)
        nlp = {"x": theta, "f": 0.5 * ca.dot(e, e)}

        # solve opt
        solver = ca.nlpsol("ols", "ipopt", nlp)
        sol = solver(x0=np.zeros(ntheta))

        self.coef_ = pd.DataFrame(data={"coef": sol["x"].full().ravel()},
                                  index=X.columns)

        return self


    def predict(self, X:pd.DataFrame):
        """y = a*x + b"""

        y_pred = X.dot(self.coef_.filter(items=X.columns, axis=0))

        if self.fit_intercept:
            y_pred += self.coef_.loc['intercept']

        return y_pred



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

    # fit & predict using romeo
    rm_reg = LinearRegression().fit(X, y)
    rm_y_pred = rm_reg.predict(X)

    temp = pd.DataFrame(data={"sk": sk_y_pred, "rm": rm_y_pred.values.ravel()})
    temp = temp.assign(**{"diff": lambda x: x["sk"]-x["rm"]})

    # Check coef and intercept
    np.testing.assert_allclose(theta0, rm_reg.coef_.loc[["IV1", "IV2"]].values.ravel(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(bias, rm_reg.coef_.loc[["intercept"]].values.ravel(), rtol=1e-10, atol=1e-10)

    # assert prediction
    np.testing.assert_allclose(sk_y_pred, rm_y_pred.values.ravel(), rtol=1e-10, atol=1e-10)