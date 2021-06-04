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

    # Behaviour Methods
    def fit(self, X, y):
        """docstring"""

        if self.fit_intercept:
            X["intercept"] = 1

        (N, ntheta) = X.shape
        theta = ca.SX.sym("theta", ntheta)

        # create residual
        e = y.values - ca.mtimes(X.values, theta)

        # create optimization problem (x: optimization parameter, f: cost function)
        nlp = {"x": theta, "f": 0.5 * ca.dot(e, e)}

        # solve opt
        solver = ca.nlpsol("ols", "ipopt", nlp)
        sol = solver(x0=np.zeros(ntheta))

        self.coef_ = sol["x"]

        return self

    def predict(self, X):
        """Give this object a raise"""
        pass  # much change here only



# executable for stand-alone program
if __name__ == "__main__":


    from sklearn.datasets import make_regression
    import sklearn.linear_model as sk

    np.random.seed(42)

    (X, y, theta0) = make_regression(n_samples=50,
                                         n_features=2,
                                         n_informative=2,
                                         n_targets=1,
                                         bias=5.0,
                                         noise=0.0,
                                         shuffle=True,
                                         coef=True,
                                         random_state=42)

    X = pd.DataFrame(data=X, columns=["IV1", "IV2"])
    y = pd.Series(data=y, name="DV")

    reg = sk.LinearRegression().fit(X, y)

    print(f"theta0 = {np.round(theta0, 3)} \n theta_sk = {np.round(reg.coef_, 3)}")
    reg.coef_

    reg.predict(np.array([[3, 5]]))

    romeo_reg = LinearRegression().fit(X, y)

    romeo_reg.coef_