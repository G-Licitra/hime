# add extra case from here
# https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html#OLS-with-dummy-variables


import pandas as pd

# Spector OLS =====================================================
df_summary = pd.Series(data={"Model": "OLS",
"Dep. Variable": "GRADE",
"R-squared (uncentered)": 0.505,
"Adj. R-squared (uncentered)": 0.454,
"Method": "Least Squares",
"F-statistic": 9.852,
"Prob (F-statistic)": 0.000121,
"Log-Likelihood": -17.077,
"No. Observations":  32,
"AIC": 40.15,
"Df Residuals": 29,
"BIC": 44.55,
"Df Model":  3,
"Covariance Type": "nonrobust"
})

df_coef = pd.DataFrame(data= [[0.1629, 0.137, 1.185, 0.246, -0.118, 0.444],
                    [-0.0136, 0.020, -0.692, 0.494, -0.054, 0.027],
                    [0.3650, 0.155, 2.349, 0.026, 0.047, 0.683]
                    ],
             columns=["coef", "std err", "t",  "P>|t|", "[0.025", "0.975]"],
             index=["GPA", "TUCE" , "PSI"]
)

df_residual = pd.Series(data={"Omnibus": 3.670,
                               "Durbin-Watson": 2.488,
                                "Prob(Omnibus)": 0.160,
                                "Jarque-Bera (JB)": 2.422,
                                "Skew": 0.484,
                                "Prob(JB)": 0.298,
                                "Kurtosis": 2.062,
                                "Cond. No.": 45.6
                              })

# Spector OLS (intercept) -------------------------------------------

df_summary = pd.Series(data={"Model": "OLS",
"Dep. Variable": "GRADE",
"R-squared": 0.416,
"Adj. R-squared": 0.353,
"Method": "Least Squares",
"F-statistic": 6.646,
"Prob (F-statistic)": 0.00157,
"Log-Likelihood": -12.978,
"No. Observations":  32,
"AIC": 33.96,
"Df Residuals": 28,
"BIC": 39.82,
"Df Model":  3,
"Covariance Type": "nonrobust"
})

df_coef = pd.DataFrame(data= [[0.4639, 0.162, 2.864, 0.008, 0.132, 0.796],
                              [0.0105, 0.019 , 0.539, 0.594, -0.029, 0.050],
                              [0.3786, 0.139, 2.720, 0.011, 0.093, 0.664],
                              [-1.4980, 0.524, -2.859, 0.008, -2.571, -0.425],
                            ],
             columns=["coef", "std err", "t",  "P>|t|", "[0.025", "0.975]"],
             index= ["GPA", "TUCE" , "PSI", "intercept"]
)

df_residual = pd.Series(data={"Omnibus": 0.176,
                               "Durbin-Watson": 2.346,
                                "Prob(Omnibus)": 0.916,
                                "Jarque-Bera (JB)": 0.167,
                                "Skew": 0.141,
                                "Prob(JB)": 0.920,
                                "Kurtosis": 2.786,
                                "Cond. No.": 176
                              })



# Spector OLS | intercept | X normalized (sklearn) -------------------------------------
# y_fit_normalized = y_fittevalues
# df_summary = df_summary without normalization
# df_residual  = df_residual  without normalization

df_coef = pd.DataFrame(data= [[0.2131, 0.074, 2.864, 0.008,  0.061, 0.365],
                              [0.0403, 0.075, 0.539, 0.594, -0.113, 0.194],
                              [0.1878, 0.069, 2.720, 0.011,  0.046, 0.329],
                              [0.3438, 0.069, 5.011, 0.000,  0.203, 0.484],
                            ],
             columns=["coef", "std err", "t",  "P>|t|", "[0.025", "0.975]"],
             index= ["GPA", "TUCE" , "PSI", "intercept"])

