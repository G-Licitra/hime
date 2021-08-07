import pandas as pd

# Spector.csv ----------------------------------------------
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
             index= ["GPA", "TUCE" , "PSI"]
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

