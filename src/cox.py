import numpy as np
import pandas as pd


def compute_gradient_cox(df, event_column, time_column, covariate_columns, beta):
    df_ordered = df.copy().sort_values(by=[time_column])

    E = df_ordered[event_column].values
    X = df_ordered[covariate_columns].values

    gradient = np.zeros_like(beta)
    for i in range(len(df_ordered)):
        # only compute gradient for uncensored data
        if E[i] == 1:
            # Determine the risk set for the current time using indexing
            risk_set = X[i:]

            # compute hazard risk group
            exp_beta_Xj = np.exp(np.dot(risk_set, beta))
            gradient += X[i] - np.dot(exp_beta_Xj, risk_set) / np.sum(exp_beta_Xj)
    return gradient

