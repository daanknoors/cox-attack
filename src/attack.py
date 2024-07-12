"""Attacks against the Cox Proportional Hazards Model"""

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from src import cox

class CoxAttack:

        def __init__(self, n_records, column_names, actual_gradients, init_coef=None, max_iter=100):
            self.n_records = n_records
            self.column_names = column_names
            self.actual_gradients = actual_gradients
            self.init_coef = init_coef
            self.max_iter = max_iter

        def init_data(self):
            return pd.DataFrame(np.random.randn(self.n_records, len(self.column_names)), columns=self.column_names)
        def fit(self, event_column, duration_column, covariate_columns=None):
            data_reconstruct = self.init_data()

            # todo: implement minimization algorithm to find the adversarial data
            # todo: limit search space over assumed possible data values, e.g. event is only binary
            # self.result_ = minimize(self._gradient_difference, data_reconstruct.values, args=(event_column, duration_column, covariate_columns), method='L-BFGS-B')
            return self

        def _gradient_difference(self, data_reconstruct, event_column, duration_column, covariate_columns):
            """Compute the gradient difference between the actual and reconstructed gradients."""

            if not isinstance(data_reconstruct, pd.DataFrame):
                data_reconstruct = pd.DataFrame(data_reconstruct, columns=self.column_names)

            cph_reconstruct = cox.CoxPH(init_coef=self.init_coef, max_iter=self.max_iter)
            cph_reconstruct.fit(data_reconstruct, event_column, duration_column, covariate_columns)

            # get first gradient of reconstructed data
            gradient_reconstruct = cph_reconstruct.gradients_[0]

            # compute the gradient difference
            gradient_diff = np.sum((self.actual_gradients - gradient_reconstruct) ** 2)
            return gradient_diff







# class CoxAttack:
#
#     def __init__(self, n_records, column_names, max_iter=100, learning_rate=0.01):
#         self.n_records = n_records
#         self.column_names = column_names
#         self.max_iter = max_iter
#         self.learning_rate = learning_rate
#
#     def init_data(self):
#         return pd.DataFrame(np.random.randn(self.n_records, len(self.column_names)), columns=self.column_names)
#     def run(self, event_column, time_column, covariate_columns, cox_params, actual_gradient):
#         data_reconstruct = self.init_data()
#
#         result = minimize(gradient_difference, data_reconstruct, args=(event_column, time_column, covariate_columns, cox_params, actual_gradient), method='L-BFGS-B')
#         return result




def gradient_difference(df_reconstruct, event_column, time_column, covariate_columns, beta, gradient_actual):
    """Compute the gradient difference of the Cox Proportional Hazards log-likelihood."""
    gradient_reconstruct = compute_gradient_cox(df_reconstruct, event_column, time_column, covariate_columns, beta)
    return np.sum((gradient_actual - gradient_reconstruct) ** 2)

def gradient_inversion(df_reconstruct, event_column, time_column, covariate_columns, beta, gradient_actual):
    """Compute the gradient inversion of the Cox Proportional Hazards log-likelihood."""
    gradient_reconstruct = compute_gradient_cox(df_reconstruct, event_column, time_column, covariate_columns, beta)
    return np.sum((gradient_actual + gradient_reconstruct) ** 2)