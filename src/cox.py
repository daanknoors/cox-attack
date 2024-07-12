"""Methods for computing the Cox Proportional Hazards Model"""
import numpy as np
import pandas as pd

from scipy.optimize import minimize

class CoxPH:
    """Simplified implementation of the Cox Proportional Hazards Model.
    For a more full-featured implementation, see the `lifelines` package."""

    def __init__(self, init_coef=None, max_iter=100, learning_rate=0.01, solver="Newton-CG", verbose=1):
        self.init_coef = init_coef
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.solver = solver
        self.verbose = verbose


    def fit(self, data, event_column, duration_column, covariate_columns=None):
        """Fit the Cox Proportional Hazards model to the data."""
        # process data and extract relevant columns
        data = self._preprocess_data(data, event_column, duration_column, covariate_columns)
        events, durations = data[event_column].values, data[duration_column].values

        data_covariates = data.drop([event_column, duration_column], axis=1)
        covariates = data_covariates.values
        self.covariate_columns = data_covariates.columns.tolist()


        # check the validity of the input arguments and initialize fit parameters
        self._check_args()
        self._init_fit(n_covariates=covariates.shape[1])

        # compute the objective function of the Cox Proportional Hazards model pre-training
        if self.verbose:
            print('Pre-training negative log partial likelihood:', self._objective_function(self.coef_, covariates, events))

        # fit the model using the specified solver
        self._fit(self.coef_, covariates, events)

        if self.verbose:
            print('Fit model with coefficients:', self.coef_)

            # compute the objective function of the Cox Proportional Hazards model post-training
            print('Post-training negative log partial likelihood:', self._objective_function(self.coef_, covariates, events))

        # todo: remove if gradients can be initialized as array in _init_fit
        self.gradients_ = np.array(self.gradients_)
        return self

    def _log_partial_likelihood(self, coef, covariates, events):
        """Compute the log partial likelihood of the Cox Proportional Hazards model.

        Assumes data is sorted in descending order of event time. Run process_data before calling this method."""
        exp_pred_cumulative = np.cumsum(np.exp(np.dot(covariates, coef)))

        # compute the log partial likelihood
        log_lik = np.sum(events * (np.dot(covariates, coef) - np.log(exp_pred_cumulative)))
        return log_lik

    def _objective_function(self, coef, covariates, events):
        """Compute the negative log partial likelihood of the Cox Proportional Hazards model,
        which is the objective function to be minimized."""
        return - self._log_partial_likelihood(coef, covariates, events)

    def _gradient_objective_function(self, coef, covariates, events):
        """Compute the gradient of the objective function of the Cox Proportional Hazards model.

        Assumes data is sorted in descending order of event time. Run process_data before calling this method."""
        # break down the computation of the gradient into smaller components
        exp_pred = np.exp(np.dot(covariates, coef))
        exp_pred_cumulative = np.cumsum(exp_pred)
        exp_pred_cumulative_by_features = np.cumsum(covariates * exp_pred.reshape(-1, 1), axis=0)

        gradient = -np.sum(events.reshape(-1, 1) * (covariates - (exp_pred_cumulative_by_features / exp_pred_cumulative.reshape(-1, 1))), axis=0)

        # save gradient progression
        self.gradients_.append(gradient)
        return gradient

    def _fit(self, coef, covariates, events):
        """Optimize the Cox Proportional Hazards model using a numerical optimization method."""
        # we minimize the negative log partial likelihood
        result = minimize(
            self._objective_function,
            coef,
            args=(covariates, events),
            method=self.solver,
            jac=self._gradient_objective_function,
            options={'maxiter': self.max_iter})
        self.coef_ = result.x

    def _check_args(self):
        """Check the validity of the input arguments."""
        # check solver and learning rate combination
        if (self.solver == 'gradient_descent') and (self.learning_rate is None):
            raise ValueError('Learning rate must be specified for gradient descent solver.')

    def _init_fit(self, n_covariates):
        """Initialize the coefficients before fitting the model."""
        # check and set start coefficients
        if self.init_coef is None:
            self.coef_ = np.zeros(n_covariates)
        elif len(self.init_coef) != n_covariates:
            raise ValueError('Initial coefficients must have the same dimension as the number of covariates.')
        else:
            self.coef_ = self.init_coef.copy()

        # save gradients
        # todo: look for way to initialize gradient vector and save at each iteration. Now simply append
        self.gradients_ = []


    @staticmethod
    def _preprocess_data(data, event_column, duration_column, covariate_columns=None):
        """Preprocess data before fitting the model."""
        data = data.copy()

        # sort observations based on descending event time
        # to ensure that the risk set is correctly computed
        data = data.sort_values(by=[duration_column], ascending=False).reset_index(drop=True)

        # add random noise to the data to avoid ties
        # as cox model assumes continuous hazard
        data[duration_column] += np.random.normal(0, 1e-8, size=len(data))

        # subset data if covariate columns are specified
        if covariate_columns is not None:
            data = data[[event_column, duration_column] + covariate_columns]

        return data


def compute_gradient_cox(df, event_column, time_column, covariate_columns, beta):
    """Compute the gradient of the Cox Proportional Hazards log-likelihood."""
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
            exp_pred = np.exp(np.dot(risk_set, beta))
            gradient += X[i] - np.dot(exp_pred, risk_set) / np.sum(exp_pred)
    return gradient


def gradient_update(beta, gradient, learning_rate):
    return beta + learning_rate * gradient
