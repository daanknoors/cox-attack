from src import cox
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

import pytest


@pytest.fixture
def data():
    df = pd.DataFrame({
        'patient': [1, 2, 3],
        'time': [5, 3, 4],
        'event': [1, 1, 0],
        'x1': [2, 1, 3],
        'x2': [1, 2, 1]
    })
    df
    return df


# params
MAX_ITER = 10000


def test_compute_gradient_cox(data):
    init_beta = np.array([0.1, 0.2])
    gradient = cox.compute_gradient_cox(df=data, event_column='event', time_column='time', covariate_columns=['x1', 'x2'], beta=init_beta)
    assert gradient.shape == (2,)
    assert np.allclose(gradient, np.array([-1, 0.65574665]))


def test_cox_ph_newton_cg(data):
    #params
    init_beta = np.array([0.1, 0.2])
    solver = "Newton-CG"

    # fit model
    model = cox.CoxPH(init_coef=init_beta, max_iter=MAX_ITER, solver=solver)
    model.fit(data, event_column='event', duration_column='time', covariate_columns=['x1', 'x2'])
    assert model.coef_.shape == (2,)


def test_compare_cox():
    # load data
    rossi = load_rossi()

    # fit model with lifelines
    cph_lifelines = CoxPHFitter()
    cph_lifelines.fit(rossi, duration_col='week', event_col='arrest')
    coef_lifelines = cph_lifelines.params_.values

    # fit model with custom implementation
    cph_custom = cox.CoxPH(max_iter=MAX_ITER, solver="Newton-CG", verbose=1)
    cph_custom.fit(rossi, event_column='arrest', duration_column='week')
    coef_custom = cph_custom.coef_

    print(f"lifelines coefficients: {coef_lifelines}")
    print(f"custom coefficients: {coef_custom}")
    assert np.allclose(coef_lifelines, coef_custom,  rtol=0.1)