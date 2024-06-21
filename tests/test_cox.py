from src import cox
import numpy as np
import pandas as pd
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


def test_compute_gradient_cox(data):
    init_beta = np.array([0.1, 0.2])
    gradient = cox.compute_gradient_cox(df=data, event_column='event', time_column='time', covariate_columns=['x1', 'x2'], beta=init_beta)
    assert gradient.shape == (2,)
    assert np.allclose(gradient, np.array([-1, 0.65574665]))