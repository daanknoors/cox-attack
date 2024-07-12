"""Test the Cox Attack module."""

import numpy as np
import pandas as pd
import pytest
from lifelines.datasets import load_rossi


from src import attack, cox

def test_cox_attack():
    rossi = load_rossi()

    # assume adversary knows the following information
    n_records = rossi.shape[0]
    event_column = 'arrest'
    duration_column = 'week'
    column_names = rossi.columns.tolist()

    # train cox model
    cph = cox.CoxPH(max_iter=100, solver="Newton-CG", learning_rate=0.01)
    cph.fit(rossi, event_column=event_column, duration_column=duration_column)

    # extract first actual gradient
    actual_gradients = cph.gradients_[0]

    # attack the cox model
    cox_attack = attack.CoxAttack(n_records=n_records, column_names=column_names, actual_gradients=actual_gradients)
    cox_attack.fit(event_column=event_column, duration_column=duration_column)
    print(cox_attack.result_)
