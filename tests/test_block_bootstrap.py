import numpy as np
import pandas as pd
import pytest

from black_litterman.block_bootstrap import run_simulation


@pytest.fixture
def daily_returns():
    rng = np.random.default_rng(7)
    values = rng.normal(0.0003, 0.008, size=(520, 2))
    return pd.DataFrame(values, columns=["A", "B"])


def test_simulation_respects_variable_iteration_count(daily_returns):
    result = run_simulation(
        daily_returns,
        pd.Series({"A": 0.6, "B": 0.4}),
        n_sims=137,
        block_size=21,
        batch_size=50,
        seed=10,
    )

    assert result.success_flags.shape == (137,)
    assert result.year_start_values.shape == (137, 16)
    assert result.ending_2042.shape == (137,)
    assert 0 <= result.funding_probability <= 1


def test_simulation_rejects_nonpositive_iterations(daily_returns):
    with pytest.raises(ValueError, match="n_sims must be positive"):
        run_simulation(
            daily_returns,
            pd.Series({"A": 0.5, "B": 0.5}),
            n_sims=0,
        )
