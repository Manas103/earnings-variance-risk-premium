"""Sharpe computation sanity on degenerate inputs."""
import numpy as np

from evrp.pnl import sharpe_ratio


def test_sharpe_zero_variance_returns_zero_not_inf():
    assert sharpe_ratio(np.array([0.01, 0.01, 0.01, 0.01]), periods_per_year=252) == 0.0


def test_sharpe_single_observation_returns_zero():
    assert sharpe_ratio(np.array([0.05]), periods_per_year=252) == 0.0


def test_sharpe_empty_returns_zero():
    assert sharpe_ratio(np.array([]), periods_per_year=252) == 0.0


def test_sharpe_sign_matches_mean_sign():
    pos = sharpe_ratio(np.array([0.02, 0.01, 0.03, -0.01]), periods_per_year=252)
    neg = sharpe_ratio(np.array([-0.02, -0.01, -0.03, 0.01]), periods_per_year=252)
    assert pos > 0
    assert neg < 0
    assert pos == -neg  # symmetric series negated should flip sign exactly


def test_sharpe_scales_with_sqrt_periods():
    r = np.array([0.02, 0.01, 0.03, -0.01, 0.015])
    sr_1 = sharpe_ratio(r, periods_per_year=1)
    sr_4 = sharpe_ratio(r, periods_per_year=4)
    assert sr_4 == sr_1 * 2.0  # sqrt(4) = 2
