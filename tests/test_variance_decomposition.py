"""The variance decomposition identity: diffusive + event == total, exactly."""
import math

import pytest

from evrp.variance import decompose, variance_from_vol_points, vol_points_from_variance
from evrp.reference_oracle import decompose_variance_loop


@pytest.mark.parametrize("total_vol,trailing_vol,holding_days", [
    (32.0, 25.0, 2.0),
    (50.0, 50.0, 2.0),   # total == diffusive: event component should be exactly zero
    (10.0, 40.0, 1.0),   # trailing richer than total: event component negative, disclosed edge case
    (65.3, 12.1, 3.0),
])
def test_decomposition_identity_exact(total_vol, trailing_vol, holding_days):
    d = decompose(total_vol, trailing_vol, holding_days)
    assert d.diffusive_variance + d.event_variance == pytest.approx(d.total_variance, abs=1e-15)
    assert d.total_variance == pytest.approx(variance_from_vol_points(total_vol, holding_days), rel=1e-12)


def test_decomposition_zero_event_case():
    d = decompose(50.0, 50.0, 2.0)
    assert d.event_variance == pytest.approx(0.0, abs=1e-12)


def test_vol_variance_roundtrip():
    for vol, days in [(12.3, 1.0), (45.0, 2.0), (0.5, 10.0)]:
        v = variance_from_vol_points(vol, days)
        back = vol_points_from_variance(v, days)
        assert back == pytest.approx(vol, rel=1e-12)


def test_vol_points_from_variance_rejects_negative():
    with pytest.raises(ValueError):
        vol_points_from_variance(-1e-9, 2.0)


@pytest.mark.parametrize("total_vol,trailing_vol,holding_days", [
    (32.0, 25.0, 2.0), (10.0, 40.0, 1.0), (65.3, 12.1, 3.0),
])
def test_decomposition_matches_independent_loop_oracle(total_vol, trailing_vol, holding_days):
    d = decompose(total_vol, trailing_vol, holding_days)
    total_loop, diff_loop, event_loop = decompose_variance_loop(total_vol, trailing_vol, holding_days)
    assert d.total_variance == pytest.approx(total_loop, abs=1e-15)
    assert d.diffusive_variance == pytest.approx(diff_loop, abs=1e-15)
    assert d.event_variance == pytest.approx(event_loop, abs=1e-15)
