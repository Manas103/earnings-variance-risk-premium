"""Front-expiry implied variance, decomposed into a diffusive and an event piece.

The identity this module exists to make testable:

    total_variance == diffusive_variance + event_variance

Variance here is "total variance over the holding window", i.e. sigma^2 * T
in Black-Scholes units (T in years), NOT an annualized vol. ann_vol_points
converts an annualized vol (in points, e.g. 32.0 for 32%) into the variance
over a given holding period, and vol_points_from_variance is its exact
inverse. The decomposition is constructed as an arithmetic identity, not
asserted after the fact: event_variance is always computed as
"total_variance - diffusive_variance", never generated independently, so a
bug anywhere else in the pipeline that broke the identity would have to
break this subtraction itself.
"""
from __future__ import annotations

from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252.0


def variance_from_vol_points(vol_points: float, holding_days: float) -> float:
    """Annualized vol in points (32.0 = 32%) -> total variance over holding_days trading days."""
    sigma = vol_points / 100.0
    t_years = holding_days / TRADING_DAYS_PER_YEAR
    return sigma * sigma * t_years


def vol_points_from_variance(variance: float, holding_days: float) -> float:
    """Inverse of variance_from_vol_points. variance must be >= 0."""
    if variance < 0.0:
        raise ValueError(f"variance must be non-negative, got {variance}")
    t_years = holding_days / TRADING_DAYS_PER_YEAR
    sigma = (variance / t_years) ** 0.5
    return sigma * 100.0


@dataclass(frozen=True)
class VarianceDecomposition:
    holding_days: float
    diffusive_variance: float
    event_variance: float

    @property
    def total_variance(self) -> float:
        return self.diffusive_variance + self.event_variance

    @property
    def diffusive_vol_points(self) -> float:
        return vol_points_from_variance(self.diffusive_variance, self.holding_days)

    @property
    def event_vol_points(self) -> float:
        return vol_points_from_variance(max(self.event_variance, 0.0), self.holding_days)

    @property
    def total_vol_points(self) -> float:
        return vol_points_from_variance(self.total_variance, self.holding_days)


def decompose(
    total_implied_vol_points: float,
    trailing_realized_vol_points: float,
    holding_days: float,
) -> VarianceDecomposition:
    """Split a front-expiry implied vol into a diffusive piece and an event residual.

    diffusive_variance is the trailing (non-event) realized vol, scaled to the
    same holding_days window the implied vol spans. event_variance is
    whatever variance is left over: total - diffusive. It can come out
    slightly negative on a name with unusually low event richness relative
    to its own trailing vol; that is a real, disclosed edge case (see
    Findings / Limitations), not clamped away, because clamping would break
    the identity this module is built to guarantee.
    """
    total_variance = variance_from_vol_points(total_implied_vol_points, holding_days)
    diffusive_variance = variance_from_vol_points(trailing_realized_vol_points, holding_days)
    event_variance = total_variance - diffusive_variance
    return VarianceDecomposition(
        holding_days=holding_days,
        diffusive_variance=diffusive_variance,
        event_variance=event_variance,
    )
