"""Black-Scholes pricing and Greeks for ATM calls and puts, r = 0.

r is fixed at 0 throughout this project. The holding periods traded here are
one to two trading days, so a realistic risk-free carry (a few bps over two
days) changes every number in this project by less than floating-point noise
on the quantities that matter (straddle P&L, Sharpe ratios). Dropping it
keeps every formula below exactly the textbook zero-rate Black-Scholes form,
which is also what the hand-computed oracle in
tests/test_straddle_pnl_oracle.py diffs against.
"""
from __future__ import annotations

import math

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d1_d2(spot: float, strike: float, t_years: float, sigma: float) -> tuple[float, float]:
    """Black-Scholes d1 and d2 at r = 0.

    Degenerate at t_years <= 0 or sigma <= 0: callers must not call this
    for an already-expired or zero-vol contract. bs_price/bs_delta handle
    those cases explicitly without calling here.
    """
    if t_years <= 0.0 or sigma <= 0.0:
        raise ValueError("d1_d2 requires t_years > 0 and sigma > 0")
    vol_sqrt_t = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(spot: float, strike: float, t_years: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes price, r = 0. At t_years == 0 this is exactly intrinsic value."""
    if t_years <= 0.0:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic
    if sigma <= 0.0:
        # Zero vol: the forward (= spot, r=0) is known, price is discounted intrinsic (no discount, r=0).
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic
    d1, d2 = d1_d2(spot, strike, t_years, sigma)
    if is_call:
        return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, t_years: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes delta, r = 0."""
    if t_years <= 0.0 or sigma <= 0.0:
        # Degenerate / expired: delta is 0 or 1 (call) / 0 or -1 (put) depending on moneyness,
        # with a coin-flip-free convention at exactly spot == strike (delta 0.5 / -0.5),
        # matching the limit of the smooth BS delta as t_years, sigma -> 0.
        if spot > strike:
            return 1.0 if is_call else 0.0
        if spot < strike:
            return 0.0 if is_call else -1.0
        return 0.5 if is_call else -0.5
    d1, _ = d1_d2(spot, strike, t_years, sigma)
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0


def straddle_price(spot: float, strike: float, t_years: float, sigma: float) -> float:
    return bs_price(spot, strike, t_years, sigma, True) + bs_price(spot, strike, t_years, sigma, False)


def straddle_delta(spot: float, strike: float, t_years: float, sigma: float) -> float:
    return bs_delta(spot, strike, t_years, sigma, True) + bs_delta(spot, strike, t_years, sigma, False)
