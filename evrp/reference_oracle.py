"""Independent reference implementations, deliberately coded a different way
from the production path, to diff against it.

- `bs_price_quad` prices a European option by numerically integrating the
  discounted payoff against the lognormal terminal density (scipy.integrate),
  which is the textbook risk-neutral valuation definition of an option price.
  It shares no formula with `evrp.options.bs_price` (which uses the closed-form
  erf-based N(d1)/N(d2) shortcut) and is the obviously-correct, obviously-slow
  side of the diff.
- `decompose_variance_loop` recomputes the variance decomposition with
  explicit day-count loops instead of the vectorized sigma^2 * T formula, to
  catch an annualization or scaling bug that could otherwise hide inside a
  single reused helper function.
- `straddle_pnl_loop` recomputes one event's delta-hedged straddle P&L
  black-box, one hedging step at a time with explicit running cash and
  share balances, instead of the closed two-term algebra in `evrp.pnl`.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, stats

from . import options


def bs_price_quad(spot: float, strike: float, t_years: float, sigma: float, is_call: bool) -> float:
    """Price via numerical integration of the discounted payoff against the
    lognormal terminal price density, r = 0. Independent of evrp.options.bs_price.
    """
    if t_years <= 0.0 or sigma <= 0.0:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)

    mu = math.log(spot) - 0.5 * sigma * sigma * t_years
    sd = sigma * math.sqrt(t_years)

    def integrand(z: float) -> float:
        s_t = math.exp(mu + sd * z)
        payoff = max(s_t - strike, 0.0) if is_call else max(strike - s_t, 0.0)
        return payoff * stats.norm.pdf(z)

    value, _ = integrate.quad(integrand, -10.0, 10.0, limit=200)
    return value


def straddle_price_quad(spot: float, strike: float, t_years: float, sigma: float) -> float:
    return bs_price_quad(spot, strike, t_years, sigma, True) + bs_price_quad(spot, strike, t_years, sigma, False)


def decompose_variance_loop(total_implied_vol_points: float, trailing_vol_points: float, holding_days: float) -> tuple[float, float, float]:
    """Loop-based reimplementation of evrp.variance.decompose's arithmetic."""
    trading_days_per_year = 252.0
    total_var = 0.0
    diffusive_var = 0.0
    # Accumulate variance one trading day at a time instead of sigma^2 * T in one shot.
    for _ in range(int(round(holding_days))):
        total_var += (total_implied_vol_points / 100.0) ** 2 / trading_days_per_year
        diffusive_var += (trailing_vol_points / 100.0) ** 2 / trading_days_per_year
    event_var = total_var - diffusive_var
    return total_var, diffusive_var, event_var


def straddle_pnl_loop(
    s_entry: float, s_mid: float, s_exit: float,
    trailing_vol_long_points: float, implied_event_vol_points: float,
    option_half_spread_frac: float, stock_half_spread_bps: float,
) -> dict:
    """Independent step-by-step reimplementation of evrp.pnl.price_event,
    using explicit running cash/share balances instead of the closed-form
    two-term hedge_pnl expression.
    """
    strike = s_entry
    # Only ONE of the 2 holding days is genuinely diffusive (the event day's return is
    # REPLACED by the earnings jump in evrp.simulate, not added to a diffusive draw);
    # see evrp.pnl.build_total_implied_vol_points for the bug this once was.
    diffusive_var_1d = (trailing_vol_long_points / 100.0) ** 2 * (1.0 / 252.0)
    event_var_1d = (implied_event_vol_points / 100.0) ** 2 * (1.0 / 252.0)
    total_var_2d = diffusive_var_1d + event_var_1d
    sigma_total = math.sqrt(total_var_2d / (2.0 / 252.0))
    sigma_mid = trailing_vol_long_points / 100.0

    premium = options.bs_price(s_entry, strike, 2.0 / 252.0, sigma_total, True) + \
        options.bs_price(s_entry, strike, 2.0 / 252.0, sigma_total, False)
    delta0 = options.bs_delta(s_entry, strike, 2.0 / 252.0, sigma_total, True) + \
        options.bs_delta(s_entry, strike, 2.0 / 252.0, sigma_total, False)

    cash = premium  # received premium
    shares = 0.0

    # Establish hedge at entry.
    trade0 = delta0 - shares
    cash -= trade0 * s_entry
    shares += trade0

    # Rebalance at mid.
    delta1 = options.bs_delta(s_mid, strike, 1.0 / 252.0, sigma_mid, True) + \
        options.bs_delta(s_mid, strike, 1.0 / 252.0, sigma_mid, False)
    trade1 = delta1 - shares
    cash -= trade1 * s_mid
    shares += trade1

    # Exit: buy back the straddle (intrinsic), unwind remaining hedge.
    mark = max(s_exit - strike, 0.0) + max(strike - s_exit, 0.0)
    cash -= mark
    trade2 = -shares
    cash -= trade2 * s_exit
    shares += trade2

    gross_pnl = cash
    option_cost_entry = option_half_spread_frac * premium
    option_cost_exit = option_half_spread_frac * mark
    stock_frac = stock_half_spread_bps / 10000.0
    hedge_cost = stock_frac * (abs(trade0) * s_entry + abs(trade1) * s_mid + abs(trade2) * s_exit)
    net_pnl = gross_pnl - option_cost_entry - option_cost_exit - hedge_cost

    return {
        "premium": premium, "mark": mark, "gross_pnl": gross_pnl, "net_pnl": net_pnl,
        "hedge_cost": hedge_cost, "option_cost_entry": option_cost_entry, "option_cost_exit": option_cost_exit,
    }
