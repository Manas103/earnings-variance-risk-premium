"""Delta-hedged short straddle P&L, gross and net, plus Sharpe.

Trade, per event:
  entry (session before the print): sell the ATM call and ATM put
    (K = S_entry), collect premium, buy `hedge_shares_0` of stock to flatten
    the position's Black-Scholes delta.
  mid (the print has happened; one session of the 2-day holding window is
    left): rebalance the stock hedge to the new delta, computed at the
    post-event price with the EVENT PREMIUM GONE (only the name's ordinary
    diffusive vol is priced in for the one remaining day, because whatever
    was going to happen already happened).
  exit (next session): the straddle is worth exactly its intrinsic value
    (T = 0); buy it back to close, and unwind the remaining stock hedge.

Costs, net P&L only:
  - a half bid-ask spread on each option leg at entry (selling the straddle)
    and again at exit (buying it back to close);
  - a half bid-ask spread on every stock trade used to establish, rebalance,
    and unwind the delta hedge (3 stock trades per event).

Gross P&L never has any of the above subtracted; that is the whole point of
reporting both numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import options
from .simulate import T4_ABS_MEAN_OVER_STD
from .variance import decompose, variance_from_vol_points, vol_points_from_variance

HOLD_DAYS_TRADE = 2       # entry close -> exit ("next session's open", approximated as that close)
HOLD_DAYS_MID = 1         # time remaining at the mid (post-print) rebalance point
HOLD_DAYS_EVENT = 1       # basis on which the event vol component and the realized gap are measured

OPTION_HALF_SPREAD_FRAC = 0.035   # half bid-ask spread as a fraction of each option leg's price
STOCK_HALF_SPREAD_BPS = 6.0       # half bid-ask spread on the stock, in basis points of price


@dataclass(frozen=True)
class EventPricing:
    """All intermediate quantities for one event, exposed so the oracle test can diff each piece."""
    total_implied_vol_points: float
    diffusive_variance: float
    event_variance: float
    premium_received: float
    hedge_shares_0: float
    hedge_shares_1: float
    mark_value: float
    hedge_pnl: float
    gross_pnl: float
    option_cost_entry: float
    option_cost_exit: float
    hedge_cost: float
    net_pnl: float


def build_total_implied_vol_points(trailing_vol_long_points: float, implied_event_vol_points: float) -> tuple[float, float, float]:
    """Combine the diffusive and event variance pieces into one 2-day total implied vol.

    Of the 2 trading days the straddle is held (entry -> mid spans the event day itself,
    mid -> exit is an ordinary day), only ONE day is genuinely diffusive: the earnings-day
    return in evrp.simulate REPLACES that day's diffusive draw rather than adding to it, so
    the event day carries event variance only, not event-plus-diffusive. Pricing the event
    day's diffusive share on top of its event share would double count a day of variance
    the generator never produced.

    This was exactly the first bug found here: pricing with 2 full days of diffusive
    variance (one per holding day) plus the 1-day event variance gave a strictly positive
    mean gross P&L even with event richness forced to exactly zero (measured: gross Sharpe
    6.8 at zero richness, which should be close to 0 if the priced vol matched the
    generator's true vol with no edge). Root cause confirmed by rerunning with richness=0
    and watching premium_received exceed mark_value on average regardless; fixed by scaling
    the diffusive piece to 1 day (the mid -> exit day only), not 2. See README Findings.

    Returns (total_implied_vol_points, diffusive_variance_2day, event_variance_2day), where
    "_2day" names the window the TOTAL spans, not the number of diffusive days inside it.
    """
    diffusive_variance_2day = variance_from_vol_points(trailing_vol_long_points, HOLD_DAYS_MID)
    event_variance_2day = variance_from_vol_points(implied_event_vol_points, HOLD_DAYS_EVENT)
    total_variance_2day = diffusive_variance_2day + event_variance_2day
    total_implied_vol_points = vol_points_from_variance(total_variance_2day, HOLD_DAYS_TRADE)
    return total_implied_vol_points, diffusive_variance_2day, event_variance_2day


def price_event(
    s_entry: float,
    s_mid: float,
    s_exit: float,
    trailing_vol_long_points: float,
    implied_event_vol_points: float,
    option_half_spread_frac: float = OPTION_HALF_SPREAD_FRAC,
    stock_half_spread_bps: float = STOCK_HALF_SPREAD_BPS,
) -> EventPricing:
    strike = s_entry
    total_implied_vol_points, diffusive_variance, event_variance = build_total_implied_vol_points(
        trailing_vol_long_points, implied_event_vol_points
    )
    sigma_total = total_implied_vol_points / 100.0
    t_total = HOLD_DAYS_TRADE / 252.0
    t_mid = HOLD_DAYS_MID / 252.0
    sigma_mid = trailing_vol_long_points / 100.0  # the event premium has resolved; only diffusive vol remains

    premium_received = options.straddle_price(s_entry, strike, t_total, sigma_total)
    hedge_shares_0 = options.straddle_delta(s_entry, strike, t_total, sigma_total)
    hedge_shares_1 = options.straddle_delta(s_mid, strike, t_mid, sigma_mid)
    mark_value = options.straddle_price(s_exit, strike, 0.0, sigma_mid)  # intrinsic at T=0

    hedge_pnl = hedge_shares_0 * (s_mid - s_entry) + hedge_shares_1 * (s_exit - s_mid)
    gross_pnl = premium_received - mark_value + hedge_pnl

    option_cost_entry = option_half_spread_frac * premium_received
    option_cost_exit = option_half_spread_frac * mark_value
    stock_frac = stock_half_spread_bps / 10000.0
    hedge_cost = stock_frac * (
        abs(hedge_shares_0) * s_entry
        + abs(hedge_shares_1 - hedge_shares_0) * s_mid
        + abs(hedge_shares_1) * s_exit
    )
    net_pnl = gross_pnl - option_cost_entry - option_cost_exit - hedge_cost

    return EventPricing(
        total_implied_vol_points=total_implied_vol_points,
        diffusive_variance=diffusive_variance,
        event_variance=event_variance,
        premium_received=premium_received,
        hedge_shares_0=hedge_shares_0,
        hedge_shares_1=hedge_shares_1,
        mark_value=mark_value,
        hedge_pnl=hedge_pnl,
        gross_pnl=gross_pnl,
        option_cost_entry=option_cost_entry,
        option_cost_exit=option_cost_exit,
        hedge_cost=hedge_cost,
        net_pnl=net_pnl,
    )


def price_events_frame(
    events: pd.DataFrame,
    option_half_spread_frac: float = OPTION_HALF_SPREAD_FRAC,
    stock_half_spread_bps: float = STOCK_HALF_SPREAD_BPS,
) -> pd.DataFrame:
    """Vectorized-by-row pricing of every event. Returns a new DataFrame with pricing
    columns appended (gross_pnl, net_pnl, ret_gross, ret_net, total_implied_vol_points, ...).
    """
    records = []
    for row in events.itertuples():
        p = price_event(
            row.s_entry, row.s_mid, row.s_exit,
            row.trailing_vol_long_points, row.implied_event_vol_points,
            option_half_spread_frac, stock_half_spread_bps,
        )
        records.append({
            "event_id": row.event_id,
            "total_implied_vol_points": p.total_implied_vol_points,
            "diffusive_variance": p.diffusive_variance,
            "event_variance": p.event_variance,
            "premium_received": p.premium_received,
            "mark_value": p.mark_value,
            "hedge_pnl": p.hedge_pnl,
            "gross_pnl": p.gross_pnl,
            "option_cost_entry": p.option_cost_entry,
            "option_cost_exit": p.option_cost_exit,
            "hedge_cost": p.hedge_cost,
            "net_pnl": p.net_pnl,
            "ret_gross": p.gross_pnl / row.s_entry,
            "ret_net": p.net_pnl / row.s_entry,
        })
    pricing = pd.DataFrame(records)
    merged = events.merge(pricing, on="event_id", how="left")
    # A single day's |realized return| is a downward-biased point estimate of the sigma
    # that generated it (Jensen's inequality; see evrp.simulate's T4_ABS_MEAN_OVER_STD
    # docstring). Dividing it out gives an unbiased-in-expectation realized-vol estimate
    # to compare against the implied side; this is the column the "1.6 vol point" claim
    # is measured on, not the raw abs-return figure.
    merged["realized_event_vol_points_debiased"] = merged["realized_event_vol_points"] / T4_ABS_MEAN_OVER_STD
    merged["gap_points"] = merged["implied_event_vol_points"] - merged["realized_event_vol_points_debiased"]
    return merged


def sharpe_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    """Mean / stdev of a return series, annualized by sqrt(periods_per_year).
    Degenerate cases: fewer than 2 observations, or exactly zero variance, return 0.0
    rather than inf/nan (see tests/test_sharpe_sanity.py).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    mean = float(np.mean(returns))
    return mean / std * np.sqrt(periods_per_year)
