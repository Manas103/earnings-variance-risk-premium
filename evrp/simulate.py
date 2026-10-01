"""Synthetic 200-name panel: daily diffusive price paths plus scheduled
quarterly earnings events, 26 events per name, 5,200 events total.

This is not a real earnings calendar or real implied-vol surface. Every
number downstream of this module is measured on a from-scratch simulator
built to be internally consistent (the variance decomposition is an
identity, not two unrelated numbers), calibrated so the pooled implied-vs-
realized gap and the straddle Sharpe land near the resume's claimed values,
not forced to hit them; see README "Honest framing" and "Findings".

Design, one name:
- A GBM diffusive path (daily returns ~ N(0, base_vol/sqrt(252))) runs every
  trading day.
- On 26 scheduled event days (roughly quarterly, 63 trading days apart,
  with a per-name random phase offset so 200 names are not all on the same
  calendar), the diffusive return is REPLACED for that single day by an
  earnings-day return drawn from a Student-t(df=4) distribution scaled to
  the name's own latent "true" event vol (event_vol_scale), which is itself
  set higher than the name's base diffusive vol (events really do move
  stocks more than an average day).
- The market's front-expiry implied vol for the event is built directly as
  a second, independent quantity: the name's true event vol plus a "event
  richness" draw (name-level mean + event-level noise). The richness is
  what this project measures: on average, does the market overprice the
  event, and by how much.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
N_NAMES = 200
N_SECTORS = 10
EVENTS_PER_NAME = 26
EVENT_SPACING_DAYS = 63
START_BUFFER_DAYS = 150
TOTAL_EVENTS = N_NAMES * EVENTS_PER_NAME  # 5,200

# Calibration constants (tuned against the simulator, never against the
# measured output directly; see README Findings for the attempt history).
BASE_VOL_LOG_MEAN = np.log(0.30)
BASE_VOL_LOG_STD = 0.22
BASE_VOL_MIN, BASE_VOL_MAX = 0.15, 0.65
EVENT_VOL_MULT_LOW, EVENT_VOL_MULT_HIGH = 1.6, 2.6

GLOBAL_RICHNESS_MEAN_POINTS = 0.7    # mean implied-vs-realized richness, in vol points; see README Findings
                                      # for why this landed below the 1.6-point claim after 3 calibration attempts
RICHNESS_CROSS_SECTIONAL_STD = 1.3   # per-name heterogeneity around the global mean
RICHNESS_EVENT_STD = 7.5             # per-event noise around a name's own mean richness

T_DOF = 4  # Student-t degrees of freedom for the earnings-day return
T_STD = (T_DOF / (T_DOF - 2)) ** 0.5  # std of a standard Student-t(4), = sqrt(2)
# E|X| for a standard Student-t(4) is systematically BELOW its own standard deviation
# (Jensen's inequality on the absolute value of any single draw; true for a normal too,
# E|X| = sigma*sqrt(2/pi) = 0.798*sigma, and more so for a fat-tailed t4). A single
# realized earnings-day move is therefore a downward-biased point estimate of the sigma
# that generated it, BEFORE any genuine implied-vs-realized richness exists. The first
# build attempt priced the straddle off a DEBIASED implied vol (dividing out this factor
# before pricing) and measured a gross Sharpe of -0.45: debiasing the PRICING side
# understates the true event variance actually used to generate the path, so the straddle
# was systematically underpriced relative to the risk it carried, losing money regardless
# of richness. Fix: pricing uses the true, undebiased event_vol_scale (below); this
# constant instead corrects the REPORTED realized-move statistic (evrp.pnl adds a
# `realized_event_vol_points_debiased` column) so the gap claim compares like with like.
# See README Findings.
T4_ABS_MEAN_OVER_STD = 0.7068

TRAILING_LOOKBACK_DAYS = 90   # span used for the diffusive trailing-vol estimate (overlaps prior event)
TRAILING_SHORT_DAYS = 20      # shorter trailing window, used for a vol-of-vol feature


@dataclass(frozen=True)
class Panel:
    """A fully simulated panel: daily prices/returns per name, plus an event table."""

    names: pd.DataFrame        # one row per name: name_id, sector_id, base_vol, event_vol_scale, richness_mean
    returns: np.ndarray        # (N_NAMES, T_DAYS) daily log returns
    prices: np.ndarray         # (N_NAMES, T_DAYS) daily close prices, price[.,0] = 100.0
    is_event_day: np.ndarray  # (N_NAMES, T_DAYS) bool
    events: pd.DataFrame       # one row per simulated event (5,200 rows)


def _simulate_name_path(rng: np.random.Generator, base_vol: float, event_vol_scale: float,
                         event_days: np.ndarray, t_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (log_returns, prices, is_event_day) for one name over t_days trading days."""
    daily_diffusive_sigma = base_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    log_returns = rng.normal(0.0, daily_diffusive_sigma, size=t_days)

    is_event = np.zeros(t_days, dtype=bool)
    is_event[event_days] = True

    # Earnings-day return: Student-t(4), scaled so its STANDARD DEVIATION equals
    # event_vol_scale / sqrt(252) (the name's true 1-day event vol).
    event_daily_sigma = event_vol_scale / np.sqrt(TRADING_DAYS_PER_YEAR)
    t_draws = rng.standard_t(T_DOF, size=len(event_days))
    event_returns = t_draws * (event_daily_sigma / T_STD)
    log_returns[event_days] = event_returns

    prices = 100.0 * np.exp(np.cumsum(log_returns))
    prices = np.concatenate(([100.0], prices))  # prepend day-0 price, length t_days + 1
    return log_returns, prices, is_event


def _trailing_diffusive_vol_points(log_returns: np.ndarray, is_event_day: np.ndarray,
                                    as_of_day: int, lookback_days: int) -> float:
    """Annualized trailing realized vol (in vol points) over [as_of_day-lookback_days, as_of_day-1],
    excluding any day flagged as an event day for this name. Returns NaN if fewer than 5
    non-event observations are available (only possible very early in a name's history).
    """
    start = max(as_of_day - lookback_days, 0)
    window_returns = log_returns[start:as_of_day]
    window_is_event = is_event_day[start:as_of_day]
    clean = window_returns[~window_is_event]
    if clean.size < 5:
        return float("nan")
    daily_std = float(np.std(clean, ddof=1))
    return daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0


def build_panel(seed: int = 20260301) -> Panel:
    rng = np.random.default_rng(seed)

    sector_ids = np.arange(N_NAMES) % N_SECTORS
    base_vols = np.clip(
        rng.lognormal(BASE_VOL_LOG_MEAN, BASE_VOL_LOG_STD, size=N_NAMES),
        BASE_VOL_MIN, BASE_VOL_MAX,
    )
    event_vol_mults = rng.uniform(EVENT_VOL_MULT_LOW, EVENT_VOL_MULT_HIGH, size=N_NAMES)
    event_vol_scales = base_vols * event_vol_mults
    richness_means = rng.normal(GLOBAL_RICHNESS_MEAN_POINTS, RICHNESS_CROSS_SECTIONAL_STD, size=N_NAMES)
    phase_offsets = rng.integers(0, EVENT_SPACING_DAYS, size=N_NAMES)

    names_df = pd.DataFrame({
        "name_id": np.arange(N_NAMES),
        "sector_id": sector_ids,
        "base_vol": base_vols,
        "event_vol_scale": event_vol_scales,
        "richness_mean_points": richness_means,
        "phase_offset": phase_offsets,
    })

    max_event_day = START_BUFFER_DAYS + phase_offsets.max() + EVENT_SPACING_DAYS * (EVENTS_PER_NAME - 1)
    t_days = int(max_event_day + 5)  # +5 trading days of buffer past the last exit

    returns = np.zeros((N_NAMES, t_days))
    prices = np.zeros((N_NAMES, t_days + 1))
    is_event_day = np.zeros((N_NAMES, t_days), dtype=bool)

    event_rows = []
    for i in range(N_NAMES):
        event_days_i = START_BUFFER_DAYS + phase_offsets[i] + EVENT_SPACING_DAYS * np.arange(EVENTS_PER_NAME)
        log_ret, px, is_evt = _simulate_name_path(rng, base_vols[i], event_vol_scales[i], event_days_i, t_days)
        returns[i] = log_ret
        prices[i] = px
        is_event_day[i] = is_evt

        name_richness_mean = richness_means[i]
        for k, event_day in enumerate(event_days_i):
            entry_day = int(event_day) - 1   # close of session before the print
            exit_day = int(event_day) + 1     # next session ("open", approximated as that day's close)

            trailing_vol_long = _trailing_diffusive_vol_points(log_ret, is_evt, entry_day, TRAILING_LOOKBACK_DAYS)
            trailing_vol_short = _trailing_diffusive_vol_points(log_ret, is_evt, entry_day, TRAILING_SHORT_DAYS)

            event_richness = rng.normal(name_richness_mean, RICHNESS_EVENT_STD)
            implied_event_vol_points = event_vol_scales[i] * 100.0 + event_richness
            realized_event_return = log_ret[int(event_day)]
            realized_event_vol_points = abs(realized_event_return) / np.sqrt(1.0 / TRADING_DAYS_PER_YEAR) * 100.0

            event_rows.append({
                "event_id": len(event_rows),
                "name_id": i,
                "sector_id": sector_ids[i],
                "event_seq": k,  # 0-based occurrence index for this name, used for causal/prequential features
                "event_day": int(event_day),
                "entry_day": entry_day,
                "exit_day": exit_day,
                # prices[i] has length t_days + 1: prices[i][0] is the pre-day-0 price,
                # prices[i][d + 1] is the close of trading day d. Hence the +1 offsets.
                "s_entry": prices[i][entry_day + 1],
                "s_mid": prices[i][event_day + 1],
                "s_exit": prices[i][exit_day + 1],
                "trailing_vol_long_points": trailing_vol_long,
                "trailing_vol_short_points": trailing_vol_short,
                "implied_event_vol_points": implied_event_vol_points,
                "realized_event_return": realized_event_return,
                "realized_event_vol_points": realized_event_vol_points,
                "diffusive_vol_1day_points": base_vols[i] * 100.0,
            })

    events_df = pd.DataFrame(event_rows)
    assert len(events_df) == TOTAL_EVENTS, f"expected {TOTAL_EVENTS} events, got {len(events_df)}"
    return Panel(names=names_df, returns=returns, prices=prices, is_event_day=is_event_day, events=events_df)
