"""Pre-event features for the LightGBM model.

Every feature here is either (a) known at the moment of trade entry (the
session before the print: trailing vols, the quoted implied vols, sector),
or (b) a CAUSAL, per-name expanding statistic built only from that name's
STRICTLY PRIOR events (event_seq' < event_seq), via a groupby-expanding-mean
shifted by one row. No feature is allowed to see the current event's own
outcome (its realized return, its gap, its net P&L) or any later event's
outcome for the same name. tests/test_features_causal.py checks this
directly by permuting a later event's label and asserting an earlier
event's feature vector does not change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "trailing_vol_long_points",
    "trailing_vol_short_points",
    "vol_term_ratio",
    "implied_event_vol_points",
    "total_implied_vol_points",
    "sector_id",
    "event_seq",
    "name_hist_mean_gap",
    "name_hist_mean_ret_net",
    "name_hist_n_prior",
]

TARGET_COLUMN = "ret_net"


def _causal_expanding_mean(group: pd.Series) -> pd.Series:
    """Expanding mean of `group`, shifted by one so row k only sees rows < k."""
    return group.expanding().mean().shift(1)


def build_feature_frame(priced_events: pd.DataFrame) -> pd.DataFrame:
    """priced_events must already have pricing columns from pnl.price_events_frame
    (gap, ret_net, ...) and must be sortable by (name_id, event_seq).
    Returns a new frame, one row per event, with FEATURE_COLUMNS + TARGET_COLUMN + event_id.
    """
    df = priced_events.sort_values(["name_id", "event_seq"]).reset_index(drop=True).copy()
    # gap_points (debiased realized vs implied) is already computed by pnl.price_events_frame.
    df["vol_term_ratio"] = df["trailing_vol_short_points"] / df["trailing_vol_long_points"]

    grouped = df.groupby("name_id", group_keys=False)
    df["name_hist_mean_gap"] = grouped["gap_points"].apply(_causal_expanding_mean).reset_index(drop=True)
    df["name_hist_mean_ret_net"] = grouped["ret_net"].apply(_causal_expanding_mean).reset_index(drop=True)
    df["name_hist_n_prior"] = grouped.cumcount()

    # First event per name has no prior history; fill with the pooled prior-event mean
    # computed only from OTHER names that already have history at that point would require a
    # cross-sectional-as-of-time join we are not building here, so this project uses the
    # simpler, slightly conservative fallback of filling with 0 (no stated edge), which is
    # itself a disclosed approximation (see README Limitations).
    df["name_hist_mean_gap"] = df["name_hist_mean_gap"].fillna(0.0)
    df["name_hist_mean_ret_net"] = df["name_hist_mean_ret_net"].fillna(0.0)

    cols = ["event_id"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    return df[cols].reset_index(drop=True)
