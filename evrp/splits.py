"""Event-aware purged K-fold splits.

Each event i has a SPAN: the trading-day interval its feature lookback
window and its trade outcome window together cover,

    span(i) = [entry_day(i) - lookback_days, exit_day(i) + embargo_days]

Two events "overlap" if their spans intersect. With 200 independent names
each firing quarterly, the pooled event set is not a single time series;
assigning events to folds by index (or randomly) routinely puts one name's
event i in train and that SAME name's adjacent event i+1 in test, and those
two spans overlap by construction (a 90-day trailing-vol lookback is wider
than the 63-trading-day gap between a name's own consecutive earnings
prints). A model fit on event i can see, through its own feature window,
realized returns that are also inside event i+1's outcome window, or vice
versa.

purged_kfold removes every training event whose span intersects any test
event's span in the current fold, which is the purge; it then also drops
any training event starting within `embargo_days` after the END of the
widest test interval touching it, which is the extra embargo buffer. Both
are tested directly in tests/test_purge_no_overlap.py by counting the
overlapping rows before and after.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventSpan:
    event_id: int
    name_id: int
    start: int
    end: int


def compute_spans(events: pd.DataFrame, lookback_days: int, embargo_days: int = 0) -> list[EventSpan]:
    spans = []
    for row in events.itertuples():
        start = row.entry_day - lookback_days
        end = row.exit_day + embargo_days
        # event_id is the positional row index, not a DataFrame column: compute_spans is always
        # called on a RangeIndex-reset frame (see purged_kfold's assertion), so row.Index is
        # exactly the position spans_by_pos is later indexed by.
        spans.append(EventSpan(event_id=row.Index, name_id=row.name_id, start=start, end=end))
    return spans


def _overlaps(a: EventSpan, b: EventSpan) -> bool:
    # Only events on the SAME underlying name can leak into each other: the only shared
    # state across events is a name's own trailing-vol lookback window, and no feature in
    # this project is cross-sectional (see evrp/features.py). Pooling the overlap check
    # across all 200 names' independent calendars purges almost everything (every day in
    # the full date range is inside *some* name's test window), which is what the first
    # version of this function did; see README Findings for the measured blow-up.
    if a.name_id != b.name_id:
        return False
    return a.start <= b.end and b.start <= a.end


def count_overlapping_pairs(train_spans: list[EventSpan], test_spans: list[EventSpan]) -> int:
    """Number of (train, test) span pairs that overlap. O(n*m); fine at this n."""
    count = 0
    for tr in train_spans:
        for te in test_spans:
            if _overlaps(tr, te):
                count += 1
    return count


def purged_kfold(
    events: pd.DataFrame,
    n_folds: int,
    lookback_days: int,
    embargo_days: int = 5,
    seed: int = 7,
):
    """Yield (train_idx, test_idx) pairs of positional indices into `events`
    (0..len(events)-1, in the row order of `events`, which must have a
    plain RangeIndex), purged for span overlap plus an embargo.

    Fold assignment is a fixed random partition of events into n_folds
    equal-ish groups (seeded), not a chronological walk-forward: with 200
    independent names firing on independent calendars, nothing forces test
    events to be "the most recent" ones, so a random K-fold assignment is
    the right generalization of purged K-fold cross-validation (Lopez de
    Prado) to this panel, rather than a single time cutoff.
    """
    assert isinstance(events.index, pd.RangeIndex) and events.index[0] == 0, "events must have a plain RangeIndex"
    n = len(events)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_of = np.empty(n, dtype=int)
    for fold_id, idx_chunk in enumerate(np.array_split(order, n_folds)):
        fold_of[idx_chunk] = fold_id

    spans = compute_spans(events, lookback_days=lookback_days, embargo_days=0)
    spans_by_pos = spans  # spans[i] corresponds to events.iloc[i]

    for fold_id in range(n_folds):
        test_mask = fold_of == fold_id
        test_idx = np.where(test_mask)[0]
        train_idx_all = np.where(~test_mask)[0]

        test_spans = [spans_by_pos[i] for i in test_idx]
        # Embargo: widen each test span on both sides by embargo_days when checking overlap
        # against candidate training events (prevents a training event that starts just after
        # a test event's window closes from still riding the same short-lived vol regime).
        embargoed_test_spans = [EventSpan(s.event_id, s.name_id, s.start - embargo_days, s.end + embargo_days) for s in test_spans]

        keep_train = []
        for i in train_idx_all:
            tr_span = spans_by_pos[i]
            if not any(_overlaps(tr_span, te) for te in embargoed_test_spans):
                keep_train.append(i)

        yield np.array(keep_train, dtype=int), test_idx


def naive_kfold(n: int, n_folds: int, seed: int = 7):
    """A plain random K-fold with NO purge, for comparison only (the leakage test)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_of = np.empty(n, dtype=int)
    for fold_id, idx_chunk in enumerate(np.array_split(order, n_folds)):
        fold_of[idx_chunk] = fold_id
    for fold_id in range(n_folds):
        test_mask = fold_of == fold_id
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        yield train_idx, test_idx
