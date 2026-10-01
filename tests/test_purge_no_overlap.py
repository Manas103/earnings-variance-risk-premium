"""The purge actually removes the overlapping rows it claims to: counted directly,
not just asserted by construction.
"""
import numpy as np
import pandas as pd

from evrp.splits import compute_spans, count_overlapping_pairs, purged_kfold, naive_kfold


def _toy_events(n=600, spacing=63, seed=1):
    rng = np.random.default_rng(seed)
    name_id = np.repeat(np.arange(n // 26 + 1), 26)[:n]
    event_seq = np.tile(np.arange(26), n // 26 + 1)[:n]
    phase = rng.integers(0, spacing, size=n)
    event_day = 150 + phase + spacing * event_seq
    entry_day = event_day - 1
    exit_day = event_day + 1
    return pd.DataFrame({
        "event_id": np.arange(n),
        "name_id": name_id,
        "entry_day": entry_day,
        "exit_day": exit_day,
    })


def test_purged_kfold_has_zero_overlap_after_embargo():
    events = _toy_events()
    lookback = 90
    embargo = 5
    for train_idx, test_idx in purged_kfold(events, n_folds=5, lookback_days=lookback, embargo_days=embargo):
        train_spans = compute_spans(events.iloc[train_idx].reset_index(drop=True), lookback)
        test_spans = compute_spans(events.iloc[test_idx].reset_index(drop=True), lookback)
        embargoed = [type(s)(s.event_id, s.name_id, s.start - embargo, s.end + embargo) for s in test_spans]
        remaining = count_overlapping_pairs(train_spans, embargoed)
        assert remaining == 0


def test_naive_kfold_leaves_real_overlap_that_purge_removes():
    events = _toy_events()
    lookback = 90
    naive_overlap_counts = []
    purged_overlap_counts = []
    for train_idx, test_idx in naive_kfold(len(events), n_folds=5, seed=7):
        train_spans = compute_spans(events.iloc[train_idx].reset_index(drop=True), lookback)
        test_spans = compute_spans(events.iloc[test_idx].reset_index(drop=True), lookback)
        naive_overlap_counts.append(count_overlapping_pairs(train_spans, test_spans))

    for train_idx, test_idx in purged_kfold(events, n_folds=5, lookback_days=lookback, embargo_days=5, seed=7):
        train_spans = compute_spans(events.iloc[train_idx].reset_index(drop=True), lookback)
        test_spans = compute_spans(events.iloc[test_idx].reset_index(drop=True), lookback)
        purged_overlap_counts.append(count_overlapping_pairs(train_spans, test_spans))

    assert sum(naive_overlap_counts) > 0, "the toy panel should have real same-name adjacent-event overlap"
    assert sum(purged_overlap_counts) == 0
    assert sum(naive_overlap_counts) > sum(purged_overlap_counts)
