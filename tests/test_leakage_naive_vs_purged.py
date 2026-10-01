"""Proves the purge matters, not just that it runs: a naive split lets a feature that
only looks informative because of time-overlap achieve suspiciously good out-of-sample
R^2, and the purged split collapses it back to roughly zero.

Construction: 60 time clusters, each a tight group of "events" (within a 1-day jitter
of each other, think several names reporting on the same earnings-heavy day) sharing an
independent latent shock Z_cluster ~ N(0, 1); `label` is Z_cluster plus small noise.
No engineered feature is used at all: the "model" is the simplest one capable of
exploiting train/test time-proximity the way a tree or kNN model implicitly can, a
1-nearest-neighbor-by-entry_day regressor that predicts a test event's label as the
label of whichever TRAINING event is closest to it in time. With a naive (unpurged)
K-fold, a test event's nearest training neighbor is routinely another member of its own
cluster (clusters are tight in time, folds are assigned event-by-event, not
cluster-by-cluster), so the 1-NN predictor effectively copies Z_cluster from a training
row and scores very well, a score that could never generalize to a genuinely new
cluster. The purge removes every training event whose span overlaps a test event's
span, which here means the whole shared cluster is removed from training whenever any
of it is held out, so the nearest surviving training neighbor belongs to an unrelated
cluster and the "prediction" carries no real information: R^2 should collapse.
"""
import numpy as np
import pandas as pd

from evrp.splits import purged_kfold, naive_kfold


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def _nearest_neighbor_predict(train_days, train_labels, test_days):
    """Predict each test event's label as the label of its nearest TRAINING event by
    entry_day. This is the simplest possible stand-in for a tree/kNN model's known
    tendency to exploit train/test time-proximity when nothing purges it away.
    """
    train_days = np.asarray(train_days)
    train_labels = np.asarray(train_labels)
    preds = np.empty(len(test_days))
    for i, day in enumerate(test_days):
        nearest = np.argmin(np.abs(train_days - day))
        preds[i] = train_labels[nearest]
    return preds


def _build_clustered_leak_dataset(n_clusters=60, events_per_cluster=8, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    event_id = 0
    cluster_spacing = 200  # wide enough that spans never touch an adjacent cluster
    for c in range(n_clusters):
        z = rng.normal(0.0, 1.0)
        cluster_start_day = 1000 + c * cluster_spacing
        for k in range(events_per_cluster):
            # All events in a cluster land within a 1-day jitter of each other (think:
            # several names reporting on the same earnings-heavy day), so every pair of
            # spans in the cluster genuinely overlaps regardless of which lookback is used.
            entry_day = cluster_start_day + int(rng.integers(0, 2))
            exit_day = entry_day + 2
            label = z + rng.normal(0.0, 0.05)
            rows.append({
                "event_id": event_id, "cluster_id": c, "name_id": c,
                "entry_day": entry_day, "exit_day": exit_day,
                "label": label,
            })
            event_id += 1
    return pd.DataFrame(rows)


def test_naive_split_overstates_leaky_feature_and_purge_fixes_it():
    df = _build_clustered_leak_dataset()
    lookback_days = 10  # wider than the within-cluster spacing (2 days), narrower than cluster_spacing (200)

    entry_days = df["entry_day"].to_numpy()
    labels = df["label"].to_numpy()

    naive_r2s = []
    for train_idx, test_idx in naive_kfold(len(df), n_folds=5, seed=11):
        preds = _nearest_neighbor_predict(entry_days[train_idx], labels[train_idx], entry_days[test_idx])
        naive_r2s.append(_r2(labels[test_idx], preds))
    naive_r2 = float(np.mean(naive_r2s))

    purged_r2s = []
    for train_idx, test_idx in purged_kfold(df, n_folds=5, lookback_days=lookback_days, embargo_days=0, seed=11):
        if len(train_idx) < 5:
            continue
        preds = _nearest_neighbor_predict(entry_days[train_idx], labels[train_idx], entry_days[test_idx])
        purged_r2s.append(_r2(labels[test_idx], preds))
    purged_r2 = float(np.mean(purged_r2s))

    # The naive split should look "suspiciously good" (same-cluster rows leak across the
    # train/test boundary); the purged split, with the whole cluster removed from train
    # whenever any of it is in test, should be far worse, because Z_cluster for an unseen
    # cluster is literally unlearnable from any other cluster's label.
    assert naive_r2 > 0.6, f"expected a suspiciously high naive R^2, got {naive_r2}"
    assert purged_r2 < 0.15, f"expected the purge to collapse R^2 near zero, got {purged_r2}"
    assert naive_r2 - purged_r2 > 0.5


def test_purge_removes_the_same_cluster_training_rows_it_claims_to():
    df = _build_clustered_leak_dataset()
    lookback_days = 10
    for train_idx, test_idx in purged_kfold(df, n_folds=5, lookback_days=lookback_days, embargo_days=0, seed=11):
        test_clusters = set(df["cluster_id"].to_numpy()[test_idx])
        train_clusters = set(df["cluster_id"].to_numpy()[train_idx])
        assert train_clusters.isdisjoint(test_clusters), "a purged fold must not train on any test event's cluster"
