"""LightGBM regressor: predicts the NET return of a delta-hedged short
straddle (ret_net = net P&L / spot), from pre-event information only.

Why this target. The trade decision a desk actually faces is not "will
implied exceed realized" (that is true on average by construction here,
see Findings) but "is THIS event's straddle worth putting on once hedging
and spread costs are included." Predicting ret_net directly, rather than
the raw implied-vs-realized gap, bakes the cost structure into the
prediction target, so a downstream filter ("only trade the top-scored
half") is answering the right question.

Validated with evrp.splits.purged_kfold (event-aware purge + embargo), out
of sample R^2 per fold, pooled. See README Findings for the measured value
and evrp.splits / tests/test_leakage_naive_vs_purged.py for why a naive
split would overstate it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

from .features import FEATURE_COLUMNS, TARGET_COLUMN
from .splits import purged_kfold

CATEGORICAL_FEATURES = ["sector_id"]

LGBM_PARAMS = dict(
    n_estimators=200,
    num_leaves=15,
    max_depth=5,
    learning_rate=0.05,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=13,
    verbosity=-1,
)


def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
        categorical_feature=[FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES],
    )
    return model.predict(test_df[FEATURE_COLUMNS])


def purged_cv_r2_with_events(
    feature_frame: pd.DataFrame,
    events_with_spans: pd.DataFrame,
    lookback_days: int,
    n_folds: int = 5,
    embargo_days: int = 5,
    seed: int = 7,
) -> dict:
    """feature_frame and events_with_spans must have the SAME row order (both produced
    from the same sort) and events_with_spans must have entry_day/exit_day columns.
    """
    assert len(feature_frame) == len(events_with_spans)
    all_preds = np.full(len(feature_frame), np.nan)
    all_truth = feature_frame[TARGET_COLUMN].to_numpy()
    fold_r2 = []

    for train_idx, test_idx in purged_kfold(events_with_spans, n_folds, lookback_days, embargo_days, seed):
        if len(test_idx) == 0 or len(train_idx) < 50:
            continue
        train_df = feature_frame.iloc[train_idx]
        test_df = feature_frame.iloc[test_idx]
        preds = _fit_predict(train_df, test_df)
        all_preds[test_idx] = preds
        fold_r2.append(r2_score(test_df[TARGET_COLUMN], preds))

    mask = ~np.isnan(all_preds)
    pooled_r2 = r2_score(all_truth[mask], all_preds[mask])
    return {
        "pooled_r2": float(pooled_r2),
        "fold_r2": [float(x) for x in fold_r2],
        "n_scored": int(mask.sum()),
        "predictions": all_preds,
    }
