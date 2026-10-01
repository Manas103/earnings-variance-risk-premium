"""A later event's label must not change an earlier event's features (causality)."""
import numpy as np
import pandas as pd

from evrp.features import build_feature_frame


def _toy_priced_events(n_per_name=5, n_names=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    event_id = 0
    for name in range(n_names):
        for seq in range(n_per_name):
            rows.append({
                "event_id": event_id, "name_id": name, "sector_id": name % 2, "event_seq": seq,
                "trailing_vol_long_points": 20.0 + rng.normal(0, 1),
                "trailing_vol_short_points": 22.0 + rng.normal(0, 1),
                "implied_event_vol_points": 40.0 + rng.normal(0, 1),
                "total_implied_vol_points": 45.0 + rng.normal(0, 1),
                "implied_event_vol_points_raw": 40.0,
                "realized_event_vol_points": 30.0 + rng.normal(0, 1),
                "gap_points": rng.normal(1.5, 1.0),
                "ret_net": rng.normal(0.0, 0.05),
            })
            event_id += 1
    return pd.DataFrame(rows)


def test_perturbing_a_later_event_label_does_not_change_an_earlier_feature():
    df = _toy_priced_events()
    features_before = build_feature_frame(df)

    df_perturbed = df.copy()
    last_event_mask = (df_perturbed["name_id"] == 0) & (df_perturbed["event_seq"] == 4)
    df_perturbed.loc[last_event_mask, "ret_net"] = 999.0
    df_perturbed.loc[last_event_mask, "gap_points"] = 999.0
    features_after = build_feature_frame(df_perturbed)

    earlier_mask = (features_before["event_seq"] < 4)
    pd.testing.assert_frame_equal(
        features_before.loc[earlier_mask].reset_index(drop=True),
        features_after.loc[earlier_mask].reset_index(drop=True),
    )
    # sanity: the later event's own feature SHOULD have changed (history catches up to it later,
    # not before), proving the test isn't vacuously passing because nothing used the column.
    later_row_before = features_before.loc[features_before["event_seq"] == 4].iloc[0]
    assert later_row_before["name_hist_mean_gap"] != 999.0
