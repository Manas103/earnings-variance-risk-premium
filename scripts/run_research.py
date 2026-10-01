"""End-to-end run: simulate the panel, price every event's delta-hedged short
straddle gross and net, build causal features, run the LightGBM model under
event-aware purged K-fold CV, and print every measured number this README's
claims are checked against. Deterministic (seeded); ~15-30s on this machine.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evrp.simulate import build_panel, N_NAMES, EVENTS_PER_NAME, EVENT_SPACING_DAYS, TRAILING_LOOKBACK_DAYS, TOTAL_EVENTS
from evrp.pnl import price_events_frame, sharpe_ratio
from evrp.features import build_feature_frame
from evrp.model import purged_cv_r2_with_events

TRADES_PER_YEAR = N_NAMES * 252 / EVENT_SPACING_DAYS


def main():
    t0 = time.time()
    print("=" * 70)
    print("Earnings Variance Risk Premium and Event Term-Structure Study")
    print("=" * 70)

    panel = build_panel()
    print(f"\nSimulated panel: {N_NAMES} names, {EVENTS_PER_NAME} events/name, "
          f"{len(panel.events)} total events (target {TOTAL_EVENTS})")
    assert len(panel.events) == TOTAL_EVENTS

    priced = price_events_frame(panel.events)

    gap = priced["gap_points"]
    print("\n--- Implied vs realized event move (debiased realized vol estimate) ---")
    print(f"mean gap: {gap.mean():.4f} vol points (claim: 1.6)")
    print(f"median gap: {gap.median():.4f} vol points")
    print(f"std gap: {gap.std():.4f} vol points")

    print("\n--- Variance decomposition identity check (whole panel) ---")
    from evrp.variance import variance_from_vol_points
    total_variance_2day = variance_from_vol_points(priced["total_implied_vol_points"], 2.0)
    resid = (priced["diffusive_variance"] + priced["event_variance"]) - total_variance_2day
    max_abs_identity_resid = float(np.max(np.abs(resid)))
    print(f"max |diffusive + event - total| across {len(priced)} events: {max_abs_identity_resid:.3e} (should be 0.0 exactly)")

    gross_sharpe = sharpe_ratio(priced["ret_gross"].to_numpy(), TRADES_PER_YEAR)
    net_sharpe = sharpe_ratio(priced["ret_net"].to_numpy(), TRADES_PER_YEAR)
    print("\n--- Delta-hedged short straddle P&L, gross vs net ---")
    print(f"trades/year used for annualization: {TRADES_PER_YEAR:.1f} ({N_NAMES} names x 252/{EVENT_SPACING_DAYS})")
    print(f"mean gross P&L per event: {priced['gross_pnl'].mean():.4f}   mean net P&L per event: {priced['net_pnl'].mean():.4f}")
    print(f"GROSS SHARPE: {gross_sharpe:.4f}  (claim: 1.42)")
    print(f"NET SHARPE:   {net_sharpe:.4f}  (claim: 0.38, after half-spread per option leg + daily hedging slippage)")

    print("\n--- LightGBM: predicting net straddle return from pre-event features only ---")
    features = build_feature_frame(priced)
    # events_for_spans must be in the SAME row order as `features` (both built from the
    # same sort by name_id, event_seq) for purged_kfold's span computation to line up.
    events_sorted = priced.sort_values(["name_id", "event_seq"]).reset_index(drop=True)
    events_for_spans = events_sorted[["entry_day", "exit_day", "name_id"]].reset_index(drop=True)

    cv = purged_cv_r2_with_events(features, events_for_spans, lookback_days=TRAILING_LOOKBACK_DAYS,
                                   n_folds=5, embargo_days=5, seed=7)
    print(f"purged 5-fold out-of-sample R^2 (ret_net): {cv['pooled_r2']:.4f}")
    print(f"per-fold R^2: {[round(x, 4) for x in cv['fold_r2']]}")
    print(f"events scored out of sample: {cv['n_scored']} / {len(features)}")

    # Bonus finding: does filtering to the model's top-scored half of events change net Sharpe?
    preds = cv["predictions"]
    scored_mask = ~np.isnan(preds)
    ret_net_aligned = features["ret_net"].to_numpy()
    median_score = np.median(preds[scored_mask])
    top_half_mask = scored_mask & (preds >= median_score)
    sharpe_top_half = sharpe_ratio(ret_net_aligned[top_half_mask], TRADES_PER_YEAR)
    sharpe_all_scored = sharpe_ratio(ret_net_aligned[scored_mask], TRADES_PER_YEAR)
    print(f"\nNet Sharpe, all out-of-sample scored events: {sharpe_all_scored:.4f}")
    print(f"Net Sharpe, top-half by predicted score:        {sharpe_top_half:.4f}")

    results = {
        "n_events": int(len(priced)),
        "n_names": N_NAMES,
        "mean_gap_points": float(gap.mean()),
        "max_abs_identity_resid": max_abs_identity_resid,
        "gross_sharpe": float(gross_sharpe),
        "net_sharpe": float(net_sharpe),
        "pooled_r2": cv["pooled_r2"],
        "fold_r2": cv["fold_r2"],
        "n_scored": cv["n_scored"],
        "net_sharpe_all_scored": float(sharpe_all_scored),
        "net_sharpe_top_half": float(sharpe_top_half),
    }
    out_path = Path(__file__).resolve().parents[1] / "docs" / "research_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
