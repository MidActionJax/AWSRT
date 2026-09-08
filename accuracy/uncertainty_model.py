#!/usr/bin/env python3
"""Phase 2 reframed: instead of correcting the exact speed value during a
disturbance (tried and not solved across two full days -- see
accuracy_improvement_tracking.txt's "MAJOR CHECKPOINT, 2026-09-12"),
classify regime (quiet vs. disturbed) causally in real time, and size an
uncertainty band around the Phase 1b blend's forecast accordingly.

Why this is more tractable than exact correction: predicting the exact
future trajectory of a specific disturbance needs to generalize from
only ~10-15 independent historical events -- a genuine small-N problem,
confirmed by two days of consistent negative results across many
approaches. Classifying CURRENT regime (quiet/disturbed) from the same
causal features is a different, easier statistical task: even though the
underlying independent events are the same ~10-15, there are thousands of
individual 10-minute samples of "what does a disturbed moment's feature
vector look like" to learn from, not just ~15 trajectories to memorize.
This doesn't fix the exact-value problem; it turns "the model might be
badly wrong right now and we don't say so" into "the model tells you when
to trust it less" -- a real, honest, deliverable product improvement
using signals already built and validated this project (AWSoM-R
trend/volatility/disagreement, DONKI activity/arrival features).

    python uncertainty_model.py
    python uncertainty_model.py --donki donki_cme_2026.json --donki-mode activity

Validates via the same leave-one-event-out split as phase2_model.py, but
scores DISCRIMINATION (does high predicted risk actually coincide with
genuine disturbance on held-out events) and CALIBRATION (does the sized
band actually contain the truth as often as intended), not RMSE/corr of
a point forecast.
"""
from __future__ import annotations

import argparse
import datetime

import numpy as np
from scipy.optimize import minimize

from eval_harness import BIN_S, DISTURBED_KMS, fetch, to_bin_map
from donki_features import (build_calibrated_index,
                             filter_earth_directed, load_events)
import early_warning as ew
import phase2_model as p2


# logistic_fit/predict_proba MOVED to phase2_model.py 2026-09-17 (so
# early_warning.py can use them without a circular import -- this module
# already does `import early_warning as ew`). Re-exported here under the
# same names so existing callers (final_correction.py,
# magnitude_recalibration_test.py: `from uncertainty_model import
# predict_proba, logistic_fit`) keep working unchanged.
from phase2_model import logistic_fit, predict_proba  # noqa: E402


def band_width(residuals, coverage=0.8):
    """Half-width such that `coverage` fraction of |residuals| fall inside
    it, empirically (no normality assumption -- residuals here are known
    from earlier work to be non-Gaussian, e.g. the disturbed regime's
    large negative bias)."""
    return float(np.quantile(np.abs(residuals), coverage))


def ew_predictions(hours):
    """Leave-one-event-out predicted future speed (anchor + learned delta,
    early_warning.py's model) for every bin where a genuine forward
    prediction is possible, keyed by bin index.

    This is a DECISION-LEVEL combination, not a feature-level one: feeding
    early_warning's own prediction into the classifier's feature vector
    was tried and added nothing (both models share the same underlying
    causal inputs, so a logistic/ridge model can already form that
    combination internally if it were useful -- see
    accuracy_improvement_tracking.txt LOG, 2026-09-12). What DOES help is
    treating early_warning as an independent trigger: it sees a genuine
    forward horizon the classifier's now-only features can't, so its OR
    can catch onsets the classifier alone misses. See the same LOG entry
    for the validated combined-trigger result.
    """
    bins, X_all, anchor, future_v, model_bins, obs_bins, donki_events, bfield = ew._dataset(hours)
    target = future_v - anchor
    events = p2.event_windows(obs_bins, threshold=600.0)

    pred = anchor.copy()
    held_out_mask = np.zeros(len(bins), dtype=bool)
    for start_b, end_b, peak in events:
        test_mask = (bins >= start_b) & (bins <= end_b)
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        mean, std, coef, intercept = p2.ridge_fit(X_all[train_mask], target[train_mask], ew.RIDGE_ALPHA)
        pred_delta = p2.ridge_predict(X_all[test_mask], mean, std, coef, intercept)
        pred[test_mask] = anchor[test_mask] + pred_delta
        held_out_mask |= test_mask
    mean, std, coef, intercept = p2.ridge_fit(X_all, target, ew.RIDGE_ALPHA)
    non_event = ~held_out_mask
    pred[non_event] = anchor[non_event] + p2.ridge_predict(X_all[non_event], mean, std, coef, intercept)

    return dict(zip(bins.tolist(), pred.tolist()))


def apply_to_live(args):
    """Apply the band-calibration model to the current live point, the way
    final_correction.py --live and early_warning.py --live already do for
    their own outputs. Previously this script only validated (LOO); there
    was no way to ask "what does the band say RIGHT NOW", which is what a
    real deployment (and combined_model.py) actually needs.

    Only supports the feature combination actually validated/recommended
    (base features + optional --donki activity mode + optional --bfield);
    --donki-mode calibrated_arrival isn't wired up here since the
    recommended config doesn't use it.
    """
    if args.donki and args.donki_mode in ("calibrated_arrival", "both"):
        raise SystemExit("uncertainty_model.py --live does not support "
                          "--donki-mode calibrated_arrival/both yet; use "
                          "--donki-mode activity or drop --donki")

    p2.FEATURES = ("model_v", "trend_3h", "trend_6h", "volatility_24h", "disagreement")
    doc = fetch(args.url, args.json)
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])

    donki_events_raw = None
    if args.donki:
        donki_events_raw = load_events(args.donki)
        p2.FEATURES = p2.FEATURES + p2.DONKI_ACTIVITY_FEATURES

    bfield_bins = None
    if args.bfield:
        bfield_bins = {"b": to_bin_map(obs["t"], obs["b"]),
                       "bz": to_bin_map(obs["t"], obs["bz"]),
                       "n": to_bin_map(obs["t"], obs["n"])}
        p2.FEATURES = p2.FEATURES + p2.BFIELD_FEATURES

    bins, data = p2.build_dataset(model_bins, obs_bins, donki_events=donki_events_raw,
                                   bfield_bins=bfield_bins)
    X_all = p2.feature_matrix(data)
    obs_all = data["obs_v"]
    blend_all = data["blend_v"]
    y_all = (obs_all > DISTURBED_KMS).astype(float)

    mean, std, w, b = logistic_fit(X_all, y_all, args.alpha)
    residual = obs_all - blend_all
    narrow_half = band_width(residual[y_all == 0], args.coverage)
    wide_half = band_width(residual[y_all == 1], args.coverage)

    b0, live_row = p2.live_features(model_bins, obs_bins, bfield_bins=bfield_bins,
                                     donki_events=donki_events_raw if args.donki else None)
    mv = live_row["model_v"]
    blend_v = live_row["_blend_v"]

    x_live = p2.live_feature_vector(live_row, p2.FEATURES)
    proba_live = predict_proba(x_live, mean, std, w, b)[0]
    triggered = bool(proba_live > args.risk_threshold)

    ew_result = None
    ew_triggered = False
    if args.early_warning:
        ew_result = ew.apply_to_live(args.ew_hours)
        ew_triggered = ew_result["forecast"] > args.ew_threshold
        triggered = triggered or ew_triggered

    band_half = wide_half if triggered else narrow_half
    when = datetime.datetime.utcfromtimestamp(b0 * BIN_S)
    lower, upper = blend_v - band_half, blend_v + band_half

    print(f"\nLive point: {when} UTC")
    print(f"  raw model:            {mv:.1f} km/s")
    print(f"  blend:                {blend_v:.1f} km/s")
    print(f"  P(disturbed):         {proba_live:.3f}  (risk-threshold {args.risk_threshold:g})")
    if args.early_warning:
        print(f"  early-warning({args.ew_hours:g}h):  {ew_result['forecast']:.1f} km/s  "
              f"(threshold {args.ew_threshold:g} km/s, triggered={ew_triggered})")
    print(f"  regime flagged risky: {triggered}")
    print(f"  band half-width:      {band_half:.1f} km/s "
          f"({'WIDE (disturbed)' if triggered else 'narrow (quiet)'})")
    print(f"  {args.coverage*100:.0f}% interval:        [{lower:.1f}, {upper:.1f}] km/s")

    return {"when": when, "raw_model": mv, "blend": blend_v, "proba_disturbed": proba_live,
            "triggered": triggered, "band_half": band_half, "lower": lower, "upper": upper,
            "early_warning": ew_result}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--donki", default=None)
    ap.add_argument("--donki-mode", choices=("activity", "calibrated_arrival", "both"),
                     default="activity")
    ap.add_argument("--coverage", type=float, default=0.8,
                     help="target band coverage, e.g. 0.8 = 80% interval")
    ap.add_argument("--risk-threshold", type=float, default=0.3,
                     help="predicted P(disturbed) above this uses the wide band")
    ap.add_argument("--bfield", action="store_true",
                     help="add real-time observed B/Bz/n features (see "
                          "phase2_model.BFIELD_FEATURES) -- these produced the "
                          "single biggest AUC jump found across two days when "
                          "added to final_correction.py; untested here until now")
    ap.add_argument("--early-warning", action="store_true",
                     help="OR-combine this classifier's risk trigger with "
                          "early_warning.py's genuine forward-horizon forecast: "
                          "flag risky whenever EITHER P(disturbed)>risk-threshold "
                          "OR the early-warning forecast exceeds --ew-threshold. "
                          "Decision-level combination -- validated to catch onsets "
                          "the classifier alone misses (disturbed-row coverage "
                          "71.2%->80.0%, see accuracy_improvement_tracking.txt LOG, "
                          "2026-09-12)")
    ap.add_argument("--ew-hours", type=float, default=6.0,
                     help="early-warning forecast lead time in hours (validated "
                          "sweet spot 6-12, default 6)")
    ap.add_argument("--ew-threshold", type=float, default=DISTURBED_KMS,
                     help="early-warning forecast above this (km/s) counts as a "
                          "trigger; default matches DISTURBED_KMS (500). Sweeping "
                          "this trades disturbed-row coverage against overall "
                          "overshoot -- 500 keeps most of the coverage gain at the "
                          "cost of a few points of overall overshoot past target; "
                          "raising it toward 600-650 tracks the target more "
                          "tightly but gives back most of the gain (see LOG)")
    ap.add_argument("--live", action="store_true",
                     help="apply to the current live point instead of "
                          "leave-one-event-out validation")
    args = ap.parse_args()

    if args.live:
        apply_to_live(args)
        return

    doc = fetch(args.url, args.json)
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])

    p2.FEATURES = ("model_v", "trend_3h", "trend_6h", "volatility_24h", "disagreement")
    donki_arrivals = donki_events_raw = None
    if args.donki:
        donki_events = load_events(args.donki)
        if args.donki_mode in ("calibrated_arrival", "both"):
            filtered = filter_earth_directed(donki_events, max_cmd=60, include_unknown=False)
            donki_arrivals = build_calibrated_index(filtered)
            p2.FEATURES = p2.FEATURES + p2.DONKI_ARRIVAL_FEATURES
        if args.donki_mode in ("activity", "both"):
            donki_events_raw = donki_events
            p2.FEATURES = p2.FEATURES + p2.DONKI_ACTIVITY_FEATURES

    bfield_bins = None
    if args.bfield:
        bfield_bins = {"b": to_bin_map(obs["t"], obs["b"]),
                       "bz": to_bin_map(obs["t"], obs["bz"]),
                       "n": to_bin_map(obs["t"], obs["n"])}
        p2.FEATURES = p2.FEATURES + p2.BFIELD_FEATURES
    bins, data = p2.build_dataset(model_bins, obs_bins, donki_arrivals, donki_events_raw,
                                   bfield_bins=bfield_bins)
    print(f"dataset: {len(bins)} rows, features: {p2.FEATURES}\n")

    X_all = p2.feature_matrix(data)
    obs_all = data["obs_v"]
    blend_all = data["blend_v"]
    y_all = (obs_all > DISTURBED_KMS).astype(float)
    print(f"{int(y_all.sum())} disturbed rows ({y_all.mean()*100:.1f}%), "
          f"{int((1-y_all).sum())} quiet rows\n")

    events = p2.event_windows(obs_bins, threshold=600.0)

    # leave-one-event-out: held-out P(disturbed) for every event row
    held_out_proba = np.full(len(bins), np.nan)
    held_out_mask = np.zeros(len(bins), dtype=bool)
    for start_b, end_b, peak in events:
        test_mask = (bins >= start_b) & (bins <= end_b)
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        mean, std, w, b = logistic_fit(X_all[train_mask], y_all[train_mask], args.alpha)
        held_out_proba[test_mask] = predict_proba(X_all[test_mask], mean, std, w, b)
        held_out_mask |= test_mask

    # quiet rows: fit on everything (no held-out event to protect there)
    mean, std, w, b = logistic_fit(X_all, y_all, args.alpha)
    quiet_mask = ~held_out_mask
    full_proba = blend_all.copy() * 0  # placeholder, overwritten below
    full_proba = np.where(held_out_mask, held_out_proba,
                           predict_proba(X_all, mean, std, w, b))

    print("Coefficients (standardized, fit on full dataset, for interpretation):")
    for name, c in zip(p2.FEATURES, w):
        print(f"  {name:20s}  {c:+7.3f}")
    print(f"  {'intercept':20s}  {b:+7.3f}\n")

    # --- discrimination on held-out event rows: does high P(disturbed)
    # actually coincide with genuinely disturbed rows we've never trained on? ---
    ho_y = y_all[held_out_mask]
    ho_p = held_out_proba[held_out_mask]
    # simple rank-based AUC (Mann-Whitney U), no sklearn dependency
    pos, neg = ho_p[ho_y == 1], ho_p[ho_y == 0]
    if len(pos) and len(neg):
        auc = float(np.mean(pos[:, None] > neg[None, :]))
    else:
        auc = float("nan")
    print(f"Held-out discrimination (event rows only): AUC={auc:.3f} "
          f"(0.5=random, 1.0=perfect; distinguishes genuinely disturbed "
          f"from quiet moments the model never trained on)\n")

    # --- calibration: size a band per regime from real residuals, check coverage ---
    residual = obs_all - blend_all
    quiet_resid = residual[y_all == 0]
    dist_resid = residual[y_all == 1]
    narrow_half = band_width(quiet_resid, args.coverage)
    wide_half = band_width(dist_resid, args.coverage)
    print(f"Band half-widths at {args.coverage*100:.0f}% empirical coverage: "
          f"quiet regime={narrow_half:.1f} km/s, disturbed regime={wide_half:.1f} km/s\n")

    # apply band width based on the CLASSIFIER's held-out risk prediction,
    # not the true (unknowable-in-advance) label -- this is the deployable test
    predicted_disturbed = full_proba > args.risk_threshold
    band_half = np.where(predicted_disturbed, wide_half, narrow_half)
    covered = np.abs(residual) <= band_half
    print(f"Adaptive band (sized by PREDICTED regime, {args.risk_threshold:g} "
          f"threshold): actual coverage = {covered.mean()*100:.1f}% "
          f"(target {args.coverage*100:.0f}%)")
    fixed_half = band_width(residual, args.coverage)
    covered_fixed = np.abs(residual) <= fixed_half
    print(f"Fixed band for comparison (one width={fixed_half:.1f} km/s "
          f"for everything): coverage = {covered_fixed.mean()*100:.1f}%, "
          f"width={fixed_half:.1f} vs adaptive's {narrow_half:.1f}/{wide_half:.1f}")

    # the real payoff check: on TRULY disturbed rows specifically, does the
    # adaptive band actually widen when it needs to, vs the fixed band
    # being wrong (too narrow) exactly when it matters most?
    dist_mask = y_all == 1
    print(f"\nOn genuinely disturbed rows only ({dist_mask.sum()} rows):")
    print(f"  fixed band coverage:    {covered_fixed[dist_mask].mean()*100:.1f}% "
          f"(width {fixed_half:.1f} km/s, same everywhere)")
    print(f"  adaptive band coverage: {covered[dist_mask].mean()*100:.1f}% "
          f"(width {wide_half:.1f} km/s when correctly flagged as risky)")

    # --- decision-level combination: OR the classifier's trigger with
    # early_warning.py's independent forward-horizon forecast ---
    if args.early_warning:
        ew_pred_by_bin = ew_predictions(args.ew_hours)
        ew_pred_at_row = np.array([ew_pred_by_bin.get(b, -np.inf) for b in bins])
        ew_has_pred = np.isfinite(ew_pred_at_row)
        ew_triggered = ew_pred_at_row > args.ew_threshold
        predicted_disturbed_or = predicted_disturbed | ew_triggered
        band_half_or = np.where(predicted_disturbed_or, wide_half, narrow_half)
        covered_or = np.abs(residual) <= band_half_or

        print(f"\n--- combined trigger: classifier P(disturbed)>{args.risk_threshold:g} "
              f"OR early-warning({args.ew_hours:g}h)>{args.ew_threshold:g} km/s "
              f"({ew_has_pred.sum()}/{len(bins)} rows have an early-warning "
              f"prediction; rest fall back to classifier-only) ---")
        print(f"  overall coverage: {covered_or.mean()*100:.1f}% "
              f"(classifier-only was {covered.mean()*100:.1f}%, target "
              f"{args.coverage*100:.0f}%)")
        print(f"  flagged-risky rate: {predicted_disturbed_or.mean()*100:.1f}% "
              f"(classifier-only was {predicted_disturbed.mean()*100:.1f}%)")
        print(f"  on genuinely disturbed rows: {covered_or[dist_mask].mean()*100:.1f}% "
              f"(classifier-only was {covered[dist_mask].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
