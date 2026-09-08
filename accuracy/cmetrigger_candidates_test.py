#!/usr/bin/env python3
"""Tier 0 was closed 2026-09-14 on one consistent diagnosis: candidate
SELECTION (matching which real CME drove a specific disturbance), not the
magnitude/timing math applied afterward, is the actual bottleneck -- our
own from-scratch DONKI matching (any Earth-directed CME within a window)
gives ~100% candidate coverage regardless of whether the match is the true
driver, diluting the signal toward a generic/modest expected effect.

2026-09-16: found that cmetrigger (CLEAR's own real, already-operational
CME-triggering system) does real candidate selection -- proper DONKI
selection + GONG magnetogram + ROI-pair (sunspot polarity) matching + a
real GLSETUP flux-rope fit -- and ships 41 of its own reference decisions
in RealTimeCME/cmetrigger/reference/baseline_nbiro_2026 (Jan 18 - Jul 31
2026), extracted via build_cmetrigger_events.py into
cmetrigger_events_2026.json. This tests whether swapping THAT candidate
set in for our own crude one (same downstream shift/magnitude mechanism,
same LOO discipline) actually fixes the undershoot that closed Tier 0.

    python cmetrigger_candidates_test.py
"""
import numpy as np

from eval_harness import BIN_S, DISTURBED_KMS, fetch, to_bin_map
from donki_features import (arrival_feature, build_calibrated_index,
                             estimate_earth_vmax, filter_earth_directed, load_events)
from magnitude_recalibration_test import corrected_series_with, per_event_compare
import phase2_model as p2
import final_correction as fc

CMETRIGGER_EVENTS = "cmetrigger_events_2026.json"


def cmetrigger_candidates(bins):
    events = load_events(CMETRIGGER_EVENTS)
    filtered = filter_earth_directed(events, max_cmd=fc.EARTH_DIRECTED_MAX_CMD,
                                      include_unknown=True)
    arrivals = build_calibrated_index(filtered)
    has_candidate = np.zeros(len(bins), dtype=bool)
    predicted_vmax = np.full(len(bins), np.nan)
    for i, b in enumerate(bins):
        offset_h, speed, has = arrival_feature(int(b) * BIN_S, arrivals)
        if has:
            has_candidate[i] = True
            predicted_vmax[i] = float(np.clip(estimate_earth_vmax(speed),
                                               fc.MAGNITUDE_MIN_VMAX, fc.MAGNITUDE_MAX_VMAX))
    return has_candidate, predicted_vmax


def main():
    p2.FEATURES = fc.FEATURES
    doc = fetch()
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])
    events = p2.event_windows(obs_bins, threshold=600.0)

    bins, data = p2.build_dataset(model_bins, obs_bins, bfield_bins=fc._bfield_bins(obs))
    X_all = p2.feature_matrix(data)
    obs_all = data["obs_v"]
    blend_all = data["blend_v"]
    disturbed_mask_all = obs_all > DISTURBED_KMS

    print("Building CURRENT locked pipeline (own DONKI-catalog candidates)...")
    pred_locked, _, has_c_locked, _ = fc.loo_corrected_series(
        bins, X_all, obs_all, blend_all, disturbed_mask_all, events)

    print("Building CMETRIGGER-candidate pipeline (real production selection)...")
    has_c_cmt, pred_vmax_cmt = cmetrigger_candidates(bins)
    pred_cmt = corrected_series_with(has_c_cmt, pred_vmax_cmt, bins, X_all, obs_all,
                                      blend_all, disturbed_mask_all, events)

    n_dist = int(disturbed_mask_all.sum())
    print(f"\ncandidate coverage on disturbed rows: "
          f"locked(own DONKI)={int((has_c_locked & disturbed_mask_all).sum())}/{n_dist}  "
          f"cmetrigger={int((has_c_cmt & disturbed_mask_all).sum())}/{n_dist}\n")

    for name, pred in (("locked (own DONKI candidates)", pred_locked),
                       ("cmetrigger (real production candidates)", pred_cmt)):
        s_all = p2.stats(obs_all, pred)
        s_dist = p2.stats(obs_all[disturbed_mask_all], pred[disturbed_mask_all])
        s_quiet = p2.stats(obs_all[~disturbed_mask_all], pred[~disturbed_mask_all])
        print(f"{name}:")
        print(f"  all:      rmse={s_all['rmse']:6.1f}  mae={s_all['mae']:6.1f}  "
              f"bias={s_all['bias']:7.1f}  corr={s_all['corr']:6.3f}")
        print(f"  disturbed:rmse={s_dist['rmse']:6.1f}  mae={s_dist['mae']:6.1f}  "
              f"bias={s_dist['bias']:7.1f}  corr={s_dist['corr']:6.3f}")
        print(f"  quiet:    rmse={s_quiet['rmse']:6.1f}  mae={s_quiet['mae']:6.1f}  "
              f"bias={s_quiet['bias']:7.1f}  corr={s_quiet['corr']:6.3f}")
        print()

    print("Per-event reliability check (the actual bar for adoption):")
    per_event_compare(bins, obs_all, pred_locked, pred_cmt, events,
                       "locked", "cmetrigger")

    # cmetrigger's reference set only covers 2026-01-18 to 2026-07-31 --
    # restrict to that same window on BOTH pipelines for a fair comparison,
    # since coverage loss outside that range is a data-availability
    # confound, not evidence about candidate-selection quality
    import datetime
    cutoff_end = datetime.datetime(2026, 7, 31, 23, 0, tzinfo=datetime.timezone.utc).timestamp()
    window_mask = (bins.astype(float) * BIN_S) <= cutoff_end
    print(f"\n--- restricted to cmetrigger's own covered window "
          f"({int(window_mask.sum())}/{len(bins)} rows, through 2026-07-31) ---")
    dist_w = disturbed_mask_all & window_mask
    for name, pred in (("locked, in-window", pred_locked),
                       ("cmetrigger, in-window", pred_cmt)):
        s_dist_w = p2.stats(obs_all[dist_w], pred[dist_w])
        print(f"  {name}: disturbed rmse={s_dist_w['rmse']:6.1f}  "
              f"bias={s_dist_w['bias']:7.1f}  corr={s_dist_w['corr']:6.3f}")
    events_w = [(s, e, p) for s, e, p in events if e * BIN_S <= cutoff_end]
    print(f"  ({len(events_w)}/{len(events)} event windows fully in range)")
    per_event_compare(bins, obs_all, pred_locked, pred_cmt, events_w,
                       "locked", "cmetrigger")


if __name__ == "__main__":
    main()
