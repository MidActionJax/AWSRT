#!/usr/bin/env python3
"""Test whether the LASCO-recalibrated magnitude formula (see
lasco_catalog.py: A=403.97, B=0.2177, fit on 303 real Richardson/Cane-
verified pairs spanning 1996-2026, vs. the currently-locked
estimate_earth_vmax's 12-pair, single-year fit) improves the full locked
pipeline -- WITHOUT the overfitting risk found earlier today in the
local (this-project's-own-43-events) recalibration, since this fit is
external and drawn from an independent, 25x larger, multi-solar-cycle
sample rather than tuned to our own small test window.

    python lasco_magnitude_test.py
"""
import numpy as np

from eval_harness import BIN_S, DISTURBED_KMS, fetch, to_bin_map
from donki_features import (arrival_feature, build_calibrated_index,
                             filter_earth_directed, load_events)
from magnitude_recalibration_test import corrected_series_with, per_event_compare
import phase2_model as p2
import final_correction as fc

LASCO_A, LASCO_B = 403.97, 0.2177  # from lasco_catalog.py, n=303, 1996-2026


def lasco_estimate_vmax(speed_kms):
    return LASCO_A + LASCO_B * speed_kms


def lasco_candidates(bins):
    donki_events = load_events("donki_cme_2026.json")
    donki_fast = [e for e in donki_events if e["speed"] >= fc.CME_MIN_SPEED_KMS]
    donki_filtered = filter_earth_directed(donki_fast, max_cmd=fc.EARTH_DIRECTED_MAX_CMD,
                                            include_unknown=True)
    arrivals = build_calibrated_index(donki_filtered)
    has_candidate = np.zeros(len(bins), dtype=bool)
    predicted_vmax = np.full(len(bins), np.nan)
    for i, b in enumerate(bins):
        offset_h, speed, has = arrival_feature(int(b) * BIN_S, arrivals)
        if has:
            has_candidate[i] = True
            predicted_vmax[i] = float(np.clip(lasco_estimate_vmax(speed),
                                               fc.MAGNITUDE_MIN_VMAX, fc.MAGNITUDE_MAX_VMAX))
    return has_candidate, predicted_vmax


def _event_pairs(bins, events, arrivals):
    pairs = []
    for ei, (start_b, end_b, peak) in enumerate(events):
        t_onset = int(start_b) * BIN_S
        offset_h, speed, has = arrival_feature(t_onset, arrivals)
        if not has:
            continue
        pairs.append((speed, peak, ei))
    return pairs


def _bin_event_map(bins, events):
    bin_event = np.full(len(bins), -1, dtype=int)
    for ei, (start_b, end_b, peak) in enumerate(events):
        mask = (bins >= start_b) & (bins <= end_b)
        bin_event[mask] = ei
    return bin_event


def lasco_intercept_shift_candidates(bins, events, arrivals):
    """Keeps the LASCO fit's robust, 303-pair, multi-solar-cycle SLOPE
    (real physical scaling, not overfit to our own small window) and only
    LOO-shifts the intercept to this project's own local severity level --
    same mechanism as final_correction's other LOO folds, applied to the
    magnitude formula specifically."""
    has_candidate = np.zeros(len(bins), dtype=bool)
    predicted_vmax = np.full(len(bins), np.nan)

    pairs = _event_pairs(bins, events, arrivals)
    if len(pairs) < 4:
        return has_candidate, predicted_vmax

    pairs_arr = np.array([(s, p) for s, p, _ in pairs])
    speeds, peaks = pairs_arr[:, 0], pairs_arr[:, 1]
    ev_idx = np.array([e for _, _, e in pairs])
    resid = peaks - np.array([lasco_estimate_vmax(s) for s in speeds])

    def shift_excluding(event_i):
        mask = ev_idx != event_i
        if mask.sum() < 3:
            return None
        return float(resid[mask].mean())

    bin_event = _bin_event_map(bins, events)
    for i, b in enumerate(bins):
        t_unix = int(b) * BIN_S
        offset_h, speed, has = arrival_feature(t_unix, arrivals)
        if not has:
            continue
        has_candidate[i] = True
        shift = shift_excluding(bin_event[i]) if bin_event[i] >= 0 else shift_excluding(-999)
        if shift is None:
            predicted_vmax[i] = np.nan
            has_candidate[i] = False
            continue
        pred = lasco_estimate_vmax(speed) + shift
        predicted_vmax[i] = float(np.clip(pred, fc.MAGNITUDE_MIN_VMAX, fc.MAGNITUDE_MAX_VMAX))
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

    print("Building CURRENT locked pipeline (12-pair external calibration)...")
    pred_locked, _, has_c_locked, _ = fc.loo_corrected_series(
        bins, X_all, obs_all, blend_all, disturbed_mask_all, events)

    print("Building LASCO-recalibrated pipeline (303-pair, 1996-2026 external calibration)...")
    has_c_lasco, pred_vmax_lasco = lasco_candidates(bins)
    pred_lasco = corrected_series_with(has_c_lasco, pred_vmax_lasco, bins, X_all, obs_all,
                                        blend_all, disturbed_mask_all, events)

    print("Building LASCO-slope + local-intercept hybrid pipeline...")
    donki_events = load_events("donki_cme_2026.json")
    donki_fast = [e for e in donki_events if e["speed"] >= fc.CME_MIN_SPEED_KMS]
    donki_filtered = filter_earth_directed(donki_fast, max_cmd=fc.EARTH_DIRECTED_MAX_CMD,
                                            include_unknown=True)
    arrivals = build_calibrated_index(donki_filtered)
    has_c_hybrid, pred_vmax_hybrid = lasco_intercept_shift_candidates(bins, events, arrivals)
    pred_hybrid = corrected_series_with(has_c_hybrid, pred_vmax_hybrid, bins, X_all, obs_all,
                                         blend_all, disturbed_mask_all, events)

    print()
    for name, pred in (("locked (12-pair calib)", pred_locked),
                       ("LASCO-recalibrated (303-pair calib)", pred_lasco),
                       ("LASCO slope + local intercept (hybrid)", pred_hybrid)):
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
    per_event_compare(bins, obs_all, pred_locked, pred_lasco, events,
                       "locked", "lasco-recal")
    per_event_compare(bins, obs_all, pred_locked, pred_hybrid, events,
                       "locked", "hybrid")


if __name__ == "__main__":
    main()
