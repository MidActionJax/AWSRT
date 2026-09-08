#!/usr/bin/env python3
"""Tier 3 item 9 (structured correction), amplitude half: test whether
recalibrating the CME-magnitude estimate on our OWN observed peaks (instead
of the original 12-pair external Richardson/Cane calibration) improves the
full locked pipeline. Diagnostic first (see accuracy_improvement_tracking.txt,
2026-09-16): the old estimate_earth_vmax() underpredicts real event peaks by
a mean of +78 km/s (43-event sample) -- a real, one-directional bias, not
noise. A per-event LOO refit on our own (donki_speed, actual_peak) pairs
cuts that error's RMSE from 96.8 to 47.0.

This script substitutes that LOO-refit magnitude estimate into
final_correction's own shift mechanism -- same event-based folds, same
blend/proba/gate logic -- so the comparison is apples to apples against the
currently-locked pipeline, not just the isolated peak-error metric above.

    python magnitude_recalibration_test.py
"""
from __future__ import annotations

import numpy as np

from eval_harness import BIN_S, DISTURBED_KMS, fetch, to_bin_map
from donki_features import (arrival_feature, build_calibrated_index,
                             filter_earth_directed, load_events,
                             estimate_earth_vmax)
from uncertainty_model import predict_proba, logistic_fit
import phase2_model as p2
import final_correction as fc


def _event_pairs(bins, events, arrivals):
    """(speed, actual_peak, event_index) for every event with a candidate
    at its own onset time."""
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


def recalibrated_candidates(bins, events, arrivals):
    """Like fc._cme_candidates(), but predicted_vmax comes from a per-event
    LOO linear refit on OUR OWN (donki_speed, actual_peak) pairs instead of
    the fixed external estimate_earth_vmax(). For each event fold, the
    refit excludes that fold's own pair(s), same LOO discipline as every
    other part of this pipeline.

    CAVEAT (see accuracy_improvement_tracking.txt, 2026-09-16 update): the
    refit slope collapses to nearly flat (donki speed only weakly predicts
    our own actual peaks, r=0.13), so this variant is really "predict near
    our own sample's mean peak" rather than a real speed-scaling law --
    fine for storms like the ones we've seen, but it would NOT scale up
    for a genuinely faster/bigger future storm outside this sample's range.
    Kept here for comparison; intercept_shift_candidates() below is the
    safer variant that was actually promoted.
    """
    has_candidate = np.zeros(len(bins), dtype=bool)
    predicted_vmax = np.full(len(bins), np.nan)

    pairs = _event_pairs(bins, events, arrivals)
    if len(pairs) < 4:
        return has_candidate, predicted_vmax   # too few to refit; no candidates

    pairs_arr = np.array([(s, p) for s, p, _ in pairs])
    speeds, peaks = pairs_arr[:, 0], pairs_arr[:, 1]
    ev_idx = np.array([e for _, _, e in pairs])

    def refit_excluding(event_i):
        mask = ev_idx != event_i
        if mask.sum() < 3:
            return None
        A = np.vstack([np.ones(mask.sum()), speeds[mask]]).T
        coef, *_ = np.linalg.lstsq(A, peaks[mask], rcond=None)
        return coef

    # per-bin: find which event (if any) this bin belongs to, so the SAME
    # per-event LOO exclusion used for the classifier/flat-shift fold also
    # applies to the magnitude refit -- a bin inside event E must never see
    # event E's own pair in its magnitude estimate either
    bin_event = _bin_event_map(bins, events)

    for i, b in enumerate(bins):
        t_unix = int(b) * BIN_S
        offset_h, speed, has = arrival_feature(t_unix, arrivals)
        if not has:
            continue
        has_candidate[i] = True
        coef = refit_excluding(bin_event[i]) if bin_event[i] >= 0 else refit_excluding(-999)
        if coef is None:
            predicted_vmax[i] = np.nan
            has_candidate[i] = False
            continue
        pred = coef[0] + coef[1] * speed
        predicted_vmax[i] = float(np.clip(pred, fc.MAGNITUDE_MIN_VMAX, fc.MAGNITUDE_MAX_VMAX))
    return has_candidate, predicted_vmax


def intercept_shift_candidates(bins, events, arrivals):
    """Safer alternative to recalibrated_candidates(): keeps the original
    external formula's slope (a real, physically-motivated speed
    dependence, fit on an independent Richardson/Cane sample) and only
    LOO-refits an additive offset from our own residuals. This fixes the
    same +78 km/s systematic bias without discarding the ability to scale
    up for a future CME faster than anything in our own 43-event sample.
    """
    has_candidate = np.zeros(len(bins), dtype=bool)
    predicted_vmax = np.full(len(bins), np.nan)

    pairs = _event_pairs(bins, events, arrivals)
    if len(pairs) < 4:
        return has_candidate, predicted_vmax

    pairs_arr = np.array([(s, p) for s, p, _ in pairs])
    speeds, peaks = pairs_arr[:, 0], pairs_arr[:, 1]
    ev_idx = np.array([e for _, _, e in pairs])
    resid = peaks - np.array([estimate_earth_vmax(s) for s in speeds])

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
        pred = estimate_earth_vmax(speed) + shift
        predicted_vmax[i] = float(np.clip(pred, fc.MAGNITUDE_MIN_VMAX, fc.MAGNITUDE_MAX_VMAX))
    return has_candidate, predicted_vmax


def corrected_series_with(has_candidate, predicted_vmax, bins, X_all, obs_all,
                           blend_all, disturbed_mask_all, events):
    """Same fold/shift/gate logic as fc.loo_corrected_series, but taking a
    pre-computed (has_candidate, predicted_vmax) instead of calling
    fc._cme_candidates() internally -- lets us swap in the recalibrated
    version while keeping everything else identical."""
    ew_gate = np.where(fc._ew_rising_mask(bins, blend_all), 1.0, fc.EW_GATE_FACTOR)
    full_pred = blend_all.copy()
    held_out_mask = np.zeros(len(bins), dtype=bool)
    for start_b, end_b, peak in events:
        test_mask = (bins >= start_b) & (bins <= end_b)
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        mean_c, std_c, w, b = logistic_fit(
            X_all[train_mask], disturbed_mask_all[train_mask].astype(float), fc.LOGISTIC_ALPHA)
        flat_shift = float((obs_all[train_mask] - blend_all[train_mask])[disturbed_mask_all[train_mask]].mean())
        proba = predict_proba(X_all[test_mask], mean_c, std_c, w, b)
        shift = np.where(has_candidate[test_mask],
                          predicted_vmax[test_mask] - blend_all[test_mask],
                          flat_shift) * ew_gate[test_mask]
        full_pred[test_mask] = blend_all[test_mask] + (proba ** fc.POWER) * shift
        held_out_mask |= test_mask

    mean_c, std_c, w, b = logistic_fit(X_all, disturbed_mask_all.astype(float), fc.LOGISTIC_ALPHA)
    flat_shift_all = float((obs_all - blend_all)[disturbed_mask_all].mean())
    proba_all = predict_proba(X_all, mean_c, std_c, w, b)
    non_event_mask = ~held_out_mask
    shift_all = np.where(has_candidate[non_event_mask],
                          predicted_vmax[non_event_mask] - blend_all[non_event_mask],
                          flat_shift_all) * ew_gate[non_event_mask]
    full_pred[non_event_mask] = (blend_all[non_event_mask]
                                  + (proba_all[non_event_mask] ** fc.POWER) * shift_all)
    return full_pred


def per_event_compare(bins, obs_all, pred_a, pred_b, events, label_a, label_b):
    wins_a, wins_b, ties = 0, 0, 0
    worst_b_over_a = 0.0
    for start_b, end_b, peak in events:
        mask = (bins >= start_b) & (bins <= end_b)
        if not mask.any():
            continue
        rmse_a = np.sqrt(np.mean((obs_all[mask] - pred_a[mask]) ** 2))
        rmse_b = np.sqrt(np.mean((obs_all[mask] - pred_b[mask]) ** 2))
        if rmse_b < rmse_a - 1e-6:
            wins_b += 1
        elif rmse_a < rmse_b - 1e-6:
            wins_a += 1
        else:
            ties += 1
        worst_b_over_a = max(worst_b_over_a, rmse_b - rmse_a)
    print(f"  per-event: {label_a} wins {wins_a}, {label_b} wins {wins_b}, "
          f"ties {ties}  (worst {label_b}-over-{label_a} degradation: "
          f"+{worst_b_over_a:.1f} RMSE)")


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

    donki_events = load_events("donki_cme_2026.json")
    donki_fast = [e for e in donki_events if e["speed"] >= fc.CME_MIN_SPEED_KMS]
    donki_filtered = filter_earth_directed(donki_fast, max_cmd=fc.EARTH_DIRECTED_MAX_CMD,
                                            include_unknown=True)
    arrivals = build_calibrated_index(donki_filtered)

    print("Building CURRENT locked pipeline (external magnitude calibration)...")
    pred_locked, _, _, _ = fc.loo_corrected_series(
        bins, X_all, obs_all, blend_all, disturbed_mask_all, events)

    print("Building RECALIBRATED pipeline (full LOO refit, flat slope)...")
    has_c, pred_vmax = recalibrated_candidates(bins, events, arrivals)
    pred_recal = corrected_series_with(has_c, pred_vmax, bins, X_all, obs_all,
                                        blend_all, disturbed_mask_all, events)

    print("Building INTERCEPT-SHIFT pipeline (keeps old slope, LOO offset)...")
    has_c2, pred_vmax2 = intercept_shift_candidates(bins, events, arrivals)
    pred_shift = corrected_series_with(has_c2, pred_vmax2, bins, X_all, obs_all,
                                        blend_all, disturbed_mask_all, events)

    print()
    for name, pred in (("locked (external calib)", pred_locked),
                       ("full refit (flat slope)", pred_recal),
                       ("intercept-shift (kept slope)", pred_shift)):
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
    per_event_compare(bins, obs_all, pred_locked, pred_recal, events,
                       "locked", "full-refit")
    per_event_compare(bins, obs_all, pred_locked, pred_shift, events,
                       "locked", "intercept-shift")


if __name__ == "__main__":
    main()
