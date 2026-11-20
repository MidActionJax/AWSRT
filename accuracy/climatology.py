#!/usr/bin/env python3
"""OMNI disturbance climatology: what does a typical high-speed disturbance
actually look like, learned from real historical OMNI data (2023-2024,
omni_2023_2024.npz -- fetched directly from NASA/SPDF the same way
fetch_insitu.py does, see that fetch's own log), completely independent of
AWSoM-R's own short 7.7-month real-time record and with zero date overlap
with the 2026 test period.

Why: Phase 2's regression approach is bottlenecked by only ~10-15 real
disturbance events existing in AWSoM-R's own history to learn from (see
accuracy_improvement_tracking.txt). OMNI's real measured record has many
more events over just these two years alone. This builds a superposed-
epoch composite (align every historical event to its onset, average the
aligned profiles) -- a standard technique in solar wind climatology, not
something invented here -- and tests whether that composite, triggered by
a real-time onset detector, predicts the 2026 held-out events better than
Phase 1b's blend or Phase 2's regression.

    python climatology.py
"""
from __future__ import annotations

import numpy as np

from eval_harness import BIN_S, fetch as fetch_live, find_events, to_bin_map
import phase2_model as p2

ONSET_HOURS_BEFORE = 24
PROFILE_HOURS_AFTER = 120


def bin_omni(path):
    d = np.load(path)
    t, v = d["t_unix"], d["v_kms"]
    bins = {}
    order = np.argsort(t)
    t, v = t[order], v[order]
    b = (t // BIN_S).astype(np.int64)
    for bi, vi in zip(b, v):
        # median-per-bin would be better than last-wins, but with 1-min data
        # collapsing to 10-min bins, a simple running mean is enough for a
        # climatology profile (not a precision product) -- accumulate then
        # average
        bins.setdefault(int(bi), []).append(float(vi))
    return {k: float(np.mean(vs)) for k, vs in bins.items()}


def build_composite(bins, events, onset_before_h=ONSET_HOURS_BEFORE,
                     profile_after_h=PROFILE_HOURS_AFTER):
    """Superposed-epoch mean profile: align each event to its onset bin
    (event start), stack [-onset_before_h, +profile_after_h] windows,
    average column-wise. Returns (offsets_h, mean_profile, n_events_used).
    """
    before_bins = round(onset_before_h * 3600 / BIN_S)
    after_bins = round(profile_after_h * 3600 / BIN_S)
    width = before_bins + after_bins + 1
    stack = []
    for start_b, end_b, peak in events:
        row = np.full(width, np.nan)
        for i, bi in enumerate(range(start_b - before_bins, start_b + after_bins + 1)):
            if bi in bins:
                row[i] = bins[bi]
        if np.isnan(row).mean() > 0.3:   # too much missing data, skip this event
            continue
        stack.append(row)
    if not stack:
        return None, None, 0
    stack = np.array(stack)
    mean_profile = np.nanmean(stack, axis=0)
    offsets_h = (np.arange(width) - before_bins) * BIN_S / 3600.0
    return offsets_h, mean_profile, len(stack)


def onset_score(model_bins, obs_bins, persist_bins, b, trend_window_bins=18):
    """Cheap real-time onset detector, using only what's available at bin b:
    how much has the model+persistence disagreement grown over the last
    3h, as a proxy for "something is starting to diverge from the quiet
    baseline". Reuses the same disagreement idea as Phase 2's regression
    features, just used here as a trigger rather than a regression input.
    """
    mv = model_bins.get(b)
    pv = persist_bins.get(b)
    mv_prior = model_bins.get(b - trend_window_bins)
    pv_prior = persist_bins.get(b - trend_window_bins)
    if None in (mv, pv, mv_prior, pv_prior):
        return None
    now = abs(mv - pv)
    prior = abs(mv_prior - pv_prior)
    return now - prior   # positive = disagreement growing


DURATION_BUCKETS = (("short", 0, 6), ("medium", 6, 24), ("long", 24, 1e9))


def bucket_events(events, buckets=DURATION_BUCKETS):
    """Split events into duration buckets (hours). Stratifying before
    averaging is the fix for the naive composite's failure (2026-09-11):
    a 20-min spike and a 74h extended event share no common shape, so
    averaging them together washes out both. Bucketing by duration is
    legitimate for BUILDING historical composites (duration is already
    known for past events) but NOT causally available for picking a
    bucket in real time for a live, still-unfolding event -- that's a
    separate problem, tackled only after checking whether stratifying
    even fixes the shape in the first place.
    """
    out = {name: [] for name, _, _ in buckets}
    for start_b, end_b, peak in events:
        dur_h = (end_b - start_b) * BIN_S / 3600
        for name, lo, hi in buckets:
            if lo <= dur_h < hi:
                out[name].append((start_b, end_b, peak))
                break
    return out


def main():
    print("Loading 2023-2024 OMNI history and building the composite...")
    hist_bins = bin_omni("omni_2023_2024.npz")
    hist_events = find_events(hist_bins, threshold=600.0)
    print(f"  {len(hist_events)} historical events (600+ km/s) found in "
          f"2023-2024 to build the composite from")

    offsets_h, profile, n_used = build_composite(hist_bins, hist_events)
    if profile is None:
        raise SystemExit("no usable historical events -- check omni_2023_2024.npz")
    print(f"  composite built from {n_used} usable events (some dropped for "
          f"missing data coverage)")
    peak_idx = int(np.nanargmax(profile))
    print(f"  composite peak: {profile[peak_idx]:.0f} km/s at "
          f"offset {offsets_h[peak_idx]:+.1f}h from onset\n")

    print("Loading 2026 test data (the same real dataset used throughout "
          "this project)...")
    doc = fetch_live()
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])
    rot_bins = round(27 * 86400 / BIN_S)
    persist_bins = {b: obs_bins[b - rot_bins] for b in obs_bins
                    if (b - rot_bins) in obs_bins}
    events_2026 = p2.event_windows(obs_bins, threshold=600.0)
    print(f"  {len(events_2026)} 2026 event windows to test against\n")

    # For each 2026 event, at its true onset bin, apply the historical
    # composite profile (scaled to the local baseline level at trigger time,
    # since the composite is in absolute km/s from ITS OWN events' baselines)
    # and score it against the blend and Phase 2 on the exact same rows.
    bins_arr, data = p2.build_dataset(model_bins, obs_bins)
    blend_by_bin = dict(zip(bins_arr, data["blend_v"]))

    results_obs, results_blend, results_clim = [], [], []
    before_bins = round(ONSET_HOURS_BEFORE * 3600 / BIN_S)
    for start_b, end_b, peak in events_2026:
        if start_b not in obs_bins:
            continue
        baseline = obs_bins.get(start_b - before_bins)
        if baseline is None:
            continue
        for i, bi in enumerate(range(start_b - before_bins,
                                      start_b + round(PROFILE_HOURS_AFTER * 3600 / BIN_S) + 1)):
            if bi not in obs_bins or bi not in blend_by_bin:
                continue
            # scale the composite's shape (delta from ITS OWN pre-onset
            # baseline) onto this event's own pre-onset baseline, rather
            # than using the composite's absolute km/s directly
            hist_baseline = profile[before_bins]
            clim_pred = baseline + (profile[i] - hist_baseline)
            results_obs.append(obs_bins[bi])
            results_blend.append(blend_by_bin[bi])
            results_clim.append(clim_pred)

    results_obs = np.array(results_obs)
    results_blend = np.array(results_blend)
    results_clim = np.array(results_clim)
    print(f"Scored on {len(results_obs)} rows across all 2026 event windows "
          f"(-{ONSET_HOURS_BEFORE}h to +{PROFILE_HOURS_AFTER}h around each onset):\n")
    s_blend = p2.stats(results_obs, results_blend)
    s_clim = p2.stats(results_obs, results_clim)
    print(f"  Phase 1b blend    n={s_blend['n']:5d}  rmse={s_blend['rmse']:6.1f}  "
          f"mae={s_blend['mae']:6.1f}  bias={s_blend['bias']:7.1f}  corr={s_blend['corr']:6.3f}")
    print(f"  Climatology       n={s_clim['n']:5d}  rmse={s_clim['rmse']:6.1f}  "
          f"mae={s_clim['mae']:6.1f}  bias={s_clim['bias']:7.1f}  corr={s_clim['corr']:6.3f}")


if __name__ == "__main__":
    main()
