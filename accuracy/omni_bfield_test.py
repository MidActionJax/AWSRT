#!/usr/bin/env python3
"""Item #6 of the "11 things" (accuracy_improvement_tracking.txt,
2026-09-17 LOG): attack the small-N problem using real, multi-year OMNI
historical B/Bz/density/speed data (fetch_omni_history.py) instead of
this project's own ~10-15 independent events (Feb-Sept 2026 only).

WHY THIS CAN WORK AT ALL WITHOUT AWSoM-R: the disturbance classifier's
own reconciliation (2026-09-12 LOG, "CLASSIFIER RECONCILIATION") found
that bfield-only features (obs_B, obs_Bz, obs_Bz_min3h/24h, obs_n,
obs_B_max24h, obs_n_max24h) perform AT LEAST AS WELL as the locked
default's fuller feature set that also includes AWSoM-R model state
(model_v, trend, volatility, disagreement) -- meaning a classifier that
never looks at the model at all is already close to this project's own
best. OMNI's real historical record has no AWSoM-R run to pair with (the
real-time model only exists for 2026), but it has real, measured
B/Bz/density/speed for decades -- enough to build a bfield-ONLY
classifier on a MUCH larger, completely independent sample (2021-2023,
zero overlap with the 2026 test window), then check two real questions:

  1. Does the same B-field-predicts-disturbance pattern this project
     found in its own small sample hold up on far more independent
     historical events (a genuine replication check, not just more
     data for its own sake)?
  2. TRANSFER TEST: does a classifier trained on 2021-2023 OMNI history
     alone generalize to this project's own 2026 real-time record, and
     if so, does COMBINING both samples beat training on 2026 alone --
     i.e. does more historical data actually make the classifier this
     project already uses more robust?

    python omni_bfield_test.py
"""
from __future__ import annotations

import argparse

import numpy as np

from eval_harness import BIN_S, DISTURBED_KMS, fetch as fetch_live, find_events, to_bin_map
import phase2_model as p2

BFIELD_FEATURES = p2.BFIELD_FEATURES  # ("obs_B","obs_Bz","obs_Bz_min3h","obs_Bz_min24h","obs_n","obs_B_max24h","obs_n_max24h")


def bin_omni(path):
    """1-minute OMNI arrays -> 10-minute bin maps {bin: value}, mean per bin
    (consistent with climatology.py's own OMNI binning convention)."""
    d = np.load(path)
    t = d["t_unix"]
    order = np.argsort(t)
    t = t[order]
    bins_idx = (t // BIN_S).astype(np.int64)
    out = {}
    for key in ("v_kms", "b_nT", "bz_nT", "n_cc"):
        v = d[key][order]
        acc = {}
        for bi, vi in zip(bins_idx, v):
            if np.isnan(vi):
                continue
            acc.setdefault(int(bi), []).append(float(vi))
        out[key] = {k: float(np.mean(vs)) for k, vs in acc.items()}
    return out["v_kms"], out["b_nT"], out["bz_nT"], out["n_cc"]


def build_bfield_dataset(obs_bins, b_map, bz_map, n_map):
    """One row per bin with full B-field feature coverage + the disturbed
    label -- the bfield-only analog of phase2_model.build_dataset(), with
    no AWSoM-R model dependency at all (OMNI history has no paired model
    run to use). Mirrors build_dataset's own window/coverage logic
    exactly so features are computed identically to the locked pipeline's.
    """
    trend3 = round(3 * 3600 / BIN_S)
    day_bins = round(24 * 3600 / BIN_S)
    rows = []
    for b in sorted(obs_bins):
        ov = obs_bins[b]
        bz_window3 = [bz_map[bb] for bb in range(b - trend3, b + 1) if bb in bz_map]
        bz_window24 = [bz_map[bb] for bb in range(b - day_bins, b + 1) if bb in bz_map]
        b_window24 = [b_map[bb] for bb in range(b - day_bins, b + 1) if bb in b_map]
        n_window24 = [n_map[bb] for bb in range(b - day_bins, b + 1) if bb in n_map]
        if b not in b_map or b not in bz_map or b not in n_map \
                or not bz_window3 or not bz_window24 or not b_window24 or not n_window24:
            continue
        rows.append({
            "bin": b, "obs_v": ov,
            "obs_B": b_map[b], "obs_Bz": bz_map[b],
            "obs_Bz_min3h": min(bz_window3), "obs_Bz_min24h": min(bz_window24),
            "obs_n": n_map[b], "obs_B_max24h": max(b_window24), "obs_n_max24h": max(n_window24),
        })
    bins = np.array([r["bin"] for r in rows])
    out = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]} if rows else {}
    return bins, out


def event_windows_from_obs(obs_bins, threshold=600.0, pad_hours=24):
    pad_bins = round(pad_hours * 3600 / BIN_S)
    events = find_events(obs_bins, threshold=threshold)
    return [(s - pad_bins, e + pad_bins, peak) for s, e, peak in events]


def loo_auc(bins, X, y, events, alpha):
    """Same per-event LOO discipline as every classifier in this project."""
    proba = np.zeros(len(bins))
    held_out = np.zeros(len(bins), dtype=bool)
    for start_b, end_b, peak in events:
        test_mask = (bins >= start_b) & (bins <= end_b)
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        mean, std, w, b = p2.logistic_fit(X[train_mask], y[train_mask], alpha)
        proba[test_mask] = p2.predict_proba(X[test_mask], mean, std, w, b)
        held_out |= test_mask
    mean, std, w, b = p2.logistic_fit(X, y, alpha)
    ne = ~held_out
    proba[ne] = p2.predict_proba(X[ne], mean, std, w, b)
    return proba


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omni", default="omni_history_bfield.npz")
    ap.add_argument("--alpha", type=float, default=0.005,
                     help="matches this project's own locked bfield-only classifier alpha")
    args = ap.parse_args()

    print(f"Loading OMNI history from {args.omni}...")
    v_map, b_map, bz_map, n_map = bin_omni(args.omni)
    print(f"  {len(v_map):,} 10-minute bins with a real speed value")

    omni_bins, omni_data = build_bfield_dataset(v_map, b_map, bz_map, n_map)
    print(f"  {len(omni_bins):,} bins with full B-field feature coverage")
    omni_events = event_windows_from_obs(v_map, threshold=600.0)
    print(f"  {len(omni_events)} real disturbance events found (>=600 km/s, "
          f"+/-24h padded) in 2021-2023 alone -- vs this project's own "
          f"~10-15 in its 7.7-month 2026 record\n")

    X_omni = np.column_stack([omni_data[f] for f in BFIELD_FEATURES])
    y_omni = (omni_data["obs_v"] > DISTURBED_KMS).astype(float)

    print("=== QUESTION 1: does the B-field-predicts-disturbance pattern "
          "replicate on this much larger, independent sample? ===")
    proba_omni_own = loo_auc(omni_bins, X_omni, y_omni, omni_events, args.alpha)
    auc_omni_own = p2.auc(y_omni, proba_omni_own)
    print(f"  LOO AUC, trained AND tested on OMNI 2021-2023: {auc_omni_own:.3f}\n")

    print("=== QUESTION 2: TRANSFER TEST -- does OMNI-trained (2021-2023) "
          "generalize to this project's OWN 2026 real-time record, and "
          "does combining beat training on 2026 alone? ===")
    doc = fetch_live()
    obs = doc["obs"]
    obs_bins_2026 = to_bin_map(obs["t"], obs["v"])
    bfield_2026 = {"b": to_bin_map(obs["t"], obs["b"]), "bz": to_bin_map(obs["t"], obs["bz"]),
                   "n": to_bin_map(obs["t"], obs["n"])}
    bins_2026, data_2026 = build_bfield_dataset(obs_bins_2026, bfield_2026["b"],
                                                 bfield_2026["bz"], bfield_2026["n"])
    X_2026 = np.column_stack([data_2026[f] for f in BFIELD_FEATURES])
    y_2026 = (data_2026["obs_v"] > DISTURBED_KMS).astype(float)
    events_2026 = event_windows_from_obs(obs_bins_2026, threshold=600.0)
    print(f"  2026 record: {len(bins_2026):,} bins, {len(events_2026)} events")

    # (a) baseline: this project's own already-validated approach --
    # train/test on 2026 alone, LOO per event (reproduces the locked
    # classifier's own AUC when using bfield-only features)
    proba_2026_own = loo_auc(bins_2026, X_2026, y_2026, events_2026, args.alpha)
    auc_2026_own = p2.auc(y_2026, proba_2026_own)
    print(f"  (a) baseline -- train+test on 2026 alone (LOO): AUC={auc_2026_own:.3f}")

    # (b) pure transfer -- train ONLY on OMNI 2021-2023 (all of it, no
    # fold needed since 2026 is a fully disjoint held-out set by
    # construction), test on ALL of 2026
    mean, std, w, b = p2.logistic_fit(X_omni, y_omni, args.alpha)
    proba_transfer = p2.predict_proba(X_2026, mean, std, w, b)
    auc_transfer = p2.auc(y_2026, proba_transfer)
    print(f"  (b) pure transfer -- train on OMNI 2021-2023 ONLY, test on "
          f"ALL of 2026: AUC={auc_transfer:.3f}")

    # (c) combined -- train on OMNI 2021-2023 + 2026's OWN other events
    # (same per-event LOO fold on the 2026 side, OMNI always fully in
    # training since it's already disjoint), test on the held-out 2026 event
    proba_combined = np.zeros(len(bins_2026))
    held_out = np.zeros(len(bins_2026), dtype=bool)
    for start_b, end_b, peak in events_2026:
        test_mask = (bins_2026 >= start_b) & (bins_2026 <= end_b)
        if not test_mask.any():
            continue
        train_mask_2026 = ~test_mask
        X_train = np.vstack([X_omni, X_2026[train_mask_2026]])
        y_train = np.concatenate([y_omni, y_2026[train_mask_2026]])
        mean, std, w, b = p2.logistic_fit(X_train, y_train, args.alpha)
        proba_combined[test_mask] = p2.predict_proba(X_2026[test_mask], mean, std, w, b)
        held_out |= test_mask
    ne = ~held_out
    X_train_all = np.vstack([X_omni, X_2026[held_out] if held_out.any() else X_2026])
    mean, std, w, b = p2.logistic_fit(np.vstack([X_omni, X_2026]),
                                       np.concatenate([y_omni, y_2026]), args.alpha)
    proba_combined[ne] = p2.predict_proba(X_2026[ne], mean, std, w, b)
    auc_combined = p2.auc(y_2026, proba_combined)
    print(f"  (c) combined -- train on OMNI 2021-2023 + 2026's own other "
          f"events (LOO on 2026 side), test on 2026: AUC={auc_combined:.3f}")

    print(f"\nSummary: (a) 2026-only={auc_2026_own:.3f}  (b) OMNI-transfer-only="
          f"{auc_transfer:.3f}  (c) combined={auc_combined:.3f}")


if __name__ == "__main__":
    main()
