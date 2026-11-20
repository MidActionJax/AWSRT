#!/usr/bin/env python3
"""Phase 2: regime-aware correction on top of the Phase 1b physics+persistence
blend, validated leave-one-event-out (see AWSRT/accuracy_improvement_tracking.txt
for why: only ~10-15 genuinely distinct major disturbances exist in the real
record, so a random time-split would leak each event across train/test and
overstate how well this generalizes to the next real event).

    python phase2_model.py
    python phase2_model.py --json path/to/earth.json
    python phase2_model.py --alpha 5.0

Deliberately a small linear ridge regression, not a deep model: with only
~10-15 independent disturbance events to learn from, a high-capacity model
would memorize those specific events rather than learn generalizable
behavior. This predicts a CORRECTION to the Phase 1b blend (residual
learning), not a from-scratch speed prediction, so quiet-period behavior
(already good) is protected by construction: the correction is small
wherever the input features that trigger it (trend, volatility, model-vs-
persistence disagreement) are themselves small, which they are in quiet
conditions.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize

from eval_harness import (BIN_S, DISTURBED_KMS, fetch, find_events,
                           to_bin_map)
from donki_features import (activity_feature, arrival_feature,
                             build_calibrated_index, build_index, load_events)

FEATURES = ("model_v", "trend_3h", "trend_6h", "volatility_24h", "disagreement")
DONKI_ARRIVAL_FEATURES = ("cme_arrival_offset_h", "cme_speed", "cme_has_candidate")
DONKI_ACTIVITY_FEATURES = ("cme_count_72h", "cme_maxspeed_72h")
BFIELD_FEATURES = ("obs_B", "obs_Bz", "obs_Bz_min3h", "obs_Bz_min24h", "obs_n",
                    "obs_B_max24h", "obs_n_max24h")

# ADAPTIVE BLEND WEIGHTING (2026-09-14, Tier 3 item 8 on the master
# next-steps map -- untried until Jaxon spotted a real 90-hour streak on
# the live dashboard where the static 50/50 blend badly overshot while
# raw AWSoM-R alone tracked fine, and the correction pipeline correctly
# applied ZERO shift there (no disturbance, early-warning gate at 0) --
# meaning the failure was in the blend ITSELF, not anything downstream.
# Mechanism: 27-day recurrence persistence assumes this solar rotation
# repeats the last one; when a stream structure has evolved or dissipated
# since then, persistence is stale and badly wrong, and blending it 50/50
# with a raw model that's tracking fine just drags the blend down with it.
# Fix: weight the blend each bin by which of {raw model, persistence} has
# been more accurate over a short CAUSAL trailing window (only genuinely
# past obs, no leakage) -- w_raw = trailing_persist_error /
# (trailing_raw_error + trailing_persist_error). Swept window length
# 0.5h-24h: results were on an extremely flat, stable plateau from
# 0.5h-4h (RMSE 62.8-63.1, corr 0.763-0.765 for the full downstream
# pipeline) -- not fit to a specific window, genuinely insensitive to the
# exact choice in that range. Picked 3h to match the existing trend_3h
# convention already used elsewhere. Checked per-event (35 held-out
# windows): wins on 34, loses on 1 (by a trivial +4.8 RMSE), best wins
# -48 to -64 RMSE. THE SINGLE BIGGEST IMPROVEMENT FOUND IN THIS PROJECT
# -- see accuracy_improvement_tracking.txt LOG, "ADAPTIVE BLEND WEIGHTING".
ADAPTIVE_BLEND_WINDOW_BINS = 18   # 3 hours at 10-min cadence
ADAPTIVE_BLEND_MIN_SAMPLES = 6    # need at least this much real trailing
                                   # history before trusting the adaptive
                                   # weight; falls back to 0.5/0.5 otherwise


def _adaptive_blend_weight(model_v, persist_v, obs_v, window=ADAPTIVE_BLEND_WINDOW_BINS):
    """Per-row weight on model_v (1-weight on persist_v), from which of the
    two has had lower trailing absolute error over the previous `window`
    rows (by row position, excluding the current row -- fully causal,
    only ever looks at genuinely past obs). Falls back to 0.5 (the
    original static blend) wherever there isn't enough real trailing
    history yet.
    """
    raw_err = np.abs(model_v - obs_v)
    persist_err = np.abs(persist_v - obs_v)
    cs_raw = np.concatenate([[0.0], np.cumsum(raw_err)])
    cs_persist = np.concatenate([[0.0], np.cumsum(persist_err)])
    n = len(model_v)
    w_raw = np.full(n, 0.5)
    min_samples = max(ADAPTIVE_BLEND_MIN_SAMPLES, window // 3)
    for i in range(n):
        lo = max(0, i - window)
        if i - lo < min_samples:
            continue
        trail_raw = (cs_raw[i] - cs_raw[lo]) / (i - lo)
        trail_persist = (cs_persist[i] - cs_persist[lo]) / (i - lo)
        w_raw[i] = trail_persist / (trail_raw + trail_persist + 1e-9)
    return w_raw


def build_dataset(model_bins, obs_bins, donki_arrivals=None, donki_events=None,
                   donki_window_hours=72, bfield_bins=None):
    """One row per bin present in obs, model, AND 27-day-prior obs
    (persistence needs that; a row missing any of these can't be scored
    or corrected, so it's dropped rather than imputed).

    Returns bins (sorted int array) and a dict of equal-length float arrays:
    model_v, persist_v, blend_v, obs_v, target (= obs_v - blend_v), plus
    every name in FEATURES (and DONKI_FEATURES if donki_arrivals is given).

    bfield_bins, if given: {"b": bin_map, "bz": bin_map, "n": bin_map} of
    REAL-TIME OBSERVED |B|, Bz, and density (obs's own b/bz/n arrays via
    to_bin_map -- legitimately causal, same status as obs velocity itself,
    which the dashboard already shows live). Adds obs_B, obs_Bz,
    obs_Bz_min3h (how southward Bz has dipped in the trailing 3h -- the
    dominant driver, see accuracy_improvement_tracking.txt LOG,
    2026-09-12 "REAL-TIME B-FIELD FEATURES"), and obs_n.
    """
    rot_bins = round(27 * 86400 / BIN_S)
    trend3 = round(3 * 3600 / BIN_S)
    trend6 = round(6 * 3600 / BIN_S)
    vol_window = round(24 * 3600 / BIN_S)

    candidate_bins = sorted(set(model_bins) & set(obs_bins))
    rows = []
    for b in candidate_bins:
        persist_v = obs_bins.get(b - rot_bins)
        if persist_v is None:
            continue
        mv = model_bins[b]
        mv3 = model_bins.get(b - trend3)
        mv6 = model_bins.get(b - trend6)
        # trailing volatility: std of model_v over the prior 24h, using
        # whatever bins in that window actually exist (real data has gaps)
        window_vals = [model_bins[bb] for bb in range(b - vol_window, b + 1)
                       if bb in model_bins]
        if len(window_vals) < vol_window // 4:   # need at least ~6h of real coverage
            continue
        if mv3 is None or mv6 is None:
            continue
        vol24 = float(np.std(window_vals))
        blend_v = 0.5 * mv + 0.5 * persist_v
        ov = obs_bins[b]
        row = {
            "bin": b, "model_v": mv, "persist_v": persist_v, "blend_v": blend_v,
            "obs_v": ov, "target": ov - blend_v,
            "trend_3h": mv - mv3, "trend_6h": mv - mv6,
            "volatility_24h": vol24, "disagreement": mv - persist_v,
        }
        if donki_arrivals is not None:
            offset_h, speed, has = arrival_feature(b * BIN_S, donki_arrivals)
            row["cme_arrival_offset_h"] = offset_h
            row["cme_speed"] = speed
            row["cme_has_candidate"] = has
        if donki_events is not None:
            count, max_speed = activity_feature(b * BIN_S, donki_events,
                                                 window_hours=donki_window_hours)
            row["cme_count_72h"] = count
            row["cme_maxspeed_72h"] = max_speed
        if bfield_bins is not None:
            b_map, bz_map, n_map = bfield_bins["b"], bfield_bins["bz"], bfield_bins["n"]
            day_bins = round(24 * 3600 / BIN_S)
            bz_window3 = [bz_map[bb] for bb in range(b - trend3, b + 1) if bb in bz_map]
            bz_window24 = [bz_map[bb] for bb in range(b - day_bins, b + 1) if bb in bz_map]
            b_window24 = [b_map[bb] for bb in range(b - day_bins, b + 1) if bb in b_map]
            n_window24 = [n_map[bb] for bb in range(b - day_bins, b + 1) if bb in n_map]
            if b not in b_map or b not in bz_map or b not in n_map \
                    or not bz_window3 or not bz_window24 or not b_window24 or not n_window24:
                continue  # no usable B-field data for this bin -- drop it rather than impute
            row["obs_B"] = b_map[b]
            row["obs_Bz"] = bz_map[b]
            row["obs_Bz_min3h"] = min(bz_window3)
            row["obs_Bz_min24h"] = min(bz_window24)
            row["obs_n"] = n_map[b]
            row["obs_B_max24h"] = max(b_window24)
            row["obs_n_max24h"] = max(n_window24)
        rows.append(row)
    bins = np.array([r["bin"] for r in rows])
    out = {k: np.array([r[k] for r in rows], dtype=float)
           for k in rows[0] if k != "bin"} if rows else {}
    if rows:
        # adaptive blend weighting (see ADAPTIVE_BLEND_WINDOW_BINS above) --
        # overwrites the static-50/50 blend_v/target computed per-row above.
        # Rows are already in bin (time) order since candidate_bins is
        # sorted, so trailing windows here are genuinely trailing in time.
        w_raw = _adaptive_blend_weight(out["model_v"], out["persist_v"], out["obs_v"])
        out["blend_v"] = w_raw * out["model_v"] + (1 - w_raw) * out["persist_v"]
        out["target"] = out["obs_v"] - out["blend_v"]
    return bins, out


def ridge_fit(X, y, alpha):
    """Closed-form ridge on standardized features (+ intercept, unregularized).

    Returns (mean, std, coef, intercept) so the same standardization is
    reproducible at predict time on new (e.g. held-out) rows.
    """
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    n, p = Xs.shape
    A = np.hstack([Xs, np.ones((n, 1))])
    reg = np.eye(p + 1) * alpha
    reg[-1, -1] = 0.0   # don't regularize the intercept
    coef = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return mean, std, coef[:-1], coef[-1]


def ridge_predict(X, mean, std, coef, intercept):
    Xs = (X - mean) / std
    return Xs @ coef + intercept


def logistic_fit(X, y, alpha=1.0):
    """Standard L2-regularized logistic regression via L-BFGS-B.

    y is 1/0 (disturbed/quiet). Returns (mean, std, w, b) in the same
    shape as ridge_fit/corr_fit above, so predict_proba below can share
    their standardization convention.

    MOVED HERE 2026-09-17 from uncertainty_model.py (which still
    re-exports both names for existing callers) so early_warning.py can
    use it directly without a circular import -- uncertainty_model.py
    already does `import early_warning as ew`, so early_warning.py
    importing FROM uncertainty_model would be circular. phase2_model.py
    is the shared base both already import from, so it's the correct
    home for this, not either script's own private copy.
    """
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    n, p = Xs.shape

    def neg_log_likelihood(params):
        w, b = params[:p], params[p]
        z = Xs @ w + b
        log_lik = -np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))
        return -log_lik + alpha * float(w @ w)

    x0 = np.zeros(p + 1)
    res = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
    w, b = res.x[:p], res.x[p]
    return mean, std, w, b


def predict_proba(X, mean, std, w, b):
    Xs = (X - mean) / std
    z = Xs @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def corr_fit(X, obs, blend, alpha):
    """Fit the same linear form (blend + Xs@w + b) as ridge_fit, but by
    directly maximizing Pearson correlation between the corrected
    prediction and obs on the training rows, instead of minimizing
    squared error.

    Motivation (2026-09-11 finding): every RMSE-minimizing correction
    tried this session made held-out disturbed-event CORRELATION worse
    than the uncorrected blend, consistently, across many feature sets.
    RMSE and correlation are different objectives; a correction that pulls
    extreme values toward a safer mean reduces RMSE while flattening the
    signal. This fits what we actually want to be good at instead of
    hoping it comes along for free.

    L2 penalty on w (not on the correlation term itself) keeps this from
    chasing a training-set correlation spike with wild coefficients,
    which the small-N event count makes a real risk.
    """
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    n, p = Xs.shape

    def neg_objective(params):
        w, b = params[:p], params[p]
        pred = blend + Xs @ w + b
        if pred.std() < 1e-9 or obs.std() < 1e-9:
            return 1.0 + alpha * float(w @ w)
        corr = np.corrcoef(pred, obs)[0, 1]
        return -corr + alpha * float(w @ w)

    x0 = np.zeros(p + 1)
    res = minimize(neg_objective, x0, method="L-BFGS-B")
    w, b = res.x[:p], res.x[p]
    return mean, std, w, b


def feature_matrix(data):
    return np.column_stack([data[f] for f in FEATURES])


def live_features(model_bins, obs_bins, bfield_bins=None, donki_events=None,
                   donki_window_hours=72):
    """Compute every possible live (as-of-right-now) feature value for the
    most recent model bin, matching build_dataset's per-row definitions
    exactly. Added 2026-09-12 to de-duplicate what used to be three
    independently copy-pasted implementations of this same logic
    (final_correction.py, early_warning.py, uncertainty_model.py's
    apply_to_live functions) -- a real maintainability/correctness risk,
    since an edit to one copy wouldn't automatically reach the others.

    Returns (b0, row): b0 is the latest bin present in model_bins; row is
    a dict with every feature this function knows how to compute (a
    caller uses live_feature_vector() to pick out its own FEATURES's exact
    subset/order), plus "_persist_v" and "_blend_v" for callers that need
    the raw blend components directly instead of via FEATURES.
    """
    rot_bins = round(27 * 86400 / BIN_S)
    trend3 = round(3 * 3600 / BIN_S)
    trend6 = round(6 * 3600 / BIN_S)
    vol_window = round(24 * 3600 / BIN_S)
    day_bins = round(24 * 3600 / BIN_S)

    b0 = max(model_bins)
    mv = model_bins[b0]
    persist_v = obs_bins.get(b0 - rot_bins)
    mv3 = model_bins.get(b0 - trend3)
    mv6 = model_bins.get(b0 - trend6)
    window_vals = [model_bins[bb] for bb in range(b0 - vol_window, b0 + 1) if bb in model_bins]
    vol24 = float(np.std(window_vals))

    # adaptive blend weight (see ADAPTIVE_BLEND_WINDOW_BINS above) -- same
    # trailing-error mechanism as build_dataset(), computed directly from
    # the bin maps since there's no row array to reuse at live-serving time
    raw_errs, persist_errs = [], []
    for bb in range(b0 - ADAPTIVE_BLEND_WINDOW_BINS, b0):
        if bb in model_bins and bb in obs_bins:
            raw_errs.append(abs(model_bins[bb] - obs_bins[bb]))
        pb = bb - rot_bins
        if pb in obs_bins and bb in obs_bins:
            persist_errs.append(abs(obs_bins[pb] - obs_bins[bb]))
    min_samples = max(ADAPTIVE_BLEND_MIN_SAMPLES, ADAPTIVE_BLEND_WINDOW_BINS // 3)
    if len(raw_errs) >= min_samples and len(persist_errs) >= min_samples:
        trail_raw = sum(raw_errs) / len(raw_errs)
        trail_persist = sum(persist_errs) / len(persist_errs)
        w_raw = trail_persist / (trail_raw + trail_persist + 1e-9)
    else:
        w_raw = 0.5
    blend_v = w_raw * mv + (1 - w_raw) * persist_v

    row = {"model_v": mv, "trend_3h": mv - mv3, "trend_6h": mv - mv6,
           "volatility_24h": vol24, "disagreement": mv - persist_v,
           "_persist_v": persist_v, "_blend_v": blend_v}

    if donki_events is not None:
        count, max_speed = activity_feature(b0 * BIN_S, donki_events, window_hours=donki_window_hours)
        row["cme_count_72h"] = count
        row["cme_maxspeed_72h"] = max_speed

    if bfield_bins is not None:
        b_map, bz_map, n_map = bfield_bins["b"], bfield_bins["bz"], bfield_bins["n"]

        def nearest(bin_map, bi, tolerance=3):
            for d in range(tolerance + 1):
                if bi + d in bin_map:
                    return bin_map[bi + d]
                if bi - d in bin_map:
                    return bin_map[bi - d]
            return None

        bz_window3 = [bz_map[bb] for bb in range(b0 - trend3, b0 + 4) if bb in bz_map]
        bz_window24 = [bz_map[bb] for bb in range(b0 - day_bins, b0 + 4) if bb in bz_map]
        b_window24 = [b_map[bb] for bb in range(b0 - day_bins, b0 + 4) if bb in b_map]
        n_window24 = [n_map[bb] for bb in range(b0 - day_bins, b0 + 4) if bb in n_map]

        # "current" reading: prefer something within nearest()'s +/-30min
        # tolerance; if the feed has a real gap right at the live edge (a
        # satellite dropout, a processing delay -- a real scenario, checked
        # 2026-09-12, not hypothetical), fall back to the most recent value
        # still inside the wider 24h window (stale but real) instead of
        # silently returning None. An unhandled None here would object-dtype
        # the whole feature vector and crash confusingly deep inside
        # predict_proba's matrix arithmetic, exactly when a degraded feed
        # most needs a clear signal instead of a cryptic failure. Only a
        # feed that's EMPTY for the full trailing 24h is allowed to fail
        # loudly (no silent imputation past that point).
        obs_B = nearest(b_map, b0)
        if obs_B is None:
            if not b_window24:
                raise ValueError(f"no |B| data anywhere in the trailing 24h "
                                  f"ending at bin {b0} -- B-field feed appears fully down")
            obs_B = b_window24[-1]
        obs_Bz = nearest(bz_map, b0)
        if obs_Bz is None:
            if not bz_window24:
                raise ValueError(f"no Bz data anywhere in the trailing 24h "
                                  f"ending at bin {b0} -- B-field feed appears fully down")
            obs_Bz = bz_window24[-1]
        obs_n = nearest(n_map, b0)
        if obs_n is None:
            if not n_window24:
                raise ValueError(f"no density data anywhere in the trailing 24h "
                                  f"ending at bin {b0} -- B-field feed appears fully down")
            obs_n = n_window24[-1]

        row["obs_B"] = obs_B
        row["obs_Bz"] = obs_Bz
        row["obs_Bz_min3h"] = min(bz_window3) if bz_window3 else obs_Bz
        row["obs_Bz_min24h"] = min(bz_window24) if bz_window24 else obs_Bz
        row["obs_n"] = obs_n
        row["obs_B_max24h"] = max(b_window24) if b_window24 else obs_B
        row["obs_n_max24h"] = max(n_window24) if n_window24 else obs_n

    return b0, row


def live_feature_vector(row, features=None):
    """row from live_features() -> a (1, len(features)) array in the given
    feature order (module-level FEATURES by default), the same convention
    feature_matrix() uses for historical rows."""
    names = features if features is not None else FEATURES
    return np.array([[row[name] for name in names]])


def event_windows(obs_bins, threshold=600.0, pad_hours=24):
    """Major events (see eval_harness.find_events), each padded by
    pad_hours on both sides so precursor/decay rows physically tied to the
    event don't leak into a training fold that's supposed to exclude it."""
    pad_bins = round(pad_hours * 3600 / BIN_S)
    events = find_events(obs_bins, threshold=threshold)
    return [(s - pad_bins, e + pad_bins, peak) for s, e, peak in events]


def auc(y, p):
    """Rank-based AUC (Mann-Whitney U), no sklearn dependency -- same
    formula already used inline in uncertainty_model.py's discrimination
    check; added here as a shared helper 2026-09-17 so early_warning.py's
    new classification-reframed forecast (see accuracy_improvement_
    tracking.txt LOG) can use the identical metric without a third
    inline copy. y is 1/0, p is a predicted probability/score.
    """
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]))


def stats(obs, pred):
    err = pred - obs
    n = len(obs)
    bias = err.mean()
    rmse = np.sqrt((err ** 2).mean())
    mae = np.abs(err).mean()
    corr = (np.corrcoef(obs, pred)[0, 1] if n > 1 and obs.std() > 0
            and pred.std() > 0 else float("nan"))
    return {"n": n, "bias": round(float(bias), 1), "rmse": round(float(rmse), 1),
            "mae": round(float(mae), 1), "corr": round(float(corr), 3)}


def report(label, obs, pred, obs_full=None):
    obs_full = obs_full if obs_full is not None else obs
    s_all = stats(obs, pred)
    dist_mask = obs > DISTURBED_KMS
    s_dist = stats(obs[dist_mask], pred[dist_mask]) if dist_mask.any() else None
    print(f"{label:38s}  n={s_all['n']:6d}  rmse={s_all['rmse']:6.1f}  "
          f"mae={s_all['mae']:6.1f}  bias={s_all['bias']:7.1f}  corr={s_all['corr']:6.3f}"
          + (f"   | disturbed: rmse={s_dist['rmse']:6.1f} bias={s_dist['bias']:7.1f} "
             f"corr={s_dist['corr']:6.3f}" if s_dist else ""))
    return s_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--alpha", type=float, default=5.0,
                     help="regularization strength on standardized features "
                          "(default 5.0; higher = more conservative "
                          "correction, appropriate given the small-N event count)")
    ap.add_argument("--objective", choices=("rmse", "corr"), default="rmse",
                     help="'rmse' = closed-form ridge (minimizes squared "
                          "error -- shown 2026-09-11 to consistently hurt "
                          "held-out disturbed-event correlation); 'corr' = "
                          "directly maximize correlation between the "
                          "corrected prediction and obs instead (see "
                          "corr_fit docstring)")
    ap.add_argument("--event-threshold", type=float, default=600.0,
                     help="km/s defining a 'major event' for the LOO folds")
    ap.add_argument("--donki", default=None,
                     help="path to a saved DONKI CME catalog (see "
                          "donki_features.load_events / donki_cme_2026.json) "
                          "to add CME-catalog features; omit to run the "
                          "AWSoM-R-state-only feature set for comparison")
    ap.add_argument("--donki-window-hours", type=float, default=72.0,
                     help="trailing window for --donki-mode activity's "
                          "count/max-speed features")
    ap.add_argument("--donki-mode", choices=("arrival", "calibrated_arrival", "activity"),
                     default="arrival",
                     help="'arrival' = naive constant-speed transit time "
                          "(shown not to work); 'calibrated_arrival' = "
                          "transit time from the Richardson/Cane-fitted "
                          "formula (see donki_features.CALIBRATED_TRANSIT_*); "
                          "'activity' = coarse recent-CME-count/max-speed, "
                          "no transit timing at all")
    args = ap.parse_args()

    global FEATURES
    donki_arrivals = donki_events_raw = None
    if args.donki:
        donki_events = load_events(args.donki)
        if args.donki_mode == "arrival":
            donki_arrivals = build_index(donki_events)
            FEATURES = FEATURES + DONKI_ARRIVAL_FEATURES
        elif args.donki_mode == "calibrated_arrival":
            donki_arrivals = build_calibrated_index(donki_events)
            FEATURES = FEATURES + DONKI_ARRIVAL_FEATURES
        else:
            donki_events_raw = donki_events
            FEATURES = FEATURES + DONKI_ACTIVITY_FEATURES
        print(f"loaded {len(donki_events)} DONKI CME entries with a speed "
              f"estimate from {args.donki} (mode={args.donki_mode})\n"
              f"features: {FEATURES}\n")

    doc = fetch(args.url, args.json)
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])

    bins, data = build_dataset(model_bins, obs_bins, donki_arrivals, donki_events_raw,
                                donki_window_hours=args.donki_window_hours)
    if not data:
        raise SystemExit("no usable rows -- check the fetched data")
    print(f"dataset: {len(bins)} rows with full features "
          f"(model + obs + 27d-prior obs + 24h trailing coverage)\n")

    events = event_windows(obs_bins, threshold=args.event_threshold)
    print(f"{len(events)} major event window(s) at {args.event_threshold:g} km/s "
          f"(+/-24h padding) used for leave-one-event-out folds\n")

    X_all = feature_matrix(data)
    y_all = data["target"]
    obs_all = data["obs_v"]
    blend_all = data["blend_v"]

    # leave-one-event-out: held-out predictions for every event row, using a
    # model that never saw that event during training
    held_out_pred = np.full(len(bins), np.nan)
    held_out_mask = np.zeros(len(bins), dtype=bool)
    for i, (start_b, end_b, peak) in enumerate(events):
        test_mask = (bins >= start_b) & (bins <= end_b)
        if not test_mask.any():
            continue
        # exclude ONLY this fold's held-out event -- training must still
        # include every OTHER event, or the model never sees a single
        # disturbed-period example and has no way to learn disturbance
        # behavior at all (caught 2026-09-11: an earlier version of this
        # loop excluded every event from every fold by mistake, which
        # produced a confidently-wrong "Phase 2 makes things worse" result
        # for the right-sounding wrong reason -- the model wasn't bad, it
        # was never shown a disturbance during training)
        train_mask = ~test_mask
        if args.objective == "corr":
            mean, std, coef, intercept = corr_fit(
                X_all[train_mask], obs_all[train_mask], blend_all[train_mask], args.alpha)
        else:
            mean, std, coef, intercept = ridge_fit(X_all[train_mask], y_all[train_mask],
                                                    args.alpha)
        pred_correction = ridge_predict(X_all[test_mask], mean, std, coef, intercept)
        held_out_pred[test_mask] = blend_all[test_mask] + pred_correction
        held_out_mask |= test_mask

    # quiet/non-event rows: no held-out event to protect here, so fit on
    # the full dataset (every event + every quiet row) for max data usage
    if args.objective == "corr":
        mean, std, coef, intercept = corr_fit(X_all, obs_all, blend_all, args.alpha)
    else:
        mean, std, coef, intercept = ridge_fit(X_all, y_all, args.alpha)
    quiet_mask = ~held_out_mask
    quiet_pred = blend_all.copy()
    quiet_pred[quiet_mask] = (blend_all[quiet_mask]
                              + ridge_predict(X_all[quiet_mask], mean, std, coef, intercept))

    full_pred = np.where(held_out_mask, held_out_pred, quiet_pred)

    print("Coefficients (on standardized features, fit on the full dataset "
          "-- shown for interpretation only, LOO folds above used their own "
          "per-fold fit):")
    for name, c in zip(FEATURES, coef):
        print(f"  {name:16s}  {c:+7.2f}")
    print(f"  {'intercept':16s}  {intercept:+7.2f}\n")

    print("Comparison (all rows in the built dataset):")
    report("raw model", obs_all, data["model_v"])
    report("persist27", obs_all, data["persist_v"])
    report("Phase 1b blend (0.5 model + 0.5 persist)", obs_all, blend_all)
    report("Phase 2 (blend + ridge correction, LOO on events)", obs_all, full_pred)

    print("\nDisturbed-event rows ONLY, held-out predictions "
          "(the actual thing Phase 2 needs to fix):")
    if held_out_mask.any():
        report("Phase 1b blend, event rows", obs_all[held_out_mask],
               blend_all[held_out_mask])
        report("Phase 2, event rows (held-out)", obs_all[held_out_mask],
               held_out_pred[held_out_mask])
    else:
        print("  no event rows in the built dataset")


if __name__ == "__main__":
    main()
