#!/usr/bin/env python3
"""Onset early-warning forecast: given only information available NOW
(recent AWSoM-R model trend, real-time observed B-field state, DONKI CME
activity), predict solar wind speed H_HOURS ahead -- a genuine forecast
with real lead time, anchored on the current real observation, not a
same-time correction.

DIFFERENT PROBLEM from final_correction.py's disturbed-period correction:
that one improves the CURRENT nowcast value; this one is specifically
about anticipating an ONSET before it arrives, which is the one thing
naive persistence structurally cannot do (see
accuracy_improvement_tracking.txt LOG, 2026-09-12, "GENUINE FORWARD-
HORIZON FORECAST" for the full derivation, including the important
caveat that predicting the SAME time (0 lead time) from a recent
observation is trivial and NOT useful -- persistence-from-now already
gets that almost perfectly since the dashboard already shows the live
obs line; the only place real forecasting value exists is genuine lead
time into the unobserved future).

VALIDATED SWEET SPOT: 6-12 hours ahead. At this range, on the actual hard
case (currently quiet, about to become disturbed), this model roughly
DOUBLES naive persistence's correlation (e.g. at h=6h: 0.109 -> 0.209)
and meaningfully reduces bias. Beyond ~18h the advantage disappears and
reverses -- current B-field/model state stops carrying useful
information that far out, so this is deliberately NOT offered past 12h.

    python early_warning.py --hours 6
    python early_warning.py --hours 6 --live
"""
from __future__ import annotations

import argparse
import datetime

import numpy as np

from eval_harness import BIN_S, DISTURBED_KMS, fetch, to_bin_map
from donki_features import (build_arrival_index, cme_arrival_signal,
                             estimate_earth_vmax_lasco, load_events)
import phase2_model as p2

FEATURES = ("model_v", "trend_3h", "trend_6h", "volatility_24h", "disagreement",
            "cme_count_72h", "cme_maxspeed_72h") + p2.BFIELD_FEATURES
RIDGE_ALPHA = 5.0
VALID_HOURS = (6, 9, 12)  # outside this range the model loses to naive persistence

# CME-ARRIVAL-ANCHORED FORECAST FEATURE, added 2026-09-17 (real, off-by-
# default experiment -- see accuracy_improvement_tracking.txt LOG). Every
# feature this forecast has used so far is a generic trailing statistic
# (trend, volatility, B-field state, CME activity COUNT) -- none of them
# say "a SPECIFIC confirmed CME is expected to arrive inside THIS
# forecast's target window." final_correction.py already has exactly that
# information (a candidate-arrival index + a physics-based magnitude
# estimate, donki_features.cme_arrival_signal/estimate_earth_vmax_lasco)
# but has only ever used it to correct the PRESENT moment, never fed it to
# the FORECAST. New idea, untried until now: when a real candidate's
# predicted arrival lands inside a forecast's target window (within
# CME_ARRIVAL_TOLERANCE_HOURS), add its physics-based Earth-arrival
# magnitude estimate as an additional regression feature (not a hard
# override -- the ridge model learns its own weight for it, same as every
# other feature here), alongside a binary flag so "no candidate" and "a
# candidate predicting near-zero" can't be confused (same convention as
# arrival_feature's has_candidate design elsewhere in this project).
# Gated behind --cme-arrival, off by default until validated.
CME_ARRIVAL_MIN_SPEED_KMS = 650.0   # matches final_correction.py's CME_MIN_SPEED_KMS
CME_ARRIVAL_MAX_CMD = 60
CME_ARRIVAL_TOLERANCE_HOURS = 3.0   # how close a candidate's predicted arrival
                                      # must land to the forecast's target time
                                      # to count as "arriving in this window" --
                                      # a first reasonable guess, not yet swept
CME_ARRIVAL_FEATURES = ("cme_arrival_vmax", "cme_arrival_has_signal")


def _cme_arrivals():
    return build_arrival_index(load_events("donki_cme_2026.json"), CME_ARRIVAL_MIN_SPEED_KMS,
                                max_cmd=CME_ARRIVAL_MAX_CMD, include_unknown=True)


def _bfield_bins(obs):
    return {"b": to_bin_map(obs["t"], obs["b"]),
            "bz": to_bin_map(obs["t"], obs["bz"]),
            "n": to_bin_map(obs["t"], obs["n"])}


def _dataset(hours, use_cme_arrival=False):
    features = FEATURES + CME_ARRIVAL_FEATURES if use_cme_arrival else FEATURES
    p2.FEATURES = features
    doc = fetch()
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])
    donki_events = load_events("donki_cme_2026.json")
    bfield = _bfield_bins(obs)
    bins, data = p2.build_dataset(model_bins, obs_bins, donki_events=donki_events,
                                   bfield_bins=bfield)
    h_bins = round(hours * 3600 / BIN_S)
    if use_cme_arrival:
        arrivals = _cme_arrivals()
        cme_vmax = np.zeros(len(bins))
        cme_has = np.zeros(len(bins))
        for i, b in enumerate(bins):
            t_now = int(b) * BIN_S
            t_target = t_now + hours * 3600.0
            has, speed = cme_arrival_signal(t_now, t_target, arrivals,
                                             tolerance_hours=CME_ARRIVAL_TOLERANCE_HOURS)
            if has:
                cme_vmax[i] = estimate_earth_vmax_lasco(speed)
                cme_has[i] = 1.0
        data["cme_arrival_vmax"] = cme_vmax
        data["cme_arrival_has_signal"] = cme_has
    X_all = p2.feature_matrix(data)
    obs_all = data["obs_v"]
    future_v = np.array([obs_bins.get(b + h_bins, np.nan) for b in bins])
    valid = ~np.isnan(future_v)
    return (bins[valid], X_all[valid], obs_all[valid], future_v[valid],
            model_bins, obs_bins, donki_events, bfield)


def loo_validate(hours, use_cme_arrival=False):
    bins, X_all, anchor, future_v, model_bins, obs_bins, _, _ = _dataset(hours, use_cme_arrival)
    target = future_v - anchor
    events = p2.event_windows(obs_bins, threshold=600.0)
    onset_mask = (anchor <= 500) & (future_v > 500)

    naive_pred = anchor.copy()
    learned_pred = naive_pred.copy()
    held_out_mask = np.zeros(len(bins), dtype=bool)
    for start_b, end_b, peak in events:
        test_mask = (bins >= start_b) & (bins <= end_b)
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        mean, std, coef, intercept = p2.ridge_fit(X_all[train_mask], target[train_mask], RIDGE_ALPHA)
        pred_delta = p2.ridge_predict(X_all[test_mask], mean, std, coef, intercept)
        learned_pred[test_mask] = anchor[test_mask] + pred_delta
        held_out_mask |= test_mask
    mean, std, coef, intercept = p2.ridge_fit(X_all, target, RIDGE_ALPHA)
    non_event = ~held_out_mask
    learned_pred[non_event] = anchor[non_event] + p2.ridge_predict(X_all[non_event], mean, std, coef, intercept)

    print(f"Onset early-warning, {hours}h ahead, leave-one-event-out validation:\n")
    s_naive_all = p2.stats(future_v, naive_pred)
    s_learned_all = p2.stats(future_v, learned_pred)
    print(f"  ALL rows:    naive rmse={s_naive_all['rmse']:.1f} corr={s_naive_all['corr']:.3f}  |  "
          f"learned rmse={s_learned_all['rmse']:.1f} corr={s_learned_all['corr']:.3f}")
    s_naive_onset = p2.stats(future_v[onset_mask], naive_pred[onset_mask])
    s_learned_onset = p2.stats(future_v[onset_mask], learned_pred[onset_mask])
    print(f"  ONSET rows ({onset_mask.sum()}, currently quiet -> becomes disturbed): "
          f"naive rmse={s_naive_onset['rmse']:.1f} bias={s_naive_onset['bias']:.1f} "
          f"corr={s_naive_onset['corr']:.3f}  |  learned rmse={s_learned_onset['rmse']:.1f} "
          f"bias={s_learned_onset['bias']:.1f} corr={s_learned_onset['corr']:.3f}")
    if hours not in VALID_HOURS:
        print(f"\n  WARNING: {hours}h is outside the validated 6-12h sweet spot -- "
              f"the learned model has been shown to underperform naive persistence "
              f"beyond ~12h. Numbers above are real but don't assume they generalize.")


# CLASSIFICATION REFRAME, added 2026-09-17 (item #3 of the "11 things"
# Jaxon logged after reading the teaching doc -- see
# accuracy_improvement_tracking.txt LOG). Every version of this forecast
# tried so far predicts the EXACT future speed value via ridge regression,
# capping out around corr 0.21 on the hard onset case. The nowcast
# classifier (final_correction.py/uncertainty_model.py) asks an easier
# question on the SAME kind of features -- not "what is the exact value"
# but "is this a disturbance" -- and gets real discrimination (AUC
# 0.75+). This has never been tried for the FORECAST target: instead of
# "what will the speed be at t+h," ask "will it be disturbed at t+h."
# LOGISTIC_ALPHA_CLASSIFY: started from the nowcast classifier's own
# value (0.005) as a first guess, then actually swept properly (0.001 to
# 10.0, at h=6h) rather than trusting a borrowed value -- this project has
# been burned twice before by assuming regularization strength transfers
# across objectives (see Model Summary Section 2). Found a REAL interior
# peak (onset AUC rises 0.001->0.3, then falls hard by 10.0 -- 0.855 at
# the peak down to 0.618, a genuine cliff, the same failure shape as both
# earlier regularization traps) -- not just a monotonic edge effect, a
# much more trustworthy signal than the earlier "flat, gently rising"
# read at 0.005 gave. 0.5 picked from the middle of the real 0.3-1.0
# plateau (same "middle of a stable region, not the single best point"
# convention as CME_MIN_SPEED_KMS elsewhere in this project), not the
# single top value found (0.3, AUC 0.855) -- still checked in the SAME
# LOO folds being reported, not an independently held-out tuning pass,
# flagged honestly rather than treated as fully settled.
LOGISTIC_ALPHA_CLASSIFY = 0.5

# VALID_HOURS_CLASSIFY: the regression version (VALID_HOURS, above)
# reverses past ~18h. Checked whether the classification reframe has the
# same ceiling or a different one -- swept 6h through 36h at the tuned
# alpha and found a MEANINGFULLY WIDER real range: onset AUC keeps
# climbing through 15-18h (peak ~0.83-0.85) and stays clearly,
# substantially above the naive baseline all the way to 36h (the longest
# horizon checked), unlike the regression framing's actual reversal.
# 36h was the longest horizon tested, not a hard discovered ceiling --
# untried whether it extends further still.
VALID_HOURS_CLASSIFY = (6, 9, 12, 15, 18, 24, 30, 36)


def classify_series(hours, alpha=LOGISTIC_ALPHA_CLASSIFY, use_cme_arrival=False):
    """THE single shared implementation of the classification-reframed
    forecast's leave-one-event-out series -- returns a dict with bins,
    y_future, quiet_now_mask, proba_future (future-trained classifier)
    and proba_now_baseline (today's-classifier-reused-naively). Extracted
    2026-09-17 so loo_validate_classify() (reporting) and
    export_dashboard_data.py (historical export) both call ONE
    implementation instead of each re-deriving this fold loop -- this
    project has hit that exact duplication-bug class twice before
    (final_correction.py's apply_to_live()/export_dashboard_data.py both
    drifting out of sync on 2026-09-14); not repeating it a third time
    for this new piece.
    """
    bins, X_all, anchor, future_v, model_bins, obs_bins, _, _ = _dataset(hours, use_cme_arrival)
    events = p2.event_windows(obs_bins, threshold=600.0)
    y_future = (future_v > DISTURBED_KMS).astype(float)
    y_now = (anchor > DISTURBED_KMS).astype(float)
    quiet_now_mask = anchor <= DISTURBED_KMS

    proba_future = np.zeros(len(bins))
    proba_now_baseline = np.zeros(len(bins))
    held_out_mask = np.zeros(len(bins), dtype=bool)
    for start_b, end_b, peak in events:
        test_mask = (bins >= start_b) & (bins <= end_b)
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        mean_f, std_f, w_f, b_f = p2.logistic_fit(X_all[train_mask], y_future[train_mask], alpha)
        proba_future[test_mask] = p2.predict_proba(X_all[test_mask], mean_f, std_f, w_f, b_f)
        mean_n, std_n, w_n, b_n = p2.logistic_fit(X_all[train_mask], y_now[train_mask], alpha)
        proba_now_baseline[test_mask] = p2.predict_proba(X_all[test_mask], mean_n, std_n, w_n, b_n)
        held_out_mask |= test_mask
    mean_f, std_f, w_f, b_f = p2.logistic_fit(X_all, y_future, alpha)
    mean_n, std_n, w_n, b_n = p2.logistic_fit(X_all, y_now, alpha)
    non_event = ~held_out_mask
    proba_future[non_event] = p2.predict_proba(X_all[non_event], mean_f, std_f, w_f, b_f)
    proba_now_baseline[non_event] = p2.predict_proba(X_all[non_event], mean_n, std_n, w_n, b_n)

    return {"bins": bins, "y_future": y_future, "quiet_now_mask": quiet_now_mask,
            "proba_future": proba_future, "proba_now_baseline": proba_now_baseline}


def loo_validate_classify(hours, alpha=LOGISTIC_ALPHA_CLASSIFY, use_cme_arrival=False):
    """Reframe the forecast as classification: P(obs(t+h) > DISTURBED_KMS)
    from features available at t, instead of regressing the exact value.

    Reports AUC in two ways, because the honest, hard case is not the
    same as the easy one: "ALL rows" includes rows that are ALREADY
    disturbed now and simply stay that way at t+h (closer to persistence,
    inflates the number) -- the real test this project has always cared
    about is "QUIET-NOW rows" (can a coming ONSET be predicted before it
    happens, the same hard case loo_validate's regression version scores
    separately). Also reports a naive baseline: a classifier trained on
    the CURRENT disturbance label (i.e. exactly today's nowcast
    classifier) but evaluated against the FUTURE label -- this isolates
    whether training specifically on the future target beats just reusing
    the present-moment classifier's own score as a forecast.
    """
    s = classify_series(hours, alpha, use_cme_arrival)
    bins, y_future, quiet_now_mask = s["bins"], s["y_future"], s["quiet_now_mask"]
    proba_future, proba_now_baseline = s["proba_future"], s["proba_now_baseline"]

    auc_all = p2.auc(y_future, proba_future)
    auc_all_baseline = p2.auc(y_future, proba_now_baseline)
    n_onset_pool = int(quiet_now_mask.sum())
    auc_onset = p2.auc(y_future[quiet_now_mask], proba_future[quiet_now_mask])
    auc_onset_baseline = p2.auc(y_future[quiet_now_mask], proba_now_baseline[quiet_now_mask])

    print(f"Onset early-warning, CLASSIFICATION reframe, {hours}h ahead, "
          f"alpha={alpha}, leave-one-event-out validation:\n")
    print(f"  ALL rows ({len(bins)}): AUC(future-trained)={auc_all:.3f}  |  "
          f"AUC(today's-classifier-reused-naively)={auc_all_baseline:.3f}")
    print(f"  QUIET-NOW rows ({n_onset_pool}, the real onset-prediction case -- "
          f"currently quiet, does it become disturbed by t+{hours:g}h):")
    print(f"    AUC(future-trained)={auc_onset:.3f}  |  "
          f"AUC(today's-classifier-reused-naively)={auc_onset_baseline:.3f}")
    if hours not in VALID_HOURS_CLASSIFY:
        print(f"\n  WARNING: {hours}h has not been checked for the classification "
              f"reframe (validated range: {VALID_HOURS_CLASSIFY} hours) -- treat "
              f"numbers outside that set as exploratory.")
    return {"auc_all": auc_all, "auc_all_baseline": auc_all_baseline,
            "auc_onset": auc_onset, "auc_onset_baseline": auc_onset_baseline,
            "n_onset_pool": n_onset_pool}


def apply_to_live_classify(hours, alpha=LOGISTIC_ALPHA_CLASSIFY, use_cme_arrival=False):
    """Live onset probability: P(obs(t+hours) > DISTURBED_KMS), fit on the
    FULL historical dataset (no held-out event for a genuinely new live
    point -- same convention as this project's other apply_to_live()
    functions, which fit their flat_shift/regression on all available
    data rather than a LOO fold) and applied to the current live feature
    vector.
    """
    bins, X_all, anchor, future_v, model_bins, obs_bins, donki_events, bfield = _dataset(
        hours, use_cme_arrival)
    y_future = (future_v > DISTURBED_KMS).astype(float)
    mean_f, std_f, w_f, b_f = p2.logistic_fit(X_all, y_future, alpha)

    b0, live_row = p2.live_features(model_bins, obs_bins, bfield_bins=bfield,
                                     donki_events=donki_events)
    features = FEATURES
    if use_cme_arrival:
        t_now = b0 * BIN_S
        t_target = t_now + hours * 3600.0
        has, speed = cme_arrival_signal(t_now, t_target, _cme_arrivals(),
                                         tolerance_hours=CME_ARRIVAL_TOLERANCE_HOURS)
        live_row["cme_arrival_vmax"] = estimate_earth_vmax_lasco(speed) if has else 0.0
        live_row["cme_arrival_has_signal"] = 1.0 if has else 0.0
        features = FEATURES + CME_ARRIVAL_FEATURES

    x_live = p2.live_feature_vector(live_row, features)
    proba_live = float(p2.predict_proba(x_live, mean_f, std_f, w_f, b_f)[0])

    when = datetime.datetime.utcfromtimestamp(b0 * BIN_S)
    target_when = when + datetime.timedelta(hours=hours)
    print(f"Onset probability issued at {when} UTC, for {target_when} UTC ({hours}h ahead):")
    print(f"  P(disturbed by then): {proba_live:.3f}")
    return {"when": when, "target_when": target_when, "hours": hours, "proba": proba_live}


# ONSET-RISK v2, added 2026-10-06 (tests: accuracy_improvement_tracking.txt,
# "ONSET-RISK CLASSIFIER -- WHAT WOULD IMPROVE IT"). Three changes to the
# classification forecast above, each checked on held-out data:
#  1. TRAIN ON QUIET-NOW ROWS ONLY. The v1 classifier trains on every row
#     (30%+ already disturbed) but is only ever shown for quiet wind, where
#     the true 6h onset rate is ~5%. Its printed percentages ran 18-24% where
#     the observed rate was 0.1-4.6% (Brier skill vs the base rate -0.63 at
#     6h). Training on the population it is applied to fixes the scale and
#     raises AUC.
#  2. WEAKER REGULARISATION (ONSET_V2_ALPHA 0.01). With quiet-only training
#     alpha 0.5 squeezes probabilities toward the middle; AUC/Brier are flat
#     between 0.003 and 0.03. (Chosen on the same folds that are reported --
#     the curve is flat, so the effect is small, but it is not a fresh holdout.)
#  3. EXTRA CAUSAL SPEED FEATURES, by lead time (what the tests supported):
#       <=12h  "speed+27d": measured speed now, its 1h/3h change, 24h max, and
#              the speed measured 27 days before the TARGET time (+-3h max)
#       >=30h  "27d": only the 27-day-ago-at-target features
#       else   "base": the original features (no extra gain was found at 12-24h)
# Validation is leave-one-10-day-block-out with a 2-day purge each side, so
# EVERY row is scored by a model that never saw it (the v1 leave-one-event-out
# convention scored non-event rows with a model that had).
ONSET_V2_ALPHA = 0.01
ONSET_V2_BLOCK_BINS = 10 * 144
ONSET_V2_PURGE_BINS = 2 * 144
_ROT_BINS = round(27 * 86400 / BIN_S)
_DAY_BINS = round(86400 / BIN_S)


def onset_v2_feature_set(hours):
    if hours <= 12:
        return "speed+27d"
    if hours >= 30:
        return "27d"
    return "base"


def _onset_v2_extra_row(obs_bins, b, h_bins, feature_set):
    """Extra features for one bin b (causal: only observations up to b, plus the
    27-day-old record around the target time). None if anything is missing."""
    if feature_set == "base":
        return []
    c = int(b) + h_bins - _ROT_BINS
    window = [obs_bins[bb] for bb in range(c - 18, c + 19) if bb in obs_bins]
    pt = obs_bins.get(c)
    anchor = obs_bins.get(int(b))
    if pt is None or not window or anchor is None:
        return None
    persist_feats = [pt, max(window), pt - anchor]
    if feature_set == "27d":
        return persist_feats
    d1 = anchor - obs_bins.get(int(b) - 6, anchor)
    d3 = anchor - obs_bins.get(int(b) - 18, anchor)
    day = [obs_bins[bb] for bb in range(int(b) - _DAY_BINS, int(b) + 1) if bb in obs_bins]
    return [anchor, d1, d3, max(day)] + persist_feats


def _onset_v2_dataset(hours):
    bins, X, anchor, future_v, model_bins, obs_bins, donki_events, bfield = _dataset(hours)
    h_bins = round(hours * 3600 / BIN_S)
    fs = onset_v2_feature_set(hours)
    rows, keep = [], np.zeros(len(bins), dtype=bool)
    for i, b in enumerate(bins):
        extra = _onset_v2_extra_row(obs_bins, b, h_bins, fs)
        if extra is not None:
            keep[i] = True
            rows.append(extra)
    extra_mat = np.array(rows, dtype=float).reshape(int(keep.sum()), -1)
    X2 = np.hstack([X[keep], extra_mat]) if extra_mat.shape[1] else X[keep]
    return (bins[keep], X2, anchor[keep], future_v[keep], model_bins, obs_bins,
            donki_events, bfield, fs)


def classify_series_v2(hours, alpha=ONSET_V2_ALPHA):
    """Held-out P(measured > DISTURBED_KMS at t+hours) for every row, trained on
    quiet-now rows only, leave-one-10-day-block-out. Probabilities are only
    meaningful for quiet-now rows (NaN elsewhere)."""
    bins, X, anchor, future_v, _, _, _, _, fs = _onset_v2_dataset(hours)
    y = (future_v > DISTURBED_KMS).astype(float)
    quiet = anchor <= DISTURBED_KMS
    ids = (bins - bins.min()) // ONSET_V2_BLOCK_BINS
    proba = np.full(len(bins), np.nan)
    for k in np.unique(ids):
        test = ids == k
        lo = bins[test].min() - ONSET_V2_PURGE_BINS
        hi = bins[test].max() + ONSET_V2_PURGE_BINS
        train = ~((bins >= lo) & (bins <= hi)) & quiet
        m, s, w, b = p2.logistic_fit(X[train], y[train], alpha)
        te = test & quiet
        if te.any():
            proba[te] = p2.predict_proba(X[te], m, s, w, b)
    return {"bins": bins, "y_future": y, "quiet_now_mask": quiet, "proba_future": proba,
            "feature_set": fs}


def loo_validate_classify_v2(hours, alpha=ONSET_V2_ALPHA):
    s = classify_series_v2(hours, alpha)
    q, y, p = s["quiet_now_mask"], s["y_future"], s["proba_future"]
    auc_q = p2.auc(y[q], p[q])
    brier = float(np.mean((p[q] - y[q]) ** 2))
    clim = float(np.mean((y[q].mean() - y[q]) ** 2))
    print(f"Onset-risk v2 ({s['feature_set']} features), {hours:g}h ahead, alpha={alpha}, "
          f"leave-one-10-day-block-out:")
    print(f"  QUIET-NOW rows ({int(q.sum())}, base rate {y[q].mean()*100:.1f}%): "
          f"AUC={auc_q:.3f}  Brier skill vs base rate={1 - brier / clim:+.3f}")
    print("  reliability (predicted -> observed):")
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.01)):
        m = (p[q] >= lo) & (p[q] < hi)
        if m.any():
            print(f"    [{lo:.2f},{min(hi, 1):.2f}) n={int(m.sum()):6d}  "
                  f"{p[q][m].mean():.3f} -> {y[q][m].mean():.3f}")
    return {"auc": auc_q, "brier_skill": 1 - brier / clim, "n": int(q.sum())}


def apply_to_live_classify_v2(hours, alpha=ONSET_V2_ALPHA):
    """Live onset probability from the v2 classifier, fit on all quiet-now
    history. Raises ValueError if the live row is missing an input (data gap)."""
    bins, X, anchor, future_v, model_bins, obs_bins, donki_events, bfield, fs = \
        _onset_v2_dataset(hours)
    y = (future_v > DISTURBED_KMS).astype(float)
    quiet = anchor <= DISTURBED_KMS
    mean_f, std_f, w_f, b_f = p2.logistic_fit(X[quiet], y[quiet], alpha)

    b0, live_row = p2.live_features(model_bins, obs_bins, bfield_bins=bfield,
                                     donki_events=donki_events)
    x_base = p2.live_feature_vector(live_row, FEATURES)
    extra = _onset_v2_extra_row(obs_bins, b0, round(hours * 3600 / BIN_S), fs)
    if extra is None:
        raise ValueError("live onset-risk v2 inputs incomplete (measured-speed gap)")
    x_live = np.hstack([x_base, np.array([extra], dtype=float)]) if extra else x_base
    proba_live = float(p2.predict_proba(x_live, mean_f, std_f, w_f, b_f)[0])
    quiet_now = obs_bins.get(b0, np.inf) <= DISTURBED_KMS
    when = datetime.datetime.utcfromtimestamp(b0 * BIN_S)
    target_when = when + datetime.timedelta(hours=hours)
    print(f"Onset probability (v2, {fs}) issued at {when} UTC, for {target_when} UTC "
          f"({hours}h ahead):")
    print(f"  P(above {DISTURBED_KMS:g} km/s at that time): {proba_live:.3f}"
          + ("" if quiet_now else "   [already disturbed now -- not meaningful]"))
    return {"when": when, "target_when": target_when, "hours": hours,
            "proba": proba_live, "quiet_now": bool(quiet_now), "feature_set": fs,
            "measured_now": obs_bins.get(b0)}


def apply_to_live(hours, use_cme_arrival=False):
    bins, X_all, anchor, future_v, model_bins, obs_bins, donki_events, bfield = _dataset(
        hours, use_cme_arrival)
    target = future_v - anchor
    mean, std, coef, intercept = p2.ridge_fit(X_all, target, RIDGE_ALPHA)

    b0, live_row = p2.live_features(model_bins, obs_bins, bfield_bins=bfield,
                                     donki_events=donki_events)
    mv = live_row["model_v"]

    anchor_now = obs_bins.get(b0)
    if anchor_now is None:
        anchor_now = mv  # fall back to model if no obs at exactly this bin

    features = FEATURES
    cme_signal = None
    if use_cme_arrival:
        t_now = b0 * BIN_S
        t_target = t_now + hours * 3600.0
        has, speed = cme_arrival_signal(t_now, t_target, _cme_arrivals(),
                                         tolerance_hours=CME_ARRIVAL_TOLERANCE_HOURS)
        live_row["cme_arrival_vmax"] = estimate_earth_vmax_lasco(speed) if has else 0.0
        live_row["cme_arrival_has_signal"] = 1.0 if has else 0.0
        features = FEATURES + CME_ARRIVAL_FEATURES
        if has:
            cme_signal = {"donki_speed": speed, "predicted_vmax": live_row["cme_arrival_vmax"]}

    x_live = p2.live_feature_vector(live_row, features)
    pred_delta = p2.ridge_predict(x_live, mean, std, coef, intercept)[0]
    forecast_v = anchor_now + pred_delta

    when = datetime.datetime.utcfromtimestamp(b0 * BIN_S)
    target_when = when + datetime.timedelta(hours=hours)
    print(f"Early-warning forecast issued at {when} UTC, for {target_when} UTC ({hours}h ahead):")
    print(f"  current observed:         {anchor_now:.1f} km/s")
    print(f"  naive (assume no change): {anchor_now:.1f} km/s")
    print(f"  early-warning forecast:   {forecast_v:.1f} km/s  (predicted change: {pred_delta:+.1f})")
    if use_cme_arrival:
        if cme_signal:
            print(f"  CME arrival signal:      DONKI speed {cme_signal['donki_speed']:.0f} km/s "
                  f"expected to arrive in this window, physics estimate "
                  f"{cme_signal['predicted_vmax']:.0f} km/s (fed as a feature, not a hard override)")
        else:
            print(f"  CME arrival signal:      none expected in this window")
    return {"when": when, "target_when": target_when, "hours": hours,
            "anchor_now": anchor_now, "forecast": forecast_v, "pred_delta": pred_delta,
            "cme_signal": cme_signal}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=6.0,
                     help="forecast lead time in hours (validated sweet spot: 6-12)")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--cme-arrival", action="store_true",
                     help="EXPERIMENTAL (added 2026-09-17), off by default: add a "
                          "feature for whether a real, confirmed DONKI CME candidate "
                          "is expected to arrive inside this forecast's target window "
                          "(within CME_ARRIVAL_TOLERANCE_HOURS), using the same "
                          "physics-based Earth-arrival magnitude estimate "
                          "final_correction.py uses for the present-moment "
                          "correction -- untried for the FORECAST target before now. "
                          "Not a hard override; the ridge regression learns its own "
                          "weight for it alongside every other feature.")
    ap.add_argument("--classify", action="store_true",
                     help="EXPERIMENTAL (added 2026-09-17), off by default: reframe "
                          "the forecast as classification (P(disturbed at t+hours)) "
                          "instead of regressing the exact future speed value -- "
                          "untried before now. Reports AUC instead of RMSE/corr; "
                          "not compatible with --live in this first pass.")
    ap.add_argument("--classify-alpha", type=float, default=LOGISTIC_ALPHA_CLASSIFY,
                     help=f"regularization strength for --classify (default "
                          f"{LOGISTIC_ALPHA_CLASSIFY}, the nowcast classifier's own "
                          f"validated value -- not independently swept for this new "
                          f"target beyond a quick sanity check, per this project's "
                          f"standing regularization-scale caution)")
    ap.add_argument("--classify-v2", action="store_true",
                     help="onset-risk v2 (added 2026-10-06): quiet-now-only training, "
                          "alpha 0.01, lead-time-specific speed features, validated "
                          "leave-one-10-day-block-out. Add --live for the live value.")
    args = ap.parse_args()
    if args.classify_v2:
        if args.live:
            apply_to_live_classify_v2(args.hours)
        else:
            loo_validate_classify_v2(args.hours)
    elif args.classify:
        loo_validate_classify(args.hours, args.classify_alpha, args.cme_arrival)
    elif args.live:
        apply_to_live(args.hours, args.cme_arrival)
    else:
        loo_validate(args.hours, args.cme_arrival)
