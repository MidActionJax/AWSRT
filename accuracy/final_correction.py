#!/usr/bin/env python3
"""LOCKED Phase 2 correction pipeline, as of 2026-09-16 (LASCO-calibrated
magnitude shift PROMOTED to the default this day -- see below;
CME-magnitude shift originally promoted 2026-09-14; classifier UNIFIED
with uncertainty_model.py on 2026-09-12).

Blend (Phase 1b) + a classifier-gated, continuously-scaled magnitude
shift (the "smoothed tradeoff" result -- see
accuracy_improvement_tracking.txt LOG, "SMOOTHED THE TRADEOFF" and
"LOCKED, 2026-09-12" entries for the full derivation and the reasoning
behind picking power=1.0).

    corrected(t) = blend(t) + P(disturbed | features(t))^POWER * SHIFT(t)

Where:
  blend       = 0.5*raw_model + 0.5*persist27          (Phase 1b, validated)
  P(disturbed) = logistic regression on causally-available features
                 (AWSoM-R trend/volatility/disagreement + real-time
                 observed B-field), trained on historical obs >
                 DISTURBED_KMS labels
  SHIFT(t)    = a per-event CME-magnitude estimate (see "CME-MAGNITUDE
                SHIFT PROMOTED" below) when a fast, Earth-directed DONKI
                candidate exists near t, else FLAT_SHIFT = mean(obs -
                blend) over historical disturbed rows ("how much does
                the blend typically underpredict during any real
                disturbance") -- the sole behavior before 2026-09-14.
  POWER = 1.0, chosen deliberately (not the technical optimum on any
          single metric): real, substantial improvement to the
          disturbed-period problem that motivated this entire project,
          at a bounded, well-tolerated cost to quiet-period bias (which
          had much more room to give, having started near zero).

CME-MAGNITUDE SHIFT PROMOTED TO DEFAULT (2026-09-14): a first attempt
(any Earth-directed DONKI CME as a candidate, no speed floor) was a real
negative (see LOG "TIER 0a IMPLEMENTED AND TESTED") -- ~100% of
disturbed rows got SOME candidate regardless of relevance, diluting the
magnitude estimate. Fixed by discovering and using CLEAR's OWN real
production CME-selection rule (RealTimeCME/cmetrigger, a sibling repo:
real-time mode only simulates CMEs with speed >= 800 km/s) instead of
"any candidate" -- see CME_MIN_SPEED_KMS below and LOG "TIER 0a FIXED"
for the full sweep. Result: disturbed-row correlation crosses
meaningfully positive for the first time in this project (-0.004 ->
+0.044), overall corr 0.528 -> 0.591. Checked per-event, not just in
aggregate (LOG, 2026-09-14 "per-event breakdown"): wins outright on
27/35 held-out event windows (ties on 2 more with no candidate at all),
and the pattern is not random -- losses are small (RMSE worse by
2-22 km/s) and concentrated on already-mild disturbances that needed
little correction to begin with, while wins are large (RMSE better by
30-140 km/s, bias corrected from -100/-140 km/s down to single digits)
and concentrated on exactly the most severe storms. Promoted on that
basis: worth trading a little precision on calm-ish moments for
reliably catching the big ones, which is what this project is actually
for. `--no-cme-magnitude` reverts to the pre-2026-09-14 flat-shift-only
behavior for comparison.

LASCO-CALIBRATED MAGNITUDE SHIFT PROMOTED TO DEFAULT (2026-09-16): the
CME-magnitude estimate above (estimate_earth_vmax, 12 pairs, 2026-only)
underpredicts real event peaks by a systematic +78 km/s -- confirmed via
a local-recalibration attempt the same day that fixed the bias but
collapsed to a nearly-flat slope (stops depending on donki_speed at all),
too risky to promote (real overfitting risk to this project's own small,
unusually-severe test window -- see accuracy_improvement_tracking.txt
LOG). Fixed properly instead: Richardson & Cane's own catalog already
records an exact LASCO CME timestamp for most of its verified rows,
matched against the real, public SOHO/LASCO CME catalog (continuous
since 1996) to build a 303-pair external calibration (donki_features.
LASCO_MAGNITUDE_A/B, see lasco_catalog.py) -- 25x the original sample,
spanning ~2.5 solar cycles instead of one recent year. Using that fit's
absolute scale as-is is a clean negative (this project's own test window
runs more severe than the general historical population), so the locked
mechanism keeps LASCO's robust SLOPE (immune to collapsing flat, since it
is fit on 25x the data) and LOO-shifts only the INTERCEPT to this
project's own local severity level (_cme_candidates_lasco_hybrid).
Result: disturbed corr 0.230->0.295, RMSE 80.1->75.2, per-event 24
wins/9 losses/4 ties, worst-case only +7.5 RMSE -- half the risk of the
local-only recalibration that was held back the same day.
`--no-lasco-calibration` reverts to the pre-2026-09-16 12-pair external
calibration for comparison.

CLASSIFIER UNIFIED WITH uncertainty_model.py (2026-09-12): this pipeline's
classifier used to be fit separately (DONKI activity features + bfield,
alpha=0.01) from uncertainty_model.py's band classifier (bfield only,
alpha=0.005) -- an unreconciled loose end where the "same moment" got two
different P(disturbed) numbers depending which script you asked. Swept
properly (see accuracy_improvement_tracking.txt LOG, "CLASSIFIER
RECONCILIATION"): DONKI activity features add nothing here, and
uncertainty_model's own (bfield-only, alpha=0.005) config does AT LEAST
AS WELL on this pipeline's own disturbed-row correlation metric (-0.003
vs the old -0.005) with essentially identical RMSE. So FEATURES/alpha
below now exactly match uncertainty_model.py's -- same feature set, same
regularization, same causal inputs -- meaning the two scripts' classifiers
are now mathematically the same fit on the same data, not just similar.

    python final_correction.py                 # LOO-validate the locked config
    python final_correction.py --live          # apply to the current live point
    python final_correction.py --no-cme-magnitude   # pre-2026-09-14 behavior, for comparison

This is still RESEARCH, not deployed anywhere -- see the project's
standing no-partial-deploy rule. This script exists so the locked
configuration is one reproducible thing, not scattered shell history.
"""
from __future__ import annotations

import argparse
import datetime

import numpy as np

from eval_harness import BIN_S, DISTURBED_KMS, fetch, to_bin_map
from donki_features import (arrival_feature, build_arrival_index,
                             build_calibrated_index,
                             estimate_earth_vmax, estimate_earth_vmax_lasco,
                             filter_earth_directed, load_events,
                             load_donki_events_with_width,
                             load_enlil_earth_arrivals,
                             load_production_cme_events)
from uncertainty_model import ew_predictions, logistic_fit, predict_proba
import early_warning as ew
import phase2_model as p2

POWER = 1.0
LOGISTIC_ALPHA = 0.005
FEATURES = ("model_v", "trend_3h", "trend_6h", "volatility_24h", "disagreement") + p2.BFIELD_FEATURES

# Tier 0a (accuracy_improvement_tracking.txt, "THE REAL QUESTION" brainstorm,
# 2026-09-12): estimate_earth_vmax() is an already-calibrated (corr 0.807 on
# 12 Richardson/Cane-verified pairs) mapping from a DONKI CME's catalog speed
# to its predicted arrival speed at Earth -- previously defined and never
# called anywhere. FLAT_SHIFT below is one constant applied to every
# disturbed row regardless of event size. This section replaces that
# constant, per-row, with a magnitude estimate for the SPECIFIC candidate
# CME when one exists -- falling back to the historical flat constant
# otherwise.
#
# FIRST ATTEMPT (2026-09-12/14) used every Earth-directed DONKI CME as a
# candidate and was a clean negative (see LOG "TIER 0a IMPLEMENTED AND
# TESTED"): with no speed floor, ~100% of disturbed rows get SOME nearby
# candidate regardless of whether it's the true driver, diluting the
# magnitude estimate toward a generic CME's modest effect.
#
# FIX (2026-09-14): reading CLEAR's own real-time CME-triggering pipeline
# (RealTimeCME/cmetrigger, a sibling repo) revealed the ACTUAL production
# selection rule for which DONKI CMEs are even worth simulating in
# real-time: speed >= 800 km/s (their `min_speed_kms`), a far more
# selective bar than "any Earth-directed CME." Re-swept our own downstream
# use of this idea against that same real threshold and neighbors
# (500-1200 km/s): every value in a broad 600-750 km/s plateau gives a
# real, clean improvement over the locked baseline (overall corr
# 0.528->0.585-0.591, disturbed-row corr crosses to +0.03 to +0.07 for the
# first time all day, disturbed RMSE 114.7->~95). CME_MIN_SPEED_KMS=650
# picked from the middle of that plateau, not the single best point --
# see LOG "REAL CME-SEP PIPELINE FOUND... CANDIDATE SELECTION FIX" for the
# full sweep and reasoning.
EARTH_DIRECTED_MAX_CMD = 60
CME_MIN_SPEED_KMS = 650.0    # see note above -- the real production system's
                              # own selectivity bar (800), refined for this
                              # downstream use via a local sweep
MAGNITUDE_MIN_VMAX = 300.0   # sanity bounds on estimate_earth_vmax's linear
MAGNITUDE_MAX_VMAX = 3000.0  # extrapolation -- fit from only 12 pairs

# CANDIDATE SOURCE OPTION ADDED 2026-09-17 (real, off-by-default experiment
# -- see accuracy_improvement_tracking.txt LOG for the full finding): the
# raw DONKI pull above (donki_cme_2026.json) has been this project's
# candidate source since day one, but CLEAR's own real, live production
# CME-selection output (synced to solsticedisk, pulled via the same SSH
# key CLEAR_web's refresh-local-data.sh uses) is a materially better
# candidate source -- already speed-filtered to CLEAR's own real >=800
# km/s selection rule, and carries a REAL per-event half_angle (from
# 2026-04-01 onward) that lets Earth-directedness be tested per-event
# (filter_earth_directed_variable_width) instead of one flat 60-degree
# cutoff for every CME regardless of its actual angular width. Kept
# OFF BY DEFAULT (candidate_source="donki") until validated -- see
# `--candidate-source production` and the LOO comparison this enables.
PRODUCTION_CME_JSON = "checking_cme_production.json"

# SECOND CANDIDATE-SOURCE OPTION, added 2026-09-17 same day as the one
# above, after that one traded away coverage for width and lost: fetched
# DONKI's own RAW API response directly (not this project's older
# normalized cache, which dropped halfAngle at build time) and confirmed
# halfAngle is present on 1239/1240 events -- i.e. DONKI's OWN dense
# ~1200-event pool, not the 71-event production list, PLUS real per-event
# width almost everywhere. This is the direct test of Experiment 1's own
# "promising untried follow-up": width-aware filtering WITHOUT giving up
# coverage. Still off by default (candidate_source="donki") until
# validated.
DONKI_RAW_JSON = "donki_raw_full.json"


def _bfield_bins(obs):
    return {"b": to_bin_map(obs["t"], obs["b"]),
            "bz": to_bin_map(obs["t"], obs["bz"]),
            "n": to_bin_map(obs["t"], obs["n"])}


def _donki_arrivals():
    return build_arrival_index(load_events("donki_cme_2026.json"), CME_MIN_SPEED_KMS,
                                max_cmd=EARTH_DIRECTED_MAX_CMD, include_unknown=True)


def _production_arrivals():
    """CLEAR's own real production CME-selection output as the candidate
    source, instead of the raw DONKI pull -- see PRODUCTION_CME_JSON note
    above. half_angle_aware=True uses each event's real angular width when
    known (2026-04-01 onward), falling back to the same flat
    EARTH_DIRECTED_MAX_CMD cutoff for older entries that predate the field.
    """
    events = load_production_cme_events(PRODUCTION_CME_JSON)
    return build_arrival_index(events, CME_MIN_SPEED_KMS, max_cmd=EARTH_DIRECTED_MAX_CMD,
                                include_unknown=True, half_angle_aware=True)


def _donki_width_arrivals():
    """DONKI's own dense candidate pool (same ~1200-event coverage as the
    locked default), but using load_donki_events_with_width() (the RAW API
    pull, with a real per-event half_angle) instead of the older
    normalized cache -- see DONKI_RAW_JSON note above.
    """
    events = load_donki_events_with_width(DONKI_RAW_JSON)
    return build_arrival_index(events, CME_MIN_SPEED_KMS, max_cmd=EARTH_DIRECTED_MAX_CMD,
                                include_unknown=True, half_angle_aware=True)


def _enlil_arrivals():
    """Real WSA-Enlil Earth-arrival predictions, straight from NASA's own
    operational model runs embedded in DONKI's raw API response -- NOT
    computed by any transit-time formula this project owns. See
    donki_features.load_enlil_earth_arrivals for the full rationale
    (2026-09-17, the "we have to figure this out alone" pivot). Applies
    the same speed floor as every other source for consistency, though
    the whole point here is real-model timing, not just speed/width.
    """
    arrivals = load_enlil_earth_arrivals(DONKI_RAW_JSON)
    return [(arrival_unix, speed, launch_unix) for arrival_unix, speed, launch_unix in arrivals
            if speed >= CME_MIN_SPEED_KMS]


def _arrivals_for(candidate_source):
    if candidate_source == "production":
        return _production_arrivals()
    if candidate_source == "donki_width":
        return _donki_width_arrivals()
    if candidate_source == "enlil":
        return _enlil_arrivals()
    return _donki_arrivals()


def _cme_candidates(bins, arrivals=None):
    """For every bin, (has_candidate, predicted_vmax) from the closest
    causally-available, fast (>=CME_MIN_SPEED_KMS), Earth-directed DONKI
    CME candidate -- see the Tier 0a note above -- using the original
    12-pair external calibration (estimate_earth_vmax). predicted_vmax is
    NaN where has_candidate is False. Kept for `--no-lasco-calibration`
    comparison; `_cme_candidates_lasco_hybrid` below is the locked default.
    """
    if arrivals is None:
        arrivals = _donki_arrivals()
    has_candidate = np.zeros(len(bins), dtype=bool)
    predicted_vmax = np.full(len(bins), np.nan)
    for i, b in enumerate(bins):
        offset_h, speed, has = arrival_feature(int(b) * BIN_S, arrivals)
        if has:
            has_candidate[i] = True
            predicted_vmax[i] = float(np.clip(estimate_earth_vmax(speed),
                                               MAGNITUDE_MIN_VMAX, MAGNITUDE_MAX_VMAX))
    return has_candidate, predicted_vmax


# LASCO-CALIBRATED MAGNITUDE SHIFT, PROMOTED 2026-09-16: replaces
# estimate_earth_vmax's 12-pair fit with a hybrid that fixes two separate
# problems found the same day (see accuracy_improvement_tracking.txt):
# (1) 12 pairs isn't enough to trust either coefficient -- fixed by using
# LASCO_MAGNITUDE_A/B (donki_features.py), fit on 303 real Richardson/
# Cane-verified pairs spanning 1996-2026 (~2.5 solar cycles), found via
# the Richardson/Cane catalog's own recorded LASCO CME timestamp column.
# (2) using that fit's absolute scale AS-IS on our own data is a clean
# NEGATIVE (disturbed corr 0.230->0.152) -- our own test window's
# disturbances run more severe than the general 1996-2026 population.
# Fix: keep LASCO's robust SLOPE (immune to collapsing flat, since it is
# fit on 25x the data) but LOO-shift only the INTERCEPT to this project's
# own local severity level, same per-event exclusion discipline as the
# classifier/flat-shift below. Result: disturbed corr 0.230->0.295, RMSE
# 80.1->75.2, per-event 24 wins/9 losses/4 ties, worst-case only +7.5 RMSE
# (half the risk of a full local-only recalibration that was tested the
# same day and held back for exactly this reason).
def _cme_candidates_lasco_hybrid(bins, events, arrivals=None):
    if arrivals is None:
        arrivals = _donki_arrivals()
    has_candidate = np.zeros(len(bins), dtype=bool)
    predicted_vmax = np.full(len(bins), np.nan)

    pairs = []
    for ei, (start_b, end_b, peak) in enumerate(events):
        t_onset = int(start_b) * BIN_S
        offset_h, speed, has = arrival_feature(t_onset, arrivals)
        if has:
            pairs.append((speed, peak, ei))
    if len(pairs) < 4:
        return has_candidate, predicted_vmax  # too few to safely LOO-shift; no candidates

    pairs_arr = np.array([(s, p) for s, p, _ in pairs])
    speeds, peaks = pairs_arr[:, 0], pairs_arr[:, 1]
    ev_idx = np.array([e for _, _, e in pairs])
    resid = peaks - np.array([estimate_earth_vmax_lasco(s) for s in speeds])

    def shift_excluding(event_i):
        mask = ev_idx != event_i
        if mask.sum() < 3:
            return None
        return float(resid[mask].mean())

    bin_event = np.full(len(bins), -1, dtype=int)
    for ei, (start_b, end_b, peak) in enumerate(events):
        mask = (bins >= start_b) & (bins <= end_b)
        bin_event[mask] = ei

    for i, b in enumerate(bins):
        offset_h, speed, has = arrival_feature(int(b) * BIN_S, arrivals)
        if not has:
            continue
        has_candidate[i] = True
        shift = shift_excluding(bin_event[i]) if bin_event[i] >= 0 else shift_excluding(-999)
        if shift is None:
            predicted_vmax[i] = np.nan
            has_candidate[i] = False
            continue
        pred = estimate_earth_vmax_lasco(speed) + shift
        predicted_vmax[i] = float(np.clip(pred, MAGNITUDE_MIN_VMAX, MAGNITUDE_MAX_VMAX))
    return has_candidate, predicted_vmax


# Found 2026-09-14, prompted by Jaxon looking at the live dashboard and
# noticing a moment where the correction overshot measured by ~80 km/s
# even though it wasn't a CME-magnitude case at all -- just the classifier
# reading "moderately elevated" (P~0.45) and applying part of the flat
# historical shift, while early_warning.py's OWN 6h-ahead forecast for
# that exact moment was already predicting a DECLINE, information the
# point-correction had no way to see (early_warning only fed the
# uncertainty BAND's OR-trigger before this, never the point estimate).
# Real, testable idea: gate the correction shift by whether early_warning
# agrees the situation is still escalating. Swept EW_GATE_FACTOR from 1.0
# (no gating, today's prior default) down to 0.0 (fully zero the shift
# whenever early_warning forecasts a decline) -- EVERY metric improved
# monotonically as the gate got stricter, with 0.0 best on every one
# simultaneously (not a trade-off like the CME-magnitude fix was).
# Checked per-event too (LOG "EARLY-WARNING GATE ADDED"): wins on 26/35
# held-out events, ties on 9 (never applies when early_warning has no
# usable forecast for that bin), LOSES ON ZERO. Locked at 0.0 (full gate).
EW_GATE_HOURS = 6.0    # matches early_warning.py's most-validated horizon
EW_GATE_FACTOR = 0.0   # multiply the shift by this when early_warning
                        # forecasts a decline; 1.0 = no gating at all


def _ew_rising_mask(bins, blend_all):
    """True where early_warning.py's own EW_GATE_HOURS-ahead forecast is
    AT OR ABOVE the current blend (still escalating or holding), False
    where it forecasts a decline. Defaults to True (don't gate) for bins
    early_warning can't produce a forecast for at all, so the gate only
    ever acts where there's a real, causally-available signal to act on.
    """
    ew_pred_by_bin = ew_predictions(EW_GATE_HOURS)
    ew_forecast = np.array([ew_pred_by_bin.get(int(b), np.nan) for b in bins])
    has_ew = ~np.isnan(ew_forecast)
    return np.where(has_ew, ew_forecast >= blend_all, True)


# QUIET-REGIME BIAS TERM, added 2026-09-24 (real, off-by-default experiment
# -- see accuracy_improvement_tracking.txt LOG). Found by Jaxon noticing a
# "no improvement here" moment on the live dashboard where BOTH raw and
# corrected were undershooting measured by 40-70 km/s for hours, obs never
# crossing DISTURBED_KMS (500) the whole time. Root cause: this pipeline's
# shift term is built ENTIRELY from disturbed-row residuals
# (flat_shift/SHIFT above) and scaled by P(disturbed)^POWER -- so a
# sustained, real, moderate-magnitude miss that stays under the 500 label
# and keeps P(disturbed) low gets essentially zero correction, regardless
# of how large or persistent the miss actually is. There has never been
# any quiet-regime bias correction in this pipeline at all.
# Design: add QUIET_BIAS = mean(obs-blend) over historically QUIET rows
# (same LOO-per-event-excluded discipline as flat_shift), blended out as
# proba rises so the already-locked, hard-won disturbed-period behavior at
# proba=1 is EXACTLY unchanged:
#   corrected(t) = blend(t) + QUIET_BIAS*(1 - P^POWER) + SHIFT(t)*P^POWER
# i.e. proba=0 (fully quiet) -> blend + QUIET_BIAS; proba=1 (fully
# disturbed) -> blend + SHIFT, identical to the locked default today.
# Off by default (`use_quiet_bias=False` / `--quiet-bias` to enable) until
# validated the same way every other lever here has been: aggregate AND
# per-event, confirming zero disturbed-period regression before promoting.
#
# TESTED 2026-09-24: a clean negative (see accuracy_improvement_tracking.txt
# LOG) -- root cause is that the quiet-regime residual (obs-blend) has
# essentially zero GLOBAL mean (-1.3 km/s over 17,130 quiet rows) despite
# swinging +-100+ km/s locally day to day (per-day window means range
# -181 to +109, std ~50). A single flat constant can't track something
# that isn't a persistent bias in the first place. Kept off by default;
# see QUIET_REGRESSION below for the follow-up this motivated.
#
# QUIET-REGIME REGRESSION, added 2026-09-24 (real, off-by-default
# experiment, the direct follow-up to QUIET_BIAS's negative result above).
# Since the quiet-regime miss is locally time-varying rather than a fixed
# bias, a flat number structurally can't help -- but the same trend/
# volatility/disagreement/bfield features that already predict
# P(disturbed) and drive early_warning.py's forecasts might predict the
# CURRENT residual's SIGN AND MAGNITUDE continuously, the quiet-regime
# equivalent of the disturbed-regime's CME-magnitude shift. Fit a ridge
# regression (p2.ridge_fit, same LOO-per-event exclusion discipline as
# everywhere else in this pipeline) on QUIET rows only, target =
# obs-blend, same FEATURES as the P(disturbed) classifier. Blended into
# the correction the same way as QUIET_BIAS -- at P(disturbed)=1 this is
# EXACTLY the locked default, unchanged:
#   corrected(t) = blend(t) + QR(t)*(1 - P^POWER) + SHIFT(t)*P^POWER
# where QR(t) is now a per-row regression prediction instead of one flat
# constant. `use_quiet_bias` and `use_quiet_regression` are mutually
# exclusive (regression wins if both given) -- see QUIET_REGRESSION_ALPHA
# below for the alpha, swept independently rather than borrowed from
# LOGISTIC_ALPHA, per this project's standing regularization-scale
# caution (burned twice before assuming an alpha transfers across a
# different objective/target).
#
# TESTED 2026-09-24, VERDICT: a real, decisive win on ordinary quiet-regime
# accuracy that comes at a genuinely bad cost to storm-time accuracy --
# NOT promoted. First look was misleading: aggregate "disturbed" RMSE/corr
# looked BETTER (75.8->73.8, corr 0.290->0.421), but that number is a trap
# -- checked per-event (standing discipline) and found only 7 wins / 27
# losses / 3 ties across 37 held-out storms. The aggregate win comes
# entirely from 4-5 of the most extreme storms improving a lot (e.g.
# +23, +31, +33 km/s RMSE), which dominate an RMSE average enough to hide
# that MOST individual storms got meaningfully worse. Tried gating the
# regression's influence off above various P(disturbed) thresholds
# (quiet_regression_max_proba) to stop it extrapolating onto disturbed-
# looking feature ranges it was never trained on -- even at the
# strictest tested gate (0.05, active only when P<0.05) it still made
# 11/37 storms worse; at a more typical gate (0.2) the sum of per-event
# losses (-143 km/s RMSE total) dwarfs the sum of wins (+11 km/s total).
# Root cause: the regression is trained overwhelmingly on ORDINARY quiet
# rows (the vast majority of quiet history), so it learns typical calm-
# period dynamics well, but that doesn't transfer to the STORM-APPROACH
# portion of an oncoming event, which is a real but rare and
# statistically distinct sub-population of "quiet-labeled" rows -- and
# that's exactly the population this project's early-warning goals care
# about most. Not deleting the code (kept off by default, correctly
# implemented) -- a real candidate follow-up, if ever revisited, would be
# training this regression on/near storm-approach windows specifically
# rather than the general quiet population, not just re-tuning alpha or
# the gate on the current design.
QUIET_REGRESSION_ALPHA = 5.0  # see accuracy_improvement_tracking.txt LOG,
                                # "QUIET-REGIME REGRESSION" for the sweep
                                # (alpha itself barely matters, 0.1-100 all
                                # near-identical -- large n/p ratio; the
                                # gate/scope issue above is the real story)


def _quiet_regression_fit(X_train, obs_train, blend_train, disturbed_mask_train, alpha):
    """Ridge fit of (obs-blend) on QUIET training rows only -- returns the
    (mean, std, coef, intercept) tuple p2.ridge_predict expects.
    """
    quiet = ~disturbed_mask_train
    target = (obs_train - blend_train)[quiet]
    return p2.ridge_fit(X_train[quiet], target, alpha)


# EARLY-WARNING AS A THIRD BLEND CANDIDATE, added 2026-10-01 (real,
# off-by-default experiment -- see accuracy_improvement_tracking.txt LOG).
# Found by Jaxon pushing back on a sustained multi-day "corrected worse
# than raw" streak (Sep 28 - Oct 1): checked early_warning.py's own 6h-
# ahead backtest series (the "ew" chart line, already computed by
# export_dashboard_data.py for display, never fed back into the
# correction itself) against the live stuck period -- MAE 9.3 km/s vs
# raw's 53.7 and corrected's 55.6 over the same 72h. Checked honestly
# whether that's real signal or the "same-time prediction is trivially
# accurate" trap this project has been burned by before (Section 2):
# naive "assume no change over 6h" ALSO scores a deceptively high 0.928
# correlation / 36.1 RMSE in aggregate (solar wind is highly
# autocorrelated at 6h -- most of the time nothing dramatic happens), so
# the aggregate number alone proves little. But early_warning's own
# FEATURES are model trend + real-time B-field/density -- never the
# observed SPEED itself -- so it is not simply "peeking" at a recent
# measurement, and its advantage during exactly this stuck-decline case
# (where blend's two existing inputs, raw and persist27, are BOTH blind
# to the live trend by construction) is real and physically distinct.
# Design: generalize the ALREADY-LOCKED adaptive blend weighting
# (phase2_model._adaptive_blend_weight, "the single biggest improvement
# found in this project") from 2 candidates to 3 -- same inverse-trailing-
# error weighting, same causal-only trailing window, just one more
# candidate. Falls back EXACTLY to the existing 2-way blend wherever
# early_warning's backtest is unavailable (dataset edges; the live bin
# itself before its own forecast exists) -- zero risk at that boundary,
# same shape as every other experiment in this pipeline.
# EW_BLEND_WINDOW_HOURS is longer than the 2-way blend's 3h window
# because early_warning's own 6h lead time means a forecast issued now
# won't have a KNOWN outcome to score until 6h from now -- the window
# needs enough historical depth to actually contain resolved residuals.
EW_BLEND_WINDOW_HOURS = 24.0  # swept, see LOG -- flat/stable across 12-48h
EW_BLEND_MIN_SAMPLES = 6

# EW-BLEND HYBRID GATE, added 2026-10-01 -- direct response to the per-event
# win/loss breakdown showing EW-blend is a clean, well-validated win on
# QUIET rows (where this whole thing was motivated from) but a MIXED bag
# on disturbed/storm events: 21 wins/16 losses vs the locked pipeline, and
# a real loss to plain raw on two short/weak onsets (events #32/#33, see
# accuracy_improvement_tracking.txt LOG).
#
# First attempt -- gate purely on predicted P(disturbed) < threshold --
# was a dead end: a single proba cutoff tight enough to protect every
# storm (<0.03) ALSO blocked EW-blend during the exact flagged episode
# that started this (its own held-out proba runs elevated during the
# decline, since the model's rising trend/volatility/disagreement looks
# statistically similar to genuine onset features even though real obs
# never crosses DISTURBED_KMS). Predicted P(disturbed) does not cleanly
# separate "genuine ambiguous quiet decline" (where EW-blend helps) from
# "quiet lead-up to a real fast onset" (where it doesn't).
#
# Working design: THREE combined, causally-available conditions, applying
# EW-blend only when ALL hold:
#   1. ground-truth-quiet (obs_all <= DISTURBED_KMS at LOO/backtest time;
#      the just-measured live obs value at live-serving time -- NOT a
#      future value, always already known when this runs)
#   2. no already-known inbound DONKI CME candidate (has_candidate==False
#      -- the locked CME-magnitude shift already handles storms with a
#      known candidate well; EW-blend only adds noise on top there)
#   3. held-out P(disturbed) < EW_BLEND_PROBA_GATE (still screens out the
#      genuinely onset-like quiet-but-about-to-turn cases within (1))
# Validated 2026-10-01: with EW_BLEND_PROBA_GATE=0.1, this combination
# keeps the FULL flagged-episode fix (RMSE 67.0->22.4, unchanged from the
# ungated result) while reducing the 37-storm per-event breakdown to a
# single borderline loss (-0.5 km/s RMSE, barely over the 0.5 km/s tie
# tolerance already used throughout this project's per-event scripts) --
# down from 16 losses ungated. Pushing the proba cap any tighter (0.08)
# does reach literal zero per-event losses, but at a steep cost: the
# flagged episode's own fix degrades from 22.4 to 35.6 RMSE, since a
# meaningful fraction of ITS rows sit in the 0.08-0.1 proba band. 0.1 is
# the deliberate choice: keep the real fix, accept one tiny, disclosed
# exception, rather than chase literal zero-loss at the cost of the thing
# that motivated this work. Only ~7.7% of all rows ever get touched.
# use_ew_blend_quiet_gate / use_ew_blend_candidate_gate (bool, both
# default False for backward compatibility) turn on conditions 1 and 2;
# EW_BLEND_PROBA_GATE (None = off) is condition 3. All three are
# independent toggles -- the validated hybrid above is all three combined.
EW_BLEND_PROBA_GATE = None


def _held_out_disturbed_proba(bins, X_all, disturbed_mask_all, events, alpha=LOGISTIC_ALPHA):
    """Same leave-one-event-out P(disturbed) the main correction loop
    computes internally (see loo_corrected_series), exposed standalone so
    EW-blend's proba-gate can decide where to apply the 3-way blend
    BEFORE blend_all is finalized -- the main loop's own per-fold
    classifier fit happens too late for that, since it needs the
    already-finalized blend_all to compute flat_shift. Deliberately only
    called when gating is actually requested (ew_blend_proba_gate is not
    None), so the default/locked path pays zero extra cost for this.
    """
    n = len(bins)
    proba = np.full(n, np.nan)
    held_out_mask = np.zeros(n, dtype=bool)
    for start_b, end_b, peak in events:
        test_mask = (bins >= start_b) & (bins <= end_b)
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        mean_c, std_c, w, b = logistic_fit(X_all[train_mask], disturbed_mask_all[train_mask].astype(float), alpha)
        proba[test_mask] = predict_proba(X_all[test_mask], mean_c, std_c, w, b)
        held_out_mask |= test_mask
    mean_c, std_c, w, b = logistic_fit(X_all, disturbed_mask_all.astype(float), alpha)
    non_event = ~held_out_mask
    proba[non_event] = predict_proba(X_all[non_event], mean_c, std_c, w, b)
    return proba


def _trailing_mae_nan_aware(errs, window_bins, min_samples):
    """Causal trailing mean absolute error, one entry per row, treating
    NaN rows (early_warning had no resolved backtest value) as absent
    from both the running sum and the sample count -- not as zero error.
    Returns (trailing_mae, n_samples) arrays; trailing_mae is NaN wherever
    n_samples < min_samples.
    """
    n = len(errs)
    valid = ~np.isnan(errs)
    errs_filled = np.where(valid, errs, 0.0)
    cs_err = np.concatenate([[0.0], np.cumsum(errs_filled)])
    cs_cnt = np.concatenate([[0], np.cumsum(valid.astype(int))])
    trail_mae = np.full(n, np.nan)
    n_samples = np.zeros(n, dtype=int)
    for i in range(n):
        lo = max(0, i - window_bins)
        cnt = cs_cnt[i] - cs_cnt[lo]
        n_samples[i] = cnt
        if cnt >= min_samples:
            trail_mae[i] = (cs_err[i] - cs_err[lo]) / cnt
    return trail_mae, n_samples


def _three_way_adaptive_blend(raw_v, persist_v, ew_v, obs_v, blend_2way,
                               window_hours=EW_BLEND_WINDOW_HOURS,
                               min_samples=EW_BLEND_MIN_SAMPLES):
    """blend_2way (the already-computed, already-locked raw/persist27
    blend) wherever early_warning's backtest is unavailable for THIS row
    or doesn't have enough trailing history yet; a three-way inverse-
    trailing-error-weighted blend of {raw, persist27, early_warning}
    wherever it is. Fully causal: every trailing window only looks at
    rows strictly before the current one.
    """
    window_bins = round(window_hours * 3600 / BIN_S)
    raw_err = np.abs(raw_v - obs_v)
    persist_err = np.abs(persist_v - obs_v)
    ew_err = np.where(np.isnan(ew_v), np.nan, np.abs(ew_v - obs_v))

    trail_raw, _ = _trailing_mae_nan_aware(raw_err, window_bins, min_samples)
    trail_persist, _ = _trailing_mae_nan_aware(persist_err, window_bins, min_samples)
    trail_ew, _ = _trailing_mae_nan_aware(ew_err, window_bins, min_samples)

    have_ew = ~np.isnan(ew_v) & ~np.isnan(trail_ew) & ~np.isnan(trail_raw) & ~np.isnan(trail_persist)

    inv_raw = 1.0 / (trail_raw + 1e-9)
    inv_persist = 1.0 / (trail_persist + 1e-9)
    inv_ew = 1.0 / (trail_ew + 1e-9)
    denom = inv_raw + inv_persist + inv_ew
    w_raw = np.where(have_ew, inv_raw / denom, np.nan)
    w_persist = np.where(have_ew, inv_persist / denom, np.nan)
    w_ew = np.where(have_ew, inv_ew / denom, np.nan)

    blend3 = np.where(
        have_ew,
        w_raw * raw_v + w_persist * persist_v + w_ew * np.where(np.isnan(ew_v), 0.0, ew_v),
        blend_2way,
    )
    return blend3


def loo_corrected_series(bins, X_all, obs_all, blend_all, disturbed_mask_all, events,
                          use_cme_magnitude=True, use_ew_gate=True, use_lasco_calibration=True,
                          arrivals=None, use_quiet_bias=False, use_quiet_regression=False,
                          quiet_regression_alpha=QUIET_REGRESSION_ALPHA,
                          quiet_regression_max_proba=1.0,
                          use_ew_blend=False, raw_all=None, persist_all=None, ew_backtest_all=None,
                          ew_blend_window_hours=EW_BLEND_WINDOW_HOURS,
                          ew_blend_proba_gate=EW_BLEND_PROBA_GATE,
                          use_ew_blend_quiet_gate=False, use_ew_blend_candidate_gate=False):
    """THE single shared implementation of the locked correction's
    leave-one-event-out series -- returns (full_pred, full_proba,
    has_candidate, ew_gate). Extracted 2026-09-14 after this exact fold
    loop was found independently duplicated in THREE places (this
    function's old inline body, combined_model.py's validate(), and
    export_dashboard_data.py's build()) and had already silently drifted
    out of sync twice in one day (see LOG "SECOND DUPLICATION BUG FOUND").
    Every caller that needs this pipeline's LOO-corrected series MUST call
    this function, not re-derive it -- that is the actual fix, not just
    patching each copy again.

    `arrivals`: optional pre-built candidate-arrival index (see
    _arrivals_for / --candidate-source, added 2026-09-17); defaults to
    the locked DONKI source when omitted, computed internally by
    _cme_candidates()/_cme_candidates_lasco_hybrid() exactly as before.
    """
    has_candidate = np.zeros(len(bins), dtype=bool)
    predicted_vmax = np.full(len(bins), np.nan)
    if use_cme_magnitude:
        if use_lasco_calibration:
            has_candidate, predicted_vmax = _cme_candidates_lasco_hybrid(bins, events, arrivals=arrivals)
        else:
            has_candidate, predicted_vmax = _cme_candidates(bins, arrivals=arrivals)

    ew_gate = (np.where(_ew_rising_mask(bins, blend_all), 1.0, EW_GATE_FACTOR)
               if use_ew_gate else np.ones(len(bins)))

    # EW-BLEND, added 2026-10-01 -- computed against the ORIGINAL 2-way
    # blend_all as reference/fallback, same "compute the gate signal
    # first, reassign blend after" ordering already used for every other
    # blend-touching experiment here, so there's no circularity.
    if use_ew_blend:
        blend3 = _three_way_adaptive_blend(
            raw_all, persist_all, ew_backtest_all, obs_all, blend_all,
            window_hours=ew_blend_window_hours)
        any_gate = (ew_blend_proba_gate is not None or use_ew_blend_quiet_gate
                    or use_ew_blend_candidate_gate)
        if not any_gate:
            blend_all = blend3
        else:
            gate_mask = np.ones(len(bins), dtype=bool)
            if use_ew_blend_quiet_gate:
                gate_mask &= ~disturbed_mask_all
            if use_ew_blend_candidate_gate:
                gate_mask &= ~has_candidate
            if ew_blend_proba_gate is not None:
                gate_proba = _held_out_disturbed_proba(bins, X_all, disturbed_mask_all, events)
                gate_mask &= gate_proba < ew_blend_proba_gate
            blend_all = np.where(gate_mask, blend3, blend_all)

    full_pred = blend_all.copy()
    full_proba = np.full(len(bins), np.nan)
    held_out_mask = np.zeros(len(bins), dtype=bool)
    for start_b, end_b, peak in events:
        test_mask = (bins >= start_b) & (bins <= end_b)
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        mean_c, std_c, w, b = logistic_fit(
            X_all[train_mask], disturbed_mask_all[train_mask].astype(float), LOGISTIC_ALPHA)
        flat_shift = float((obs_all[train_mask] - blend_all[train_mask])[disturbed_mask_all[train_mask]].mean())
        if use_quiet_regression:
            qr_mean, qr_std, qr_coef, qr_intercept = _quiet_regression_fit(
                X_all[train_mask], obs_all[train_mask], blend_all[train_mask],
                disturbed_mask_all[train_mask], quiet_regression_alpha)
            local_bias = p2.ridge_predict(X_all[test_mask], qr_mean, qr_std, qr_coef, qr_intercept)
        elif use_quiet_bias:
            local_bias = float((obs_all[train_mask] - blend_all[train_mask])[~disturbed_mask_all[train_mask]].mean())
        else:
            local_bias = 0.0
        proba = predict_proba(X_all[test_mask], mean_c, std_c, w, b)
        shift = np.where(has_candidate[test_mask],
                          predicted_vmax[test_mask] - blend_all[test_mask],
                          flat_shift) * ew_gate[test_mask]
        power_proba = proba ** POWER
        local_weight = (1.0 - power_proba) * (proba < quiet_regression_max_proba) \
            if use_quiet_regression else (1.0 - power_proba)
        full_pred[test_mask] = blend_all[test_mask] + local_bias * local_weight + power_proba * shift
        full_proba[test_mask] = proba
        held_out_mask |= test_mask

    mean_c, std_c, w, b = logistic_fit(X_all, disturbed_mask_all.astype(float), LOGISTIC_ALPHA)
    flat_shift_all = float((obs_all - blend_all)[disturbed_mask_all].mean())
    non_event_mask = ~held_out_mask
    if use_quiet_regression:
        qr_mean, qr_std, qr_coef, qr_intercept = _quiet_regression_fit(
            X_all, obs_all, blend_all, disturbed_mask_all, quiet_regression_alpha)
        local_bias_all = p2.ridge_predict(X_all[non_event_mask], qr_mean, qr_std, qr_coef, qr_intercept)
    elif use_quiet_bias:
        local_bias_all = float((obs_all - blend_all)[~disturbed_mask_all].mean())
    else:
        local_bias_all = 0.0
    proba_all = predict_proba(X_all, mean_c, std_c, w, b)
    shift_all = np.where(has_candidate[non_event_mask],
                          predicted_vmax[non_event_mask] - blend_all[non_event_mask],
                          flat_shift_all) * ew_gate[non_event_mask]
    power_proba_all = proba_all[non_event_mask] ** POWER
    local_weight_all = (1.0 - power_proba_all) * (proba_all[non_event_mask] < quiet_regression_max_proba) \
        if use_quiet_regression else (1.0 - power_proba_all)
    full_pred[non_event_mask] = (blend_all[non_event_mask]
                                  + local_bias_all * local_weight_all
                                  + power_proba_all * shift_all)
    full_proba[non_event_mask] = proba_all[non_event_mask]

    return full_pred, full_proba, has_candidate, ew_gate


def loo_validate(use_cme_magnitude=True, use_ew_gate=True, use_lasco_calibration=True,
                  candidate_source="donki", use_quiet_bias=False, use_quiet_regression=False,
                  quiet_regression_alpha=QUIET_REGRESSION_ALPHA,
                  use_ew_blend=False, ew_blend_window_hours=EW_BLEND_WINDOW_HOURS,
                  ew_blend_proba_gate=EW_BLEND_PROBA_GATE,
                  use_ew_blend_quiet_gate=False, use_ew_blend_candidate_gate=False):
    p2.FEATURES = FEATURES
    doc = fetch()
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])
    events = p2.event_windows(obs_bins, threshold=600.0)

    bins, data = p2.build_dataset(model_bins, obs_bins, bfield_bins=_bfield_bins(obs))
    X_all = p2.feature_matrix(data)
    obs_all = data["obs_v"]
    blend_all = data["blend_v"]
    raw_all = data["model_v"]
    persist_all = data["persist_v"]
    disturbed_mask_all = obs_all > DISTURBED_KMS

    ew_backtest_all = None
    if use_ew_blend:
        ew_pred_by_bin = ew_predictions(6.0)
        lead_bins = round(6.0 * 3600 / BIN_S)
        ew_backtest_all = np.array([ew_pred_by_bin.get(int(b) - lead_bins, np.nan) for b in bins])

    arrivals = _arrivals_for(candidate_source) if use_cme_magnitude else None
    full_pred, full_proba, has_candidate, ew_gate = loo_corrected_series(
        bins, X_all, obs_all, blend_all, disturbed_mask_all, events,
        use_cme_magnitude, use_ew_gate, use_lasco_calibration, arrivals=arrivals,
        use_quiet_bias=use_quiet_bias, use_quiet_regression=use_quiet_regression,
        quiet_regression_alpha=quiet_regression_alpha,
        use_ew_blend=use_ew_blend, raw_all=raw_all, persist_all=persist_all,
        ew_backtest_all=ew_backtest_all, ew_blend_window_hours=ew_blend_window_hours,
        ew_blend_proba_gate=ew_blend_proba_gate,
        use_ew_blend_quiet_gate=use_ew_blend_quiet_gate,
        use_ew_blend_candidate_gate=use_ew_blend_candidate_gate)

    s_all = p2.stats(obs_all, full_pred)
    s_dist = p2.stats(obs_all[disturbed_mask_all], full_pred[disturbed_mask_all])
    s_quiet = p2.stats(obs_all[~disturbed_mask_all], full_pred[~disturbed_mask_all])
    s_blend_all = p2.stats(obs_all, blend_all)
    s_blend_dist = p2.stats(obs_all[disturbed_mask_all], blend_all[disturbed_mask_all])
    s_blend_quiet = p2.stats(obs_all[~disturbed_mask_all], blend_all[~disturbed_mask_all])

    mode = "CME-magnitude-aware shift (locked default, 2026-09-14)" if use_cme_magnitude else "flat shift only (pre-2026-09-14 behavior)"
    calib_desc = "LASCO-slope+local-intercept (locked default, 2026-09-16)" if use_lasco_calibration else "12-pair external calibration (pre-2026-09-16)"
    gate_desc = "ON (locked default)" if use_ew_gate else "OFF"
    if use_quiet_regression:
        qb_desc = f"REGRESSION (experimental, 2026-09-24, alpha={quiet_regression_alpha:g})"
    elif use_quiet_bias:
        qb_desc = "FLAT (experimental, 2026-09-24, tested negative)"
    else:
        qb_desc = "OFF (locked default)"
    if use_ew_blend:
        gate_bits = []
        if use_ew_blend_quiet_gate:
            gate_bits.append("quiet-only")
        if use_ew_blend_candidate_gate:
            gate_bits.append("no-known-CME")
        if ew_blend_proba_gate is not None:
            gate_bits.append(f"proba<{ew_blend_proba_gate:g}")
        gate_str = (", gated: " + "+".join(gate_bits)) if gate_bits else ", no gate (applies wherever available)"
        ewb_desc = f"ON (experimental, 2026-10-01, window={ew_blend_window_hours:g}h{gate_str})"
    else:
        ewb_desc = "OFF (locked default)"
    print(f"LOCKED pipeline (power={POWER}, alpha={LOGISTIC_ALPHA}), mode={mode}, "
          f"early-warning gate={gate_desc}, quiet-regime correction={qb_desc}, "
          f"EW-blend={ewb_desc}, leave-one-event-out validation:")
    if use_cme_magnitude:
        print(f"  magnitude calibration: {calib_desc}")
        print(f"  candidate source: {candidate_source}"
              + ("  (real production selection, half_angle-aware -- EXPERIMENTAL, "
                 "off by default)" if candidate_source == "production" else " (locked default)"))
        n_dist_with_candidate = int((has_candidate & disturbed_mask_all).sum())
        print(f"  {n_dist_with_candidate}/{int(disturbed_mask_all.sum())} disturbed rows had an "
              f"Earth-directed DONKI candidate within the causal arrival window "
              f"({int(has_candidate.sum())}/{len(bins)} rows overall)")
    if use_ew_gate:
        n_gated = int((ew_gate < 1.0).sum())
        print(f"  {n_gated}/{len(bins)} rows had their shift gated to {EW_GATE_FACTOR:g}x "
              f"(early_warning forecasts a decline there)")
    print()
    print(f"{'':20s}{'RMSE':>10s}{'MAE':>10s}{'bias':>10s}{'corr':>10s}")
    print(f"{'blend, all':20s}{s_blend_all['rmse']:10.1f}{s_blend_all['mae']:10.1f}"
          f"{s_blend_all['bias']:10.1f}{s_blend_all['corr']:10.3f}")
    print(f"{'corrected, all':20s}{s_all['rmse']:10.1f}{s_all['mae']:10.1f}"
          f"{s_all['bias']:10.1f}{s_all['corr']:10.3f}")
    print()
    print(f"{'blend, disturbed':20s}{s_blend_dist['rmse']:10.1f}{s_blend_dist['mae']:10.1f}"
          f"{s_blend_dist['bias']:10.1f}{s_blend_dist['corr']:10.3f}")
    print(f"{'corrected, disturbed':20s}{s_dist['rmse']:10.1f}{s_dist['mae']:10.1f}"
          f"{s_dist['bias']:10.1f}{s_dist['corr']:10.3f}")
    print()
    print(f"{'blend, quiet':20s}{s_blend_quiet['rmse']:10.1f}{s_blend_quiet['mae']:10.1f}"
          f"{s_blend_quiet['bias']:10.1f}{s_blend_quiet['corr']:10.3f}")
    print(f"{'corrected, quiet':20s}{s_quiet['rmse']:10.1f}{s_quiet['mae']:10.1f}"
          f"{s_quiet['bias']:10.1f}{s_quiet['corr']:10.3f}")


def apply_to_live(use_cme_magnitude=True, use_ew_gate=True, use_lasco_calibration=True,
                   candidate_source="donki", use_quiet_bias=False, use_quiet_regression=False,
                   quiet_regression_alpha=QUIET_REGRESSION_ALPHA,
                   use_ew_blend=False, ew_blend_window_hours=EW_BLEND_WINDOW_HOURS,
                   ew_blend_proba_gate=EW_BLEND_PROBA_GATE,
                   use_ew_blend_quiet_gate=False, use_ew_blend_candidate_gate=False):
    p2.FEATURES = FEATURES
    doc = fetch()
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])
    bfield = _bfield_bins(obs)

    bins, data = p2.build_dataset(model_bins, obs_bins, bfield_bins=bfield)
    X_all = p2.feature_matrix(data)
    obs_all = data["obs_v"]
    blend_all = data["blend_v"]
    disturbed_mask_all = obs_all > DISTURBED_KMS

    mean_c, std_c, w, b = logistic_fit(X_all, disturbed_mask_all.astype(float), LOGISTIC_ALPHA)
    flat_shift = float((obs_all - blend_all)[disturbed_mask_all].mean())

    b0, live_row = p2.live_features(model_bins, obs_bins, bfield_bins=bfield)
    mv = live_row["model_v"]
    blend_v = live_row["_blend_v"]
    x_live = p2.live_feature_vector(live_row, FEATURES)
    proba_live = predict_proba(x_live, mean_c, std_c, w, b)[0]

    # EW-BLEND live path, added 2026-10-01. CRITICAL DISTINCTION from the
    # EW_GATE bug fixed 2026-09-28: that bug needed a forecast ISSUED NOW
    # (target 6h in the FUTURE -- structurally never resolved, hence
    # always missing from the historical dict). This needs the opposite
    # direction: the forecast ISSUED 6h AGO, whose TARGET is right now --
    # and "right now" typically already has a real obs reading by the
    # time this runs (obs lag is usually far shorter than the model's own
    # publish lag -- confirmed directly 2026-09-30: obs ~16min vs model
    # ~2h40min), so ew_predictions()'s historical dict genuinely can
    # contain this exact entry. Still never assumed blind: falls back to
    # the existing 2-way blend_v untouched whenever this specific lookup,
    # or any of the three trailing-error windows, comes up short -- same
    # zero-risk-at-the-boundary shape as every other experiment here.
    ew_blend_applied = False
    if use_ew_blend:
        rot_bins = round(27 * 86400 / BIN_S)
        lead_bins_blend = round(6.0 * 3600 / BIN_S)
        window_bins_blend = round(ew_blend_window_hours * 3600 / BIN_S)
        ew_pred_by_bin_blend = ew_predictions(6.0)
        ew_v_for_now = ew_pred_by_bin_blend.get(b0 - lead_bins_blend)

        persist_v_live = live_row["_persist_v"]
        raw_errs, persist_errs, ew_errs = [], [], []
        for bb in range(b0 - window_bins_blend, b0):
            if bb in model_bins and bb in obs_bins:
                raw_errs.append(abs(model_bins[bb] - obs_bins[bb]))
            pb = bb - rot_bins
            if pb in obs_bins and bb in obs_bins:
                persist_errs.append(abs(obs_bins[pb] - obs_bins[bb]))
            ew_val = ew_pred_by_bin_blend.get(bb - lead_bins_blend)
            if ew_val is not None and bb in obs_bins:
                ew_errs.append(abs(ew_val - obs_bins[bb]))

        # HYBRID GATE, added 2026-10-01 -- see EW_BLEND_PROBA_GATE's
        # module-level comment for the full validated design (proba alone
        # wasn't enough; needs quiet-now + no-known-candidate too). All
        # three checks use only already-known-at-this-moment information:
        # proba_live from the existing classifier call above, obs_now from
        # the just-measured live observation (never a future value), and
        # candidate_now from the same DONKI arrival lookup the main shift
        # block below uses (computed here too, since this block runs
        # first; a second cheap arrival_feature() call, not a behavior
        # change to that block).
        gate_ok = ew_blend_proba_gate is None or proba_live < ew_blend_proba_gate
        if gate_ok and use_ew_blend_quiet_gate:
            obs_now = obs_bins.get(b0)
            gate_ok = obs_now is not None and obs_now <= DISTURBED_KMS
        if gate_ok and use_ew_blend_candidate_gate:
            _, _, candidate_now = arrival_feature(b0 * BIN_S, _arrivals_for(candidate_source))
            gate_ok = not candidate_now
        if (gate_ok and ew_v_for_now is not None
                and len(raw_errs) >= EW_BLEND_MIN_SAMPLES
                and len(persist_errs) >= EW_BLEND_MIN_SAMPLES
                and len(ew_errs) >= EW_BLEND_MIN_SAMPLES):
            trail_raw = sum(raw_errs) / len(raw_errs)
            trail_persist = sum(persist_errs) / len(persist_errs)
            trail_ew = sum(ew_errs) / len(ew_errs)
            inv_raw, inv_persist, inv_ew = (1.0 / (trail_raw + 1e-9),
                                             1.0 / (trail_persist + 1e-9),
                                             1.0 / (trail_ew + 1e-9))
            denom = inv_raw + inv_persist + inv_ew
            w_raw3, w_persist3, w_ew3 = inv_raw / denom, inv_persist / denom, inv_ew / denom
            blend_v = w_raw3 * mv + w_persist3 * persist_v_live + w_ew3 * ew_v_for_now
            ew_blend_applied = True

    if use_quiet_regression:
        qr_mean, qr_std, qr_coef, qr_intercept = _quiet_regression_fit(
            X_all, obs_all, blend_all, disturbed_mask_all, quiet_regression_alpha)
        local_bias = float(p2.ridge_predict(x_live, qr_mean, qr_std, qr_coef, qr_intercept)[0])
    elif use_quiet_bias:
        local_bias = float((obs_all - blend_all)[~disturbed_mask_all].mean())
    else:
        local_bias = 0.0

    cme_candidate = None
    shift = flat_shift
    if use_cme_magnitude:
        # Same candidate filter as loo_validate()/_cme_candidates() (fast,
        # >=CME_MIN_SPEED_KMS, Earth-directed). This used to be a separately
        # duplicated block that never got the speed floor applied when it
        # was added, silently treating any-speed CMEs as candidates at
        # live-serving time despite the LOO validation (and its locked
        # numbers) requiring >=650 km/s throughout -- caught by actually
        # running --live and noticing a 207 km/s "candidate," not assumed
        # correct. Now calls the same _arrivals_for() shared builder every
        # other caller uses (2026-09-17), rather than its own inline block,
        # so a candidate-source change can't drift out of sync here again.
        arrivals = _arrivals_for(candidate_source)
        offset_h, speed, has = arrival_feature(b0 * BIN_S, arrivals)
        if has:
            if use_lasco_calibration:
                # Live serving has no held-out event to exclude (this is a
                # genuinely new point, not a historical test row), so the
                # intercept shift uses ALL matched historical events --
                # same convention as flat_shift above using all disturbed
                # rows, not a LOO fold.
                events = p2.event_windows(obs_bins, threshold=600.0)
                hist_pairs = []
                for start_b, end_b, peak in events:
                    oh, sp, hh = arrival_feature(int(start_b) * BIN_S, arrivals)
                    if hh:
                        hist_pairs.append((sp, peak))
                if len(hist_pairs) >= 3:
                    hist_arr = np.array(hist_pairs)
                    intercept_shift = float((hist_arr[:, 1]
                                             - estimate_earth_vmax_lasco(hist_arr[:, 0])).mean())
                else:
                    intercept_shift = 0.0
                predicted_vmax = float(np.clip(estimate_earth_vmax_lasco(speed) + intercept_shift,
                                                MAGNITUDE_MIN_VMAX, MAGNITUDE_MAX_VMAX))
            else:
                predicted_vmax = float(np.clip(estimate_earth_vmax(speed),
                                                MAGNITUDE_MIN_VMAX, MAGNITUDE_MAX_VMAX))
            shift = predicted_vmax - blend_v
            cme_candidate = {"offset_h": offset_h, "donki_speed": speed,
                              "predicted_vmax": predicted_vmax}

    # BUG FIXED 2026-09-28: this used to check the gate via
    # _ew_rising_mask()/ew_predictions() -- the HISTORICAL LOO-backtest
    # dict, keyed only by bins that already have a KNOWN FUTURE outcome
    # (early_warning._dataset() filters out any bin without one). The live
    # bin b0 is, by definition, never in that dict -- its future hasn't
    # happened yet -- so ew_pred_by_bin.get(b0) was ALWAYS None here, and
    # the gate ALWAYS defaulted to "rising, don't gate" at live-serving
    # time, no matter what early_warning's real live forecast said. Caught
    # by Jaxon's live "no improvement here" screenshot (2026-09-28,
    # 336 measured vs 351 corrected while raw was 340): early_warning.
    # apply_to_live(6.0) genuinely forecast a DECLINE for that exact
    # moment (335.8 -> 331.5) that should have gated the shift to zero,
    # but never got the chance to. This is a real, previously-unknown
    # defect in the EW_GATE mechanism's live path specifically -- the
    # LOO-validated numbers (loo_validate/loo_corrected_series, which use
    # _ew_rising_mask correctly on HISTORICAL bins that do have known
    # futures) were never wrong, but every live snapshot this whole
    # project has served was silently missing the gate's protection.
    # Fixed by calling early_warning.apply_to_live() directly for a
    # genuine live forecast instead of looking it up in a dict that
    # structurally can't contain the live bin.
    ew_forecast_v = None
    gated = False
    if use_ew_gate:
        live_forecast = ew.apply_to_live(EW_GATE_HOURS)
        ew_forecast_v = live_forecast["forecast"]
        is_rising = ew_forecast_v >= blend_v
        if not is_rising:
            shift = shift * EW_GATE_FACTOR
            gated = True

    power_proba_live = proba_live ** POWER
    corrected_v = blend_v + local_bias * (1.0 - power_proba_live) + power_proba_live * shift

    when = datetime.datetime.utcfromtimestamp(b0 * BIN_S)
    print(f"Live point: {when} UTC")
    print(f"  raw model:        {mv:.1f} km/s")
    print(f"  blend:            {blend_v:.1f} km/s"
          + (" (EW-blend applied)" if ew_blend_applied else
             "  (EW-blend requested but unavailable this moment -- 2-way blend used)" if use_ew_blend else ""))
    print(f"  P(disturbed):     {proba_live:.3f}")
    if use_quiet_regression:
        print(f"  quiet-regime regression: {local_bias:+.1f} km/s (blended out as P(disturbed) rises)")
    elif use_quiet_bias:
        print(f"  quiet-regime bias: {local_bias:+.1f} km/s (blended out as P(disturbed) rises)")
    if use_cme_magnitude:
        if cme_candidate:
            print(f"  CME candidate:    DONKI speed {cme_candidate['donki_speed']:.0f} km/s, "
                  f"offset {cme_candidate['offset_h']:+.1f}h, predicted Earth vmax "
                  f"{cme_candidate['predicted_vmax']:.0f} km/s")
        else:
            print(f"  CME candidate:    none within window -- using flat shift ({flat_shift:.1f})")
    if use_ew_gate and ew_forecast_v is not None:
        print(f"  early-warning({EW_GATE_HOURS:g}h): {ew_forecast_v:.1f} km/s vs blend {blend_v:.1f} -- "
              f"{'GATED shift to ' + format(EW_GATE_FACTOR, 'g') + 'x (forecasts decline)' if gated else 'rising/holding, shift unchanged'}")
    print(f"  corrected:        {corrected_v:.1f} km/s")
    return {"when": when, "raw_model": mv, "blend": blend_v,
            "proba_disturbed": proba_live, "corrected": corrected_v,
            "cme_candidate": cme_candidate, "ew_gated": gated}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                     help="apply the locked pipeline to the current live point "
                          "instead of running leave-one-event-out validation")
    ap.add_argument("--no-cme-magnitude", action="store_true",
                     help="revert to the pre-2026-09-14 behavior: flat "
                          "historical shift for every disturbed row, no "
                          "per-event CME-magnitude scaling. For comparison "
                          "only -- CME-magnitude scaling is now the locked "
                          "default (see module docstring, 'CME-MAGNITUDE "
                          "SHIFT PROMOTED')")
    ap.add_argument("--no-ew-gate", action="store_true",
                     help="disable the early-warning gate (zeroing the "
                          "correction shift when early_warning.py forecasts "
                          "a decline) -- on by default, see module docstring "
                          "'EARLY-WARNING GATE ADDED'")
    ap.add_argument("--no-lasco-calibration", action="store_true",
                     help="revert the magnitude formula to the pre-2026-09-16 "
                          "12-pair external calibration (estimate_earth_vmax), "
                          "instead of the LASCO-slope + local-intercept hybrid "
                          "(303-pair, 1996-2026 external calibration) that is "
                          "now the locked default. For comparison only.")
    ap.add_argument("--candidate-source", choices=("donki", "production", "donki_width", "enlil"),
                     default="donki",
                     help="EXPERIMENTAL (added 2026-09-17), off by default: "
                          "'donki' (locked default) uses this project's own "
                          "cached DONKI pull (donki_cme_2026.json); "
                          "'production' uses CLEAR's own real production "
                          "CME-selection output (checking_cme_production.json, "
                          "synced from solsticedisk) -- tested, a real "
                          "negative, see LOG (trades away coverage for width); "
                          "'donki_width' uses DONKI's own RAW API response "
                          "(donki_raw_full.json) instead of the older "
                          "normalized cache -- same dense ~1200-event "
                          "coverage as the locked default, but with a real "
                          "per-event half_angle (present on 1239/1240 events) "
                          "for a physically-correct variable-width "
                          "Earth-directedness test, without giving up "
                          "coverage the way 'production' did.")
    ap.add_argument("--quiet-bias", action="store_true",
                     help="EXPERIMENTAL (added 2026-09-24), TESTED NEGATIVE (see "
                          "accuracy_improvement_tracking.txt LOG) -- kept for "
                          "comparison only. Adds a FLAT quiet-regime bias term "
                          "(mean historical obs-blend residual over QUIET rows), "
                          "blended out as P(disturbed) rises. Superseded by "
                          "--quiet-regression, which targets the same gap with a "
                          "per-row prediction instead of one constant.")
    ap.add_argument("--quiet-regression", action="store_true",
                     help="EXPERIMENTAL (added 2026-09-24): the follow-up to "
                          "--quiet-bias's negative result -- see module "
                          "docstring, 'QUIET-REGIME REGRESSION'. Fits a ridge "
                          "regression predicting the current quiet-regime "
                          "residual (obs-blend) continuously from the same "
                          "trend/volatility/bfield features the classifier "
                          "uses, instead of one flat number. Blended out as "
                          "P(disturbed) rises, same as --quiet-bias. Wins over "
                          "--quiet-bias if both are given.")
    ap.add_argument("--quiet-regression-alpha", type=float, default=QUIET_REGRESSION_ALPHA,
                     help=f"ridge regularization strength for --quiet-regression "
                          f"(default {QUIET_REGRESSION_ALPHA:g}, swept independently -- "
                          f"see accuracy_improvement_tracking.txt LOG)")
    ap.add_argument("--ew-blend", action="store_true",
                     help="EXPERIMENTAL (added 2026-10-01): extend the locked "
                          "adaptive blend (raw vs. persist27, Section 4.1) to a "
                          "third candidate -- early_warning.py's own 6h-ahead "
                          "backtest -- using the same inverse-trailing-error "
                          "weighting. See module docstring, 'EARLY-WARNING AS A "
                          "THIRD BLEND CANDIDATE'. Falls back to the existing "
                          "2-way blend wherever early_warning's backtest is "
                          "unavailable.")
    ap.add_argument("--ew-blend-window-hours", type=float, default=EW_BLEND_WINDOW_HOURS,
                     help=f"trailing window (hours) for --ew-blend's inverse-error "
                          f"weighting (default {EW_BLEND_WINDOW_HOURS:g}, swept "
                          f"independently -- see accuracy_improvement_tracking.txt LOG)")
    ap.add_argument("--ew-blend-proba-gate", type=float, default=EW_BLEND_PROBA_GATE,
                     help="EXPERIMENTAL (added 2026-10-01): only apply --ew-blend "
                          "where leave-one-event-out P(disturbed) is below this "
                          "threshold, leaving rows at/above it IDENTICAL to the "
                          "locked default. One of three gate conditions -- see "
                          "--ew-blend-quiet-gate/--ew-blend-candidate-gate; the "
                          "validated hybrid (module docstring, 'EW-BLEND HYBRID "
                          "GATE') uses all three together with this set to 0.1. "
                          "Omit for no gating.")
    ap.add_argument("--ew-blend-quiet-gate", action="store_true",
                     help="EXPERIMENTAL (added 2026-10-01): only apply --ew-blend "
                          "where obs is currently <=DISTURBED_KMS (ground truth "
                          "at backtest time; the just-measured live value at "
                          "--live time -- never a future value). One of three "
                          "gate conditions, see --ew-blend-proba-gate.")
    ap.add_argument("--ew-blend-candidate-gate", action="store_true",
                     help="EXPERIMENTAL (added 2026-10-01): only apply --ew-blend "
                          "where there is no already-known inbound DONKI CME "
                          "candidate (the locked CME-magnitude shift already "
                          "handles those well on its own). One of three gate "
                          "conditions, see --ew-blend-proba-gate.")
    args = ap.parse_args()
    use_cme_magnitude = not args.no_cme_magnitude
    use_ew_gate = not args.no_ew_gate
    use_lasco_calibration = not args.no_lasco_calibration
    if args.live:
        apply_to_live(use_cme_magnitude, use_ew_gate, use_lasco_calibration, args.candidate_source,
                       use_quiet_bias=args.quiet_bias, use_quiet_regression=args.quiet_regression,
                       quiet_regression_alpha=args.quiet_regression_alpha,
                       use_ew_blend=args.ew_blend, ew_blend_window_hours=args.ew_blend_window_hours,
                       ew_blend_proba_gate=args.ew_blend_proba_gate,
                       use_ew_blend_quiet_gate=args.ew_blend_quiet_gate,
                       use_ew_blend_candidate_gate=args.ew_blend_candidate_gate)
    else:
        loo_validate(use_cme_magnitude, use_ew_gate, use_lasco_calibration, args.candidate_source,
                      use_quiet_bias=args.quiet_bias, use_quiet_regression=args.quiet_regression,
                      quiet_regression_alpha=args.quiet_regression_alpha,
                      use_ew_blend=args.ew_blend, ew_blend_window_hours=args.ew_blend_window_hours,
                      ew_blend_proba_gate=args.ew_blend_proba_gate,
                      use_ew_blend_quiet_gate=args.ew_blend_quiet_gate,
                      use_ew_blend_candidate_gate=args.ew_blend_candidate_gate)
