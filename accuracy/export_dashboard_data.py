#!/usr/bin/env python3
"""Export the full historical raw/corrected/measured/band series to JSON for
the Solar Wind Accuracy Tracker artifact (dashboard_data.json).

This is a SNAPSHOT export, not a live feed: the source API
(clear-dev-clear.apps.campus.web.umich.edu) is on a campus-internal
network a public artifact page can't reach, so there is no way to make
that page poll live. Re-run this script whenever you want an updated
picture, then republish the artifact with the new dashboard_data.json.

Reproduces final_correction.py's own LOO-consistent historical series
(not an oracle fit -- each point's correction was computed with its own
disturbance event held out of training) plus uncertainty_model.py's
config-D band (blend-residual widths, centered on the corrected estimate,
OR-combined with the early-warning trigger) -- see
accuracy_improvement_tracking.txt for what these numbers mean and how
they were validated.

FIXED 2026-09-14: this script had its OWN separately-duplicated
flat-shift-only reconstruction of final_correction's correction logic,
which never picked up the CME-magnitude shift promoted to the locked
default the same day -- the dashboard was silently showing the OLD
(pre-2026-09-14) numbers for a full export cycle after the fix landed
elsewhere, caught only by noticing the exported "latest" value hadn't
changed at all after the promotion. Now calls fc._cme_candidates()
directly instead of re-deriving the shift inline, so this can't drift
out of sync with final_correction.py's actual behavior again.

    python export_dashboard_data.py [--out dashboard_data.json]
"""
from __future__ import annotations

import argparse
import datetime
import json

import numpy as np

from eval_harness import BIN_S, DISTURBED_KMS, fetch, to_bin_map
from uncertainty_model import band_width, ew_predictions
import early_warning as ew
import onset_scorecard
import phase2_model as p2
import final_correction as fc

EW_LEAD_HOURS = 6.0  # early_warning.py's most-validated horizon (VALID_HOURS = 6, 9, 12)

# FORECAST TILES, added 2026-09-20 per Jaxon's dashboard-layout request:
# one backtest tile per horizon ("predicted Nh ago, actual now"), same
# style as the original single 6h tile. 6/9/12 are early_warning.py's own
# VALID_HOURS (regression framing) -- 18h is included per Jaxon's ask but
# is explicitly OUTSIDE that validated range (early_warning.py's own
# docstring: "Beyond ~18h the advantage disappears and reverses"), so the
# frontend tile for 18h carries its own honest caveat rather than being
# presented as equally trustworthy.
FORECAST_HOURS = (6.0, 9.0, 12.0, 18.0)

# ONSET-RISK CARD, added 2026-09-17: early_warning.py's classification
# reframe (see accuracy_improvement_tracking.txt LOG, item #3 of the "11
# things") -- a genuine, validated forward-looking onset PROBABILITY,
# distinct from the "ew"/"ew_live" exact-value forecast above. 12h picked
# as a well-justified middle ground, not the single best number the sweep
# found (that was ~15-18h, AUC~0.83-0.85): 12h stays inside BOTH the
# original regression forecast's own validated range and the classify
# reframe's much wider one (6-36h, see VALID_HOURS_CLASSIFY), giving a
# meaningfully useful half-day lead time with strong, decisive
# discrimination (onset AUC 0.813 vs a 0.619 naive baseline).
ONSET_HOURS = 12.0


def build():
    doc = fetch()
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])
    bfield_bins = {"b": to_bin_map(obs["t"], obs["b"]), "bz": to_bin_map(obs["t"], obs["bz"]),
                   "n": to_bin_map(obs["t"], obs["n"])}

    p2.FEATURES = fc.FEATURES
    bins, data = p2.build_dataset(model_bins, obs_bins, bfield_bins=bfield_bins)
    X_all = p2.feature_matrix(data)
    obs_all = data["obs_v"]
    blend_all = data["blend_v"]
    model_all = data["model_v"]
    disturbed_mask_all = obs_all > DISTURBED_KMS
    events = p2.event_windows(obs_bins, threshold=600.0)

    # THE single shared implementation (final_correction.loo_corrected_series)
    # -- this script used to re-derive the fold loop independently and
    # silently fell out of sync when the pipeline's behavior changed twice
    # in one day; see accuracy_improvement_tracking.txt LOG, "SECOND
    # DUPLICATION BUG FOUND" and the loo_corrected_series() docstring.
    # Never re-derive this shift locally again.
    corrected_all, full_proba, has_candidate, ew_gate = fc.loo_corrected_series(
        bins, X_all, obs_all, blend_all, disturbed_mask_all, events)

    # config-D band: blend-residual widths, centered on corrected, + early-warning OR
    resid_blend = obs_all - blend_all
    narrow_half = band_width(resid_blend[~disturbed_mask_all], 0.8)
    wide_half = band_width(resid_blend[disturbed_mask_all], 0.8)

    ew_pred_by_bin = ew_predictions(6.0)
    ew_pred_at_row = np.array([ew_pred_by_bin.get(int(b), -np.inf) for b in bins])
    ew_triggered = ew_pred_at_row > 500.0
    predicted_disturbed_or = (full_proba > 0.32) | ew_triggered
    band_half = np.where(predicted_disturbed_or, wide_half, narrow_half)

    # Early-warning backtest, aligned to the TARGET time it predicted (not
    # the time it was issued): row at bin b gets whatever forecast was made
    # EW_LEAD_HOURS earlier, for this exact moment -- so on a chart, this
    # series plotted at the same x-position as "obs" directly shows
    # forecast-vs-actual, honestly including the real misses (see
    # accuracy_improvement_tracking.txt, 2026-09-16: this 6h point forecast
    # missed the Sep 15 storm onset badly -- that miss is visible here too,
    # not hidden. This is a genuinely different, harder-won kind of honesty
    # than the nowcast correction: it is scored against the future, not the
    # present.
    lead_bins = round(EW_LEAD_HOURS * 3600 / BIN_S)
    ew_backtest = [ew_pred_by_bin.get(int(b) - lead_bins) for b in bins]

    # Same backtest, repeated for every horizon in FORECAST_HOURS (see note
    # above) -- 6h reuses ew_pred_by_bin already computed for the band
    # trigger above rather than re-running early_warning's LOO fit twice.
    forecast_backtest = {}
    for h in FORECAST_HOURS:
        pred_by_bin_h = ew_pred_by_bin if h == EW_LEAD_HOURS else ew_predictions(h)
        lead_bins_h = round(h * 3600 / BIN_S)
        forecast_backtest[h] = [pred_by_bin_h.get(int(b) - lead_bins_h) for b in bins]

    # Onset-risk series (see ONSET_HOURS note above): a real, LOO-validated
    # P(disturbed within ONSET_HOURS) at every bin -- built from
    # early_warning.classify_series(), the ONE shared implementation (not
    # re-derived here), keyed by bin so it can be looked up against this
    # script's own bin array even though the two feature sets/row-sets
    # aren't guaranteed identical.
    onset_p_by_bin = {}
    try:
        # onset-risk v2 (2026-10-06): trained on quiet-now rows only, alpha 0.01,
        # held out in 10-day blocks; probabilities exist only for quiet-now rows.
        classify_result = ew.classify_series_v2(ONSET_HOURS)
        onset_p_by_bin = {int(b_): float(p_) for b_, p_ in
                          zip(classify_result["bins"].tolist(),
                              classify_result["proba_future"].tolist())
                          if not np.isnan(p_)}
    except Exception as e:  # noqa: BLE001
        print(f"onset-risk series unavailable, skipping: {e}")

    out = []
    for i, b in enumerate(bins):
        row = {
            "t": int(b * BIN_S),
            "raw": round(float(model_all[i]), 1),
            "corrected": round(float(corrected_all[i]), 1),
            "obs": round(float(obs_all[i]), 1),
            "band_lo": round(float(corrected_all[i] - band_half[i]), 1),
            "band_hi": round(float(corrected_all[i] + band_half[i]), 1),
            "flagged": bool(predicted_disturbed_or[i]),
        }
        if ew_backtest[i] is not None:
            row["ew"] = round(float(ew_backtest[i]), 1)
        for h in FORECAST_HOURS:
            val = forecast_backtest[h][i]
            if val is not None:
                row[f"ew_{int(h)}"] = round(float(val), 1)
        onset_p = onset_p_by_bin.get(int(b))
        if onset_p is not None:
            row["onset_p"] = round(float(onset_p), 3)
        out.append(row)

    # BUG FIXED 2026-09-28: the most recent row(s) in `out` -- i.e. exactly
    # the "Corrected Prediction" tile's value, since the dashboard reads
    # the LAST row with a real obs -- come from loo_corrected_series's
    # non-event branch, which gates via _ew_rising_mask()/ew_predictions(),
    # a HISTORICAL LOO-backtest dict that only contains bins with an
    # already-KNOWN future outcome. The most recent ~EW_GATE_HOURS worth
    # of bins never has that (their future hasn't happened yet), so the
    # gate silently defaulted to "rising, don't gate" for exactly the bins
    # this dashboard actually displays as "now" -- the same root cause as
    # the apply_to_live() bug fixed the same day in final_correction.py
    # (see accuracy_improvement_tracking.txt LOG, 2026-09-28), just hit
    # from the export side instead of the CLI --live side. Fix: recompute
    # the last row's correction via fc.apply_to_live(), which calls
    # early_warning.apply_to_live() for a genuine live forecast instead of
    # this dict lookup, and only apply it when its timestamp actually
    # matches the last row (falls back to the old value otherwise, e.g. if
    # data lag ever puts them a bin apart -- never worse than before).
    if out:
        try:
            live_corr = fc.apply_to_live()
            live_t = int(live_corr["when"].replace(tzinfo=datetime.timezone.utc).timestamp())
            if live_t == out[-1]["t"]:
                band_half_last = band_half[-1]
                out[-1]["corrected"] = round(float(live_corr["corrected"]), 1)
                out[-1]["band_lo"] = round(float(live_corr["corrected"] - band_half_last), 1)
                out[-1]["band_hi"] = round(float(live_corr["corrected"] + band_half_last), 1)
        except Exception as e:  # noqa: BLE001
            print(f"live-gated correction for the most recent row unavailable, "
                  f"keeping historical value: {e}")

    # Genuine forward-looking point: early_warning's own live forecast,
    # issued from the current most-recent bin, for EW_LEAD_HOURS from now --
    # a real point beyond "now" on the chart, not a backtest. Appended only
    # (no raw/obs/corrected -- that moment hasn't happened yet).
    live_row = {}
    try:
        live_fc = ew.apply_to_live(EW_LEAD_HOURS)
        live_row = {
            "t": int(live_fc["target_when"].replace(tzinfo=datetime.timezone.utc).timestamp()),
            "ew": round(float(live_fc["forecast"]), 1),
            "ew_live": True,
        }
    except Exception as e:  # noqa: BLE001
        print(f"live early-warning forecast unavailable, skipping: {e}")

    # Live onset-risk probability -- issued from the CURRENT moment (not
    # the future target time above), so attached to the same trailing
    # live row rather than given its own x-position on the chart.
    try:
        live_onset = ew.apply_to_live_classify_v2(ONSET_HOURS)
        try:
            # live scorecard (onset_scorecard.py): log the issued probability so it
            # can be checked against what is measured once its target time passes
            onset_scorecard.append_row(live_onset)
        except Exception as e:  # noqa: BLE001
            print(f"onset scorecard logging failed, continuing: {e}")
        live_row["onset_p_live"] = round(float(live_onset["proba"]), 3)
        live_row["onset_hours"] = ONSET_HOURS
        live_row.setdefault("t", int(live_onset["when"].replace(
            tzinfo=datetime.timezone.utc).timestamp()))
    except Exception as e:  # noqa: BLE001
        print(f"live onset-risk probability unavailable, skipping: {e}")

    if live_row:
        out.append(live_row)

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dashboard_data.json")
    args = ap.parse_args()

    rows = build()
    with open(args.out, "w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"latest: {rows[-1]}")
