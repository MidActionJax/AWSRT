#!/usr/bin/env python3
"""FIRST INTEGRATION PROTOTYPE: one live report combining the four
independently-validated pieces built this project --

  final_correction.py   locked point-estimate correction (blend + P(disturbed)
                         classifier-gated shift, RMSE 93.6, corr 0.524 overall)
  uncertainty_model.py   honest confidence band, sized by a SEPARATE regime
                         classifier + OR-combined with...
  early_warning.py       ...a genuine 6-12h-ahead forecast, so the band widens
                         even when the CURRENT moment still looks quiet but an
                         onset is predicted to arrive soon

This is explicitly NOT "the entire model, done and complete" -- it is a
first pass at the integration/design work flagged as real, unstarted work
in accuracy_improvement_tracking.txt's Open Questions ("how to actually
combine the four validated pieces into ONE system before anything can
ship"). Per the project's standing rule, nothing here goes near the live
dashboard until that full design (including the product/display question,
also still undiscussed) is actually settled with Jaxon/Zhao.

CLASSIFIERS UNIFIED (2026-09-12): this prototype originally surfaced final_
correction.py's classifier and uncertainty_model.py's classifier as
independently fit (different alpha, different feature set), reporting two
different P(disturbed) numbers for the same moment. Swept properly (see
accuracy_improvement_tracking.txt LOG, "CLASSIFIER RECONCILIATION"):
uncertainty_model's own config (bfield-only features, alpha=0.005) does
at least as well on final_correction's own disturbed-row correlation
metric as its previous separately-tuned config (bfield+DONKI, alpha=0.01)
-- DONKI activity features added nothing here. final_correction.py now
uses that same feature set and alpha, so the two scripts' classifiers are
mathematically the same fit on the same data, not just similar -- the
live SUMMARY below prints one P(disturbed) number now (from each script
independently, as a live cross-check that they agree, which they do).

VALIDATED (2026-09-12, `python combined_model.py --validate`): centering
the confidence interval on final_correction's CORRECTED point estimate,
instead of on the raw blend uncertainty_model.py was originally fit
against, was an untested assumption baked into the SUMMARY section below
-- checked properly rather than left as a guess. Result: it's a genuine
improvement, not just harmless. Band half-widths stay computed from
obs-vs-BLEND residuals (recomputing them from obs-vs-corrected residuals
is WORSE -- narrower quantiles that give back most of the gain, see
--validate output), but applying that same band around the CORRECTED
center, combined with the early-warning OR-trigger, gets disturbed-row
coverage to 86.8% (vs 80.2% centered-on-corrected without early-warning,
vs 70.6% for the original blend-centered/classifier-only baseline) with
overall coverage at 84.1% (target 80%). Best disturbed-row coverage found
anywhere in this project so far. No code change was needed -- the
SUMMARY section already did this by construction -- but it was an
un-validated accident until now.

Also worth being explicit about: "obs" (the real-time measured speed) is
available at essentially the same cadence as the model's own output --
see accuracy_improvement_tracking.txt LOG, "GENUINE FORWARD-HORIZON
FORECAST", on why a same-time prediction of obs from obs is trivially
perfect and useless. Nothing below claims genuine lead time except the
early-warning line -- the point estimate and band are a real-time
NOWCAST/correction of the model's own contemporaneous estimate, not a
forecast into times obs hasn't already reached.

    python combined_model.py
"""
from __future__ import annotations

import argparse

import numpy as np

from eval_harness import DISTURBED_KMS, fetch, to_bin_map
from uncertainty_model import band_width, ew_predictions
import final_correction as fc
import uncertainty_model as um
import phase2_model as p2

ALPHA_FC = fc.LOGISTIC_ALPHA
POWER_FC = fc.POWER
ALPHA_UM = 0.005
RISK_THRESHOLD_UM = 0.32


def validate(coverage=0.8, ew_hours=6.0, ew_threshold=500.0):
    """Reproduce the claim in this module's docstring: does the confidence
    band still hold its coverage when centered on final_correction's
    CORRECTED point estimate (what the live SUMMARY below actually shows)
    instead of the raw blend uncertainty_model.py was originally validated
    against? Was an untested assumption until this was written -- see the
    VALIDATED note in the module docstring for the answer and the numbers.
    """
    doc = fetch()
    model, obs = doc["model"], doc["obs"]
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])
    bfield_bins = {"b": to_bin_map(obs["t"], obs["b"]), "bz": to_bin_map(obs["t"], obs["bz"]),
                   "n": to_bin_map(obs["t"], obs["n"])}

    # final_correction.py's and uncertainty_model.py's classifiers are now
    # unified (same feature set, same alpha) -- one dataset build serves both
    p2.FEATURES = ("model_v", "trend_3h", "trend_6h", "volatility_24h", "disagreement") + p2.BFIELD_FEATURES
    bins, data = p2.build_dataset(model_bins, obs_bins, bfield_bins=bfield_bins)
    X_fc = p2.feature_matrix(data)
    obs_all = data["obs_v"]
    blend_all = data["blend_v"]
    disturbed_mask_all = obs_all > DISTURBED_KMS
    events = p2.event_windows(obs_bins, threshold=600.0)

    # ONE classifier now serves both jobs (unified 2026-09-12: same feature
    # set, same alpha for final_correction's point-shift gate and
    # uncertainty_model's band gate -- see module docstring)
    assert ALPHA_FC == ALPHA_UM, "classifiers were unified -- these should no longer differ"
    # THE single shared implementation (final_correction.loo_corrected_series)
    # -- this function used to re-derive the fold loop independently THREE
    # times across this codebase and silently fell out of sync twice in one
    # day when the pipeline's behavior changed; see
    # accuracy_improvement_tracking.txt LOG, "THIRD DUPLICATION BUG FOUND"
    # and the loo_corrected_series() docstring. Never re-derive this again.
    corrected_all, full_proba, has_candidate, ew_gate = fc.loo_corrected_series(
        bins, X_fc, obs_all, blend_all, disturbed_mask_all, events)
    predicted_disturbed = full_proba > RISK_THRESHOLD_UM

    resid_blend = obs_all - blend_all
    narrow_half_blend = band_width(resid_blend[disturbed_mask_all == 0], coverage)
    wide_half_blend = band_width(resid_blend[disturbed_mask_all == 1], coverage)
    resid_corr = obs_all - corrected_all
    narrow_half_corr = band_width(resid_corr[disturbed_mask_all == 0], coverage)
    wide_half_corr = band_width(resid_corr[disturbed_mask_all == 1], coverage)

    ew_pred_by_bin = ew_predictions(ew_hours)
    ew_pred_at_row = np.array([ew_pred_by_bin.get(b, -np.inf) for b in bins])
    predicted_disturbed_or = predicted_disturbed | (ew_pred_at_row > ew_threshold)

    dist_mask = disturbed_mask_all == 1

    def report(label, center, half_narrow, half_wide, trigger):
        half = np.where(trigger, half_wide, half_narrow)
        covered = np.abs(obs_all - center) <= half
        print(f"{label:70s} overall={covered.mean()*100:5.1f}%  "
              f"disturbed-row={covered[dist_mask].mean()*100:5.1f}%")

    print(f"Band half-widths from BLEND residuals:     narrow={narrow_half_blend:.1f}  wide={wide_half_blend:.1f}")
    print(f"Band half-widths from CORRECTED residuals: narrow={narrow_half_corr:.1f}  wide={wide_half_corr:.1f}")
    print(f"\n(target coverage = {coverage*100:.0f}%)")
    report("A) band(blend-widths), centered on BLEND [uncertainty_model.py's own validation]",
           blend_all, narrow_half_blend, wide_half_blend, predicted_disturbed)
    report("B) band(blend-widths), centered on CORRECTED, classifier only",
           corrected_all, narrow_half_blend, wide_half_blend, predicted_disturbed)
    report("C) band(corrected-widths), centered on CORRECTED, classifier only",
           corrected_all, narrow_half_corr, wide_half_corr, predicted_disturbed)
    report("D) band(blend-widths), centered on CORRECTED, + early-warning OR [this script's live SUMMARY]",
           corrected_all, narrow_half_blend, wide_half_blend, predicted_disturbed_or)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                     help="reproduce the band-centering validation described "
                          "in this module's docstring instead of the live report")
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--risk-threshold", type=float, default=0.32,
                     help="uncertainty_model's band-classifier threshold "
                          "(validated config: 0.32)")
    ap.add_argument("--alpha", type=float, default=0.005,
                     help="the (now unified) classifier's regularization -- "
                          "0.005 matches final_correction.py's own "
                          "LOGISTIC_ALPHA, see module docstring")
    ap.add_argument("--ew-hours", type=float, default=6.0)
    ap.add_argument("--ew-threshold", type=float, default=500.0)
    ap.add_argument("--no-early-warning", action="store_true",
                     help="disable the early-warning OR-trigger (on by default "
                          "-- this is the validated, integrated configuration)")
    args = ap.parse_args()

    if args.validate:
        validate(args.coverage, args.ew_hours, args.ew_threshold)
        return

    print("=" * 70)
    print("COMBINED LIVE REPORT -- research prototype, not a deployed system")
    print("=" * 70)

    print("\n--- point estimate (final_correction.py, LOCKED) ---")
    fc_result = fc.apply_to_live()

    um_args = argparse.Namespace(
        url=None, json=None, alpha=args.alpha, donki=None, donki_mode="activity",
        coverage=args.coverage, risk_threshold=args.risk_threshold, bfield=True,
        early_warning=not args.no_early_warning, ew_hours=args.ew_hours,
        ew_threshold=args.ew_threshold,
    )
    print("\n--- confidence band (uncertainty_model.py"
          + (" + early_warning.py OR-trigger" if um_args.early_warning else "")
          + ") ---")
    um_result = um.apply_to_live(um_args)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    lower = fc_result["corrected"] - um_result["band_half"]
    upper = fc_result["corrected"] + um_result["band_half"]
    print(f"  as of {fc_result['when']} UTC:")
    print(f"    raw model:                {fc_result['raw_model']:.1f} km/s")
    print(f"    corrected point estimate: {fc_result['corrected']:.1f} km/s")
    print(f"    {args.coverage*100:.0f}% interval (centered on corrected):  "
          f"[{lower:.1f}, {upper:.1f}] km/s")
    print(f"    regime:                   "
          f"{'ELEVATED RISK' if um_result['triggered'] else 'quiet'}")
    if um_result["early_warning"] is not None:
        ew_r = um_result["early_warning"]
        onset_flag = ew_r["forecast"] > args.ew_threshold
        if onset_flag and not (fc_result["proba_disturbed"] > args.risk_threshold):
            print(f"    ** EARLY WARNING: current moment reads quiet, but "
                  f"early_warning forecasts {ew_r['forecast']:.1f} km/s by "
                  f"{ew_r['target_when']} UTC ({args.ew_hours:g}h ahead) -- "
                  f"this is the scenario the OR-trigger exists for **")
        else:
            print(f"    early-warning({args.ew_hours:g}h): "
                  f"{ew_r['forecast']:.1f} km/s by {ew_r['target_when']} UTC")
    print(f"\n  classifiers are unified (same feature set + alpha, see module "
          f"docstring): final_correction's P(disturbed)={fc_result['proba_disturbed']:.3f} "
          f"and uncertainty_model's P(disturbed)={um_result['proba_disturbed']:.3f} "
          f"should read the same (each fit independently as a live cross-check).")


if __name__ == "__main__":
    main()
