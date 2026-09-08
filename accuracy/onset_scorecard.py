"""Live scorecard for the onset-risk forecast (early_warning v2), added 2026-10-07.

Everything the onset-risk forecast has been scored on so far is historical
(held out in 10-day blocks, but still the same 7-month record it was designed
on). This file builds the first truly out-of-sample record: every time a live
probability is issued, it is appended to onset_scorecard.csv with its issue
time; once its target time has passed, `--score` checks it against what was
actually measured.

What is predicted (matches the training label exactly): P(measured speed at
the TARGET time, issue time + hours, is above DISTURBED_KMS), for moments
that are calm (measured <= DISTURBED_KMS) when issued. It is NOT "crosses 500
at any point within the window" -- the score also reports that rate, for
information only.

Usage:
  python onset_scorecard.py                 # record a live forecast (default 12h)
  python onset_scorecard.py --hours 6 12 24 # record several lead times
  python onset_scorecard.py --score         # score every row whose target has passed
export_dashboard_data.py also appends the dashboard's own live value each run.
"""
import argparse
import csv
import datetime
import os

import numpy as np

from eval_harness import BIN_S, DISTURBED_KMS, fetch, to_bin_map
import early_warning as ew

SCORECARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onset_scorecard.csv")
MODEL_TAG = f"onset_v2_alpha{ew.ONSET_V2_ALPHA}"
FIELDS = ["logged_utc", "issued_utc", "issued_epoch", "target_utc", "target_epoch", "hours",
          "proba", "measured_now", "quiet_now", "feature_set", "model"]


def _utc(dt_naive):
    return dt_naive.replace(tzinfo=datetime.timezone.utc)


def append_row(result, path=SCORECARD):
    """Append one live result from early_warning.apply_to_live_classify_v2().
    Skips a duplicate (same issue time, lead time and model). Returns True if written."""
    issued = _utc(result["when"])
    target = _utc(result["target_when"])
    key = (str(int(issued.timestamp())), str(float(result["hours"])), MODEL_TAG)
    exists = os.path.exists(path)
    if exists:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if (r["issued_epoch"], str(float(r["hours"])), r["model"]) == key:
                    return False
    measured = result.get("measured_now")
    row = {
        "logged_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "issued_utc": issued.strftime("%Y-%m-%d %H:%M"),
        "issued_epoch": int(issued.timestamp()),
        "target_utc": target.strftime("%Y-%m-%d %H:%M"),
        "target_epoch": int(target.timestamp()),
        "hours": float(result["hours"]),
        "proba": round(float(result["proba"]), 4),
        "measured_now": "" if measured is None else round(float(measured), 1),
        "quiet_now": int(bool(result["quiet_now"])),
        "feature_set": result.get("feature_set", ""),
        "model": MODEL_TAG,
    }
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    return True


def record(hours_list):
    for h in hours_list:
        res = ew.apply_to_live_classify_v2(float(h))
        wrote = append_row(res)
        print(f"  {'recorded' if wrote else 'already recorded'}: {h:g}h, issued "
              f"{res['when']} UTC, P={res['proba']:.3f}, quiet_now={res['quiet_now']}")


def score(path=SCORECARD):
    if not os.path.exists(path):
        print("no scorecard yet -- run without --score first")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    obs = fetch()["obs"]
    ob = to_bin_map(obs["t"], obs["v"])
    latest_obs_bin = max(ob)
    print(f"{len(rows)} rows logged; measured data available through "
          f"{datetime.datetime.utcfromtimestamp(latest_obs_bin * BIN_S)} UTC\n")
    for h in sorted({float(r["hours"]) for r in rows}):
        sub = [r for r in rows if float(r["hours"]) == h and r["quiet_now"] == "1"]
        p, y_at, y_within, pending, missing = [], [], [], 0, 0
        for r in sub:
            tb = int(r["target_epoch"]) // BIN_S
            ib = int(r["issued_epoch"]) // BIN_S
            if tb > latest_obs_bin:
                pending += 1
                continue
            v = ob.get(tb)
            if v is None:
                missing += 1
                continue
            window = [ob[b] for b in range(ib + 1, tb + 1) if b in ob]
            p.append(float(r["proba"]))
            y_at.append(1.0 if v > DISTURBED_KMS else 0.0)
            y_within.append(1.0 if window and max(window) > DISTURBED_KMS else 0.0)
        print(f"== {h:g}h lead | calm-when-issued rows: {len(sub)} | scored {len(p)} | "
              f"pending {pending} | no measurement at target {missing}")
        if not p:
            continue
        p, y_at, y_within = map(np.array, (p, y_at, y_within))
        brier = float(np.mean((p - y_at) ** 2))
        print(f"   mean predicted {p.mean():.3f} | observed rate (above 500 AT target) "
              f"{y_at.mean():.3f} | crossed 500 at any point in window {y_within.mean():.3f}")
        print(f"   Brier {brier:.4f}" + (f"  (Brier skill vs always saying the observed rate: "
              f"{1 - brier / float(np.mean((y_at.mean() - y_at) ** 2)):+.3f})"
              if 0 < y_at.mean() < 1 else "  (only one outcome so far -- skill undefined)"))
        if 0 < y_at.mean() < 1:
            print(f"   AUC {ew.p2.auc(y_at, p):.3f}")
        print("   CAUTION: rows logged close together overlap in time and are not independent;"
              " treat small counts as anecdotes, not statistics.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, nargs="+", default=[12.0])
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.score:
        score()
    else:
        record(args.hours)
