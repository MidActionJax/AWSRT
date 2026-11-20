#!/usr/bin/env python3
"""Write the Justified Adaptive fluX (forecast) version of each IH-only
forecast, from the frozen model forecast_confidence.py saved.

For every RESULTS_<src>_<stamp>/ under --forecasts, writes under --out:
  RESULTS_<src>_<stamp>/IH/sat_earth_<stamp>.sat
      same file, same columns, every row kept; only ux uy uz are rescaled
      so the speed |u| is the corrected speed (flow direction unchanged).
      Rows where the correction can't be computed keep the raw values.
  RESULTS_<src>_<stamp>/IH/jaf_band_<stamp>.csv
      time_utc, lead_h, raw_v, jaf_v, lo80, hi80, lo50, hi50 (km/s), one
      row per 10-min bin
  RESULTS_<src>_<stamp>/PARAM.in   (copied)
  manifest_jaf.csv, README_jaf.md
Only uses information available at each forecast's issue time: the
forecast itself, the real-time run before it, observations up to it.

    python forecast_apply.py --forecasts C:/.../forecast_ih --out C:/.../forecasts_jaf
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import os
import pickle
import shutil

import numpy as np

import forecast_correction as fcx
import forecast_eval as fe
from eval_harness import BIN_S
from forecast_confidence import unc_matrix
from phase2_model import ridge_predict

H = fcx.H
GAP_MAX_BINS = 6   # 1 hour


def fmt(x):
    """SWMF-style float with a 3-digit exponent, like the u columns."""
    m, e = f"{x:.5E}".split("E")
    return f"{m}E{int(e):+04d}"


def correct_forecast(S, M, fc, mode="start", ahead_h=24.0, latency_h=3.6, now_bin=None):
    """mode "start": every target corrected with observations up to the
    run's start (one fixed correction per run).
    mode "fresh": each target t corrected as a forecaster would see it
    ahead_h before t: issue = max(run start + latency_h, t - ahead_h), i.e.
    with the observations in hand at that moment (the run is in hand
    latency_h after its start, and the correction is just arithmetic, so
    it can be redone whenever new observations arrive).
    now_bin (fresh mode, live use): never issue after "now"; targets whose
    ideal issue time is still in the future are corrected with the
    observations up to now, at the correspondingly longer lead, and get
    recomputed on later runs."""
    idx = {k: j for j, k in enumerate(M["features"])}
    at = fcx.forecast_model_at(S, fc)
    b0 = fc["issue_bin"]
    run_end = b0 + 48 * H
    lat, ahead = round(latency_h * H), round(ahead_h * H)
    out = {}
    for k in sorted(fc["v"]):
        Lrun = k - b0
        if Lrun < 0 or Lrun > 48 * H:
            continue
        issue = b0 if mode == "start" else max(b0 + lat, k - ahead)
        if now_bin is not None:
            issue = min(issue, now_bin)
            if issue < b0 + lat:          # the run wasn't in hand yet
                continue
        if issue > k:
            continue
        Lb = k - issue
        f = fcx.features(S, at, k, issue, end=run_end)
        if f is None:
            continue
        L = min(48, round(Lb / H))
        P = M["leads"][L]
        x = np.array([[f[c] for c in M["features"]]])
        blend = 0.5 * f["raw"] + 0.5 * f["p27"]
        pred = float(M["mix"][L] * ridge_predict(x, *P["rd"])[0] + (1 - M["mix"][L]) * blend)
        size = max(float(ridge_predict(unc_matrix(x, idx), *P["sc"])[0]), M["min_scale"])
        band = {b: (pred + M["quant"][L][b][0] * size, pred + M["quant"][L][b][1] * size)
                for b in M["bands"]}
        out[k] = (Lrun / H, f["raw"], pred, band)
    # short gaps (usually the 27-day-old observation missing for a few
    # bins): carry the correction across them, interpolating its offset
    # from the raw speed (and the band's offsets) between the nearest
    # corrected bins up to GAP_MAX_BINS away on each side
    done = sorted(out)
    for k in sorted(fc["v"]):
        Lb = k - b0
        if Lb < 0 or Lb > 48 * H or k in out or not done:
            continue
        if mode != "start" and Lb < lat:
            continue
        lo = max((d for d in done if d < k and k - d <= GAP_MAX_BINS), default=None)
        hi = min((d for d in done if d > k and d - k <= GAP_MAX_BINS), default=None)
        if lo is None and hi is None:
            continue
        nb = [d for d in (lo, hi) if d is not None]
        wts = [1.0] if len(nb) == 1 else [(hi - k) / (hi - lo), (k - lo) / (hi - lo)]
        raw = at(k)
        off = sum(w * (out[d][2] - out[d][1]) for w, d in zip(wts, nb))
        band = {b: tuple(raw + off + sum(w * (out[d][3][b][i] - out[d][2]) for w, d in zip(wts, nb))
                         for i in (0, 1)) for b in M["bands"]}
        out[k] = (Lb / H, raw, raw + off, band)
    return out


def write_one(src_dir, dst_dir, corr):
    stamp = os.path.basename(src_dir).split("_")[-1]
    os.makedirs(os.path.join(dst_dir, "IH"), exist_ok=True)
    sat_in = glob.glob(os.path.join(src_dir, "IH", "sat_earth_*.sat"))[0]
    n_corr = n_rows = 0
    with open(sat_in) as fi, open(os.path.join(dst_dir, "IH", os.path.basename(sat_in)), "w") as fo:
        for line in fi:
            p = line.split()
            if len(p) < 20 or not p[1].isdigit() or len(p[1]) != 4:
                fo.write(line)
                continue
            n_rows += 1
            t = dt.datetime(int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]),
                            tzinfo=dt.timezone.utc).timestamp()
            c = corr.get(round(t / BIN_S))
            ux, uy, uz = float(p[12]), float(p[13]), float(p[14])
            v = math.sqrt(ux * ux + uy * uy + uz * uz)
            if c is not None and v > 0:
                s = c[2] / v
                p[12], p[13], p[14] = fmt(ux * s), fmt(uy * s), fmt(uz * s)
                n_corr += 1
            fo.write(" ".join(p) + "\n")
    with open(os.path.join(dst_dir, "IH", f"jaf_band_{stamp}.csv"), "w") as f:
        f.write("time_utc,lead_h,raw_v,jaf_v,lo80,hi80,lo50,hi50\n")
        for k in sorted(corr):
            lead, raw, pred, band = corr[k]
            tt = dt.datetime.utcfromtimestamp(k * BIN_S).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(f"{tt},{lead:.2f},{raw:.1f},{pred:.1f},{band['80'][0]:.1f},{band['80'][1]:.1f},"
                    f"{band['50'][0]:.1f},{band['50'][1]:.1f}\n")
    if os.path.exists(os.path.join(src_dir, "PARAM.in")):
        shutil.copy(os.path.join(src_dir, "PARAM.in"), dst_dir)
    return n_corr, n_rows


README = """# Justified Adaptive fluX (forecast): corrected IH-only forecasts

Same folders and file names as the raw forecasts (`forecasts/`), so the same
scoring script can be pointed at this folder instead.

- `IH/sat_earth_<stamp>.sat`: the raw forecast's file with ONLY the speed
  corrected: ux, uy, uz are rescaled so |u| is the corrected speed (flow
  direction kept). Density, B, temperature etc. are the raw model's.
- `IH/jaf_band_<stamp>.csv`: per 10-min bin: lead_h, raw_v, jaf_v (corrected
  speed), and the 80% / 50% confidence band (lo80, hi80, lo50, hi50), km/s.
- `manifest_jaf.csv`: per forecast, how many sat rows were corrected.

The correction only uses what was known at the forecast's issue time: the
forecast, the real-time run before it, observations up to the issue time,
and observations one solar rotation (27 d) before each target time. It is
a per-lead-hour linear model trained on the real-time run's Jan-Sep 2026
history ({train_end} cutoff), so Oct-Nov forecasts were never trained on.
Model file: {model}.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="jaf_forecast_model.pkl")
    ap.add_argument("--series", default="series_cache_20261119.json")
    ap.add_argument("--mode", choices=("start", "fresh"), default="start")
    ap.add_argument("--ahead-h", type=float, default=24.0)
    a = ap.parse_args()
    M = pickle.load(open(a.model, "rb"))
    fcx.FEATURES = M["features"]
    S = fcx.Series(fe.load_series(a.series))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for fc in fe.load_forecasts(a.forecasts):
        corr = correct_forecast(S, M, fc, mode=a.mode, ahead_h=a.ahead_h)
        nc, nr = write_one(os.path.join(a.forecasts, fc["name"]), os.path.join(a.out, fc["name"]), corr)
        rows.append((fc["name"], nc, nr))
        print(f"{fc['name']:26} corrected {nc}/{nr} rows")
    with open(os.path.join(a.out, "manifest_jaf.csv"), "w") as f:
        f.write("forecast,rows_corrected,rows\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    with open(os.path.join(a.out, "README_jaf.md"), "w") as f:
        f.write(README.format(train_end=M["train_end"], model=os.path.basename(a.model)))


if __name__ == "__main__":
    main()
