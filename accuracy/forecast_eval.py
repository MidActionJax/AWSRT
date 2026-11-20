#!/usr/bin/env python3
"""Score the IH-only 24-48 h forecasts against observations, by lead time.

Data:
  forecasts  RESULTS_<src>_<stamp>/IH/sat_earth_<stamp>.sat (one 48 h
             forecast each; spacedisk input/forecast_ih/forecasts, copied
             locally), speed = |u| from ux uy uz (km/s)
  obs        the dashboard series (eval_harness.fetch: observed OMNI/SWPC
             speed, 10-min bins) cached to --series so reruns are fixed

Everything is put on the 10-min bin grid (eval_harness.BIN_S).  A
forecast issued at t0 predicts bins t0 .. t0+48 h; lead = target - t0.

Reference predictors scored alongside the raw forecast, all causal
(known at t0):
  persist0   the last observed speed at or before t0, held flat
  persist27  observed speed one solar rotation (27 d) before the target
  blend      0.5 raw + 0.5 persist27 (the Phase 1b blend)

    python forecast_eval.py --forecasts C:/.../AWSRT/forecast_ih
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os

import numpy as np

from eval_harness import BIN_S, fetch

ROT_BINS = round(27 * 86400 / BIN_S)
LEAD_BINS = round(48 * 3600 / BIN_S)


def load_series(path):
    """Dashboard series, cached: fetch once, then always read the file."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    doc = fetch()
    with open(path, "w") as f:
        json.dump(doc, f)
    return doc


def bin_map(t, v):
    return {round(ti / BIN_S): vi for ti, vi in zip(t, v) if vi is not None}


def load_forecast(sat_path):
    """{bin: speed}, issue bin, plus n and |B| maps from one sat file."""
    v, n, b = {}, {}, {}
    with open(sat_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 20 or not p[1].isdigit() or len(p[1]) != 4:
                continue
            t = dt.datetime(int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]),
                            int(p[6]), tzinfo=dt.timezone.utc).timestamp()
            k = round(t / BIN_S)
            ux, uy, uz = float(p[12]), float(p[13]), float(p[14])
            bx, by, bz = float(p[15]), float(p[16]), float(p[17])
            v[k] = math.sqrt(ux * ux + uy * uy + uz * uz)
            n[k] = float(p[11])
            b[k] = math.sqrt(bx * bx + by * by + bz * bz) * 1e5
    return v, n, b, min(v)


def load_forecasts(root):
    out = []
    for sat in sorted(glob.glob(os.path.join(root, "RESULTS_*", "IH", "sat_earth_*.sat"))):
        name = sat.split(os.sep)[-3]
        v, n, b, b0 = load_forecast(sat)
        out.append({"name": name, "src": name.split("_")[1], "issue_bin": b0,
                    "v": v, "n": n, "b": b})
    return out


def last_obs(obs_bins, b0, max_back=round(6 * 3600 / BIN_S)):
    for k in range(b0, b0 - max_back - 1, -1):
        if k in obs_bins:
            return obs_bins[k]
    return None


def table(fcs, obs_bins):
    """Rows (forecast idx, lead bin, obs, raw, persist0, persist27)."""
    rows = []
    for i, fc in enumerate(fcs):
        b0 = fc["issue_bin"]
        p0 = last_obs(obs_bins, b0)
        for k, raw in fc["v"].items():
            lead = k - b0
            if lead < 0 or lead > LEAD_BINS or k not in obs_bins:
                continue
            p27 = obs_bins.get(k - ROT_BINS)
            rows.append((i, lead, obs_bins[k], raw,
                         np.nan if p0 is None else p0,
                         np.nan if p27 is None else p27))
    return np.array(rows, dtype=float)


def scores(obs, pred):
    m = ~np.isnan(pred)
    o, p = obs[m], pred[m]
    if len(o) < 3:
        return None
    e = p - o
    return {"n": len(o), "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(np.mean(e)),
            "corr": float(np.corrcoef(o, p)[0, 1])}


def report(T, preds, buckets=((0, 6), (6, 12), (12, 24), (24, 36), (36, 48), (0, 48))):
    lead_h = T[:, 1] * BIN_S / 3600
    print(f"{'lead (h)':>9} {'predictor':>10} {'n':>6} {'MAE':>7} {'RMSE':>7} {'bias':>7} {'corr':>6}")
    for lo, hi in buckets:
        m = (lead_h >= lo) & (lead_h < hi if hi < 48 else lead_h <= hi)
        for name, p in preds.items():
            s = scores(T[m, 2], p[m])
            if s:
                print(f"{lo:>3}-{hi:<5} {name:>10} {s['n']:>6} {s['mae']:7.1f} {s['rmse']:7.1f} "
                      f"{s['bias']:+7.1f} {s['corr']:6.3f}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", required=True)
    ap.add_argument("--series", default="series_cache.json")
    a = ap.parse_args()
    doc = load_series(a.series)
    obs_bins = bin_map(doc["obs"]["t"], doc["obs"]["v"])
    fcs = load_forecasts(a.forecasts)
    T = table(fcs, obs_bins)
    print(f"{len(fcs)} forecasts, {len(T)} scored (forecast, lead) points")
    raw, p0, p27 = T[:, 3], T[:, 4], T[:, 5]
    report(T, {"raw": raw, "persist0": p0, "persist27": p27,
               "blend": 0.5 * raw + 0.5 * p27})
