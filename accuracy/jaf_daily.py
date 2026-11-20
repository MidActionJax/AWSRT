#!/usr/bin/env python3
"""Hourly Justified Adaptive fluX (forecast) for new IH-only forecasts, on
spacedisk: corrects every run that has arrived in input/forecast_ih/forecasts
and writes the same files the Oct-Nov delivery has, so the scorer and the
website read them the same way.

  forecasts_jaf/RESULTS_<src>_<stamp>/jaf_<stamp>.csv        v n b bz T
  forecasts_jaf_band/RESULTS_<src>_<stamp>/jaf_band_<stamp>.csv  speed band

Fresh observations, as scored: the value at target t uses what a
forecaster has 24 h before t (issue = max(run start + 3.6 h, t - 24 h)).
A run still in progress has targets whose issue time is in the future;
those are corrected with the observations up to now (longer lead) and
recomputed on later runs.  Once every target's issue time has passed, the
run is FINAL and never touched again.  Rows where any variable can't be
corrected (e.g. an observation gap) are left out, never filled with raw.

Only runs starting on or after --first are processed (default
2026-11-20, the freeze): the Oct 1-Nov 19 files were made by the
pre-October models and must stay as delivered.

Inputs on spacedisk: the dashboard's own series (products/earth.json:
observations + the real-time model, rebuilt every 10 min by
tick_rtsw.sh) and the frozen models.  Nothing is fetched from outside.

    python3 jaf_daily.py            (cron, hourly; defaults are spacedisk paths)
    python3 jaf_daily.py --now 2026-11-21T06:00 ...   (replay for testing)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import sys
import time
import warnings

import numpy as np

# Python 3.12+ deprecates utcnow/utcfromtimestamp, and empty windows in the
# rolling stats warn; neither affects the results, both would fill the
# hourly cron log
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import forecast_apply as fa          # noqa: E402
import forecast_correction as fcx    # noqa: E402
import forecast_eval as fe           # noqa: E402
import forecast_multivar as mv       # noqa: E402
import jaf_models                    # noqa: E402
from eval_harness import BIN_S, fetch  # noqa: E402

H = round(3600 / BIN_S)
ROOT = "/data/clear/Website_CLEAR/Website/RealTimeSW"
OBS_STALE_H = 6.0


def log(msg):
    print(f"[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def ubin_iso(s):
    return round(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp() / BIN_S)


def group_readable(path, is_dir=False):
    try:
        os.chmod(path, 0o2775 if is_dir else 0o664)
    except OSError:
        pass


def write_band(path, corr, b0, lat):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("time_utc,lead_h,raw_v,jaf_v,lo80,hi80,lo50,hi50\n")
        for k in sorted(corr):
            if k - b0 < lat:
                continue
            lead, raw, pred, band = corr[k]
            tt = dt.datetime.utcfromtimestamp(k * BIN_S).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(f"{tt},{lead:.2f},{raw:.1f},{pred:.1f},{band['80'][0]:.1f},{band['80'][1]:.1f},"
                    f"{band['50'][0]:.1f},{band['50'][1]:.1f}\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", default=f"{ROOT}/input/forecast_ih/forecasts")
    ap.add_argument("--out", default=f"{ROOT}/input/forecast_ih/forecasts_jaf")
    ap.add_argument("--band-out", default=f"{ROOT}/input/forecast_ih/forecasts_jaf_band")
    ap.add_argument("--series", default=f"{ROOT}/products/earth.json")
    ap.add_argument("--speed-model", default=os.path.join(HERE, "jaf_forecast_model_ops_20261119.json"))
    ap.add_argument("--multi-model", default=os.path.join(HERE, "jaf_multivar_model_ops_20261119.json"))
    ap.add_argument("--state", default=os.path.join(HERE, "state"))
    ap.add_argument("--first", default="2026-11-20", help="only runs starting on/after this (UTC)")
    ap.add_argument("--now", help="pretend it is this UTC time (testing)")
    ap.add_argument("--latency-h", type=float, default=3.6)
    ap.add_argument("--ahead-h", type=float, default=24.0)
    a = ap.parse_args()

    now_bin = ubin_iso(a.now) if a.now else round(time.time() / BIN_S)
    lat, ahead = round(a.latency_h * H), round(a.ahead_h * H)
    first_bin = ubin_iso(a.first)
    os.makedirs(a.state, exist_ok=True)

    doc = fetch(path=a.series)
    if a.now:   # replay: hide everything that didn't exist yet
        for key in ("obs", "model"):
            d = doc[key]
            keep = [i for i, t in enumerate(d["t"]) if round(t / BIN_S) <= now_bin]
            for f in list(d):
                if isinstance(d[f], list) and len(d[f]) == len(d["t"]):
                    d[f] = [d[f][i] for i in keep]
    last_obs = max(round(t / BIN_S) for t, v in zip(doc["obs"]["t"], doc["obs"]["v"]) if v is not None)
    obs_ok = (now_bin - last_obs) <= OBS_STALE_H * H
    if not obs_ok:
        log(f"WARNING: newest observation is {(now_bin - last_obs) / H:.1f} h old; "
            "correcting anyway, but no run is marked final")

    M = jaf_models.load(a.speed_model)
    MM = jaf_models.load(a.multi_model)
    fcx.FEATURES = M["features"]
    mv.OPTS.update(MM.get("opts", {}))
    S = fcx.Series(doc)
    G = mv.Grid(doc)

    done = skipped = 0
    for sat in sorted(glob.glob(os.path.join(a.forecasts, "RESULTS_*", "IH", "sat_earth_*.sat"))):
        name = os.path.basename(os.path.dirname(os.path.dirname(sat)))
        stamp = name.split("_")[-1]
        b0_stamp = round(dt.datetime.strptime(stamp, "%y%m%dt%H%M").replace(
            tzinfo=dt.timezone.utc).timestamp() / BIN_S)
        if b0_stamp < first_bin or os.path.exists(os.path.join(a.state, name + ".final")):
            skipped += 1
            continue
        v, n, b, b0 = fe.load_forecast(sat)
        fc = {"name": name, "issue_bin": b0, "v": v, "n": n, "b": b}
        corr = fa.correct_forecast(S, M, fc, mode="fresh", ahead_h=a.ahead_h,
                                   latency_h=a.latency_h, now_bin=now_bin)
        if not corr:
            log(f"{name}: nothing correctable yet")
            continue
        vspeed = {k: c[2] for k, c in corr.items()}
        targets, out, full = mv.apply_run(G, MM["models"], sat, vspeed, lat, ahead, now_bin=now_bin)
        targets = [k for k in targets if full[k]]
        for d in (os.path.join(a.out, name), os.path.join(a.band_out, name)):
            os.makedirs(d, exist_ok=True)
            group_readable(d, is_dir=True)
        p = os.path.join(a.out, name, f"jaf_{stamp}.csv")
        mv.write_csv(p, targets, out)
        group_readable(p)
        pb = os.path.join(a.band_out, name, f"jaf_band_{stamp}.csv")
        write_band(pb, corr, b0, lat)
        group_readable(pb)
        final = obs_ok and now_bin >= b0 + 48 * H - ahead + H
        if final:
            open(os.path.join(a.state, name + ".final"), "w").write(f"{dt.datetime.utcnow():%FT%TZ}\n")
        done += 1
        log(f"{name}: {len(targets)} bins corrected{' (FINAL)' if final else ''}")
    log(f"done: {done} runs updated, {skipped} skipped (before {a.first} or final)")


if __name__ == "__main__":
    main()
