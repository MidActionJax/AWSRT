#!/usr/bin/env python3
"""Build the website's forecast product: one forecast.json (+ latest.csv)
holding everything the Forecast tab shows.  Runs on spacedisk right after
jaf_daily.py (same cron line), reads only spacedisk files.

  meta       newest run, data cut-off, when the next run is due
  now        the latest measurement and its age
  measured   last 24 h of measurements (v n b bz T)
  forecast   newest run, corrected with the observations up to NOW
             (lead = target - now), with the 80% / 50% speed band and the
             raw model; from now to the run's end
  cards      four 12 h blocks: label from the corrected speed, its range,
             and a confidence word from the band width
  next_fast  when the corrected speed next reaches fast-stream level
  verify     last 48 h: what was forecast 24 h ahead (the archived
             corrected runs, newest run in hand 24 h before each time --
             the operational rule Keheng scores) vs the raw model vs
             measured; average miss, biggest miss, share inside the band
  scorecard  last 7 UTC days: daily mean measured / raw / corrected speed
             and daily mean miss
  runs       the last four runs' corrected 24 h-ahead speed
  skill      24 h-ahead RMSE skill vs 27-day persistence per variable,
             since Oct 1 and over the last 30 days (all periods: unlike
             Keheng's scoring, CMEs are not removed here)
  cme        fast, roughly Earth-directed CMEs in the DONKI mirror whose
             estimated arrival is within the window (catalog only; NOT
             part of the forecast)

    python3 forecast_products.py            (spacedisk defaults)
    python3 forecast_products.py --now 2026-11-19T18:00 ...   (replay)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import sys
import time
import warnings

import numpy as np

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
from donki_features import calibrated_transit_hours, parse_loc  # noqa: E402
from eval_harness import BIN_S, fetch  # noqa: E402

H = round(3600 / BIN_S)
ROOT = "/data/clear/Website_CLEAR/Website/RealTimeSW"
FH = ROOT + "/input/forecast_ih"
ROT_S = 360.0 / (14.1844 - 360.0 / 365.2422) * 86400     # as Keheng's scorer
LAT_H, AHEAD_H = 3.6, 24.0
VARS = ("v", "n", "b", "bz", "T")
FAST_KMS = 500.0
LABELS = ((400, "Calm"), (500, "Breezy"), (650, "Fast stream"), (1e9, "Very fast"))
CONF = ((100, "high"), (200, "medium"), (1e9, "low"))      # 80% band width, km/s
SLOTS_H = (0, 12)


def iso(k):
    return dt.datetime.utcfromtimestamp(k * BIN_S).strftime("%Y-%m-%dT%H:%M:%SZ")


def ubin_iso(s):
    return round(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp() / BIN_S)


def stamp_bin(stamp):
    return round(dt.datetime.strptime(stamp, "%y%m%dt%H%M").replace(
        tzinfo=dt.timezone.utc).timestamp() / BIN_S)


def r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def label(v):
    return next(name for lim, name in LABELS if v < lim)


def confidence(width):
    return next(name for lim, name in CONF if width < lim)


def read_csv_bins(path, key="time"):
    out = {}
    if not os.path.exists(path):
        return out
    for row in csv.DictReader(open(path)):
        k = round(dt.datetime.fromisoformat(row[key].replace("Z", "")).replace(
            tzinfo=dt.timezone.utc).timestamp() / BIN_S)
        out[k] = {c: float(v) for c, v in row.items() if c != key and v != ""}
    return out


def load_runs(a):
    """{run name: run} for every raw run, with its archived corrected and
    band files when they exist.  Keyed by NAME: two runs can start at the
    same time (a JAX run and a production rerun from the same restart time,
    e.g. both 261118t0014), and keying by start let one overwrite the other."""
    runs = {}
    for sat in glob.glob(os.path.join(a.forecasts, "RESULTS_*", "IH", "sat_earth_*.sat")):
        name = os.path.basename(os.path.dirname(os.path.dirname(sat)))
        stamp = name.split("_")[-1]
        raw = mv.load_forecast_all(sat)                 # {bin: (v, n, btotal, bz, T)}
        if not raw:
            continue
        b0 = min(raw)
        runs[name] = {"name": name, "src": name.split("_")[1], "sat": sat, "start": b0,
                    "raw": raw,
                    "jaf": read_csv_bins(os.path.join(a.jaf, name, f"jaf_{stamp}.csv")),
                    "band": read_csv_bins(os.path.join(a.band, name, f"jaf_band_{stamp}.csv"),
                                          key="time_utc")}
    return runs


def operational(runs, k, lat):
    """The run a forecaster would read for time k at 24 h ahead: the newest
    run in hand (start + latency) by k - 24 h that covers k."""
    best = None
    for run in runs.values():
        b0 = run["start"]
        if b0 + lat <= k - AHEAD_H * H and k <= b0 + 48 * H and k in run["jaf"]:
            if best is None or b0 > best["start"]:
                best = run
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", default=FH + "/forecasts")
    ap.add_argument("--jaf", default=FH + "/forecasts_jaf")
    ap.add_argument("--band", default=FH + "/forecasts_jaf_band")
    ap.add_argument("--series", default=ROOT + "/products/earth.json")
    ap.add_argument("--donki", default=ROOT + "/products/donki/events.json")
    ap.add_argument("--speed-model", default=os.path.join(HERE, "jaf_forecast_model_ops_20261119.json"))
    ap.add_argument("--multi-model", default=os.path.join(HERE, "jaf_multivar_model_ops_20261119.json"))
    ap.add_argument("--out", default=FH + "/site")
    ap.add_argument("--now", help="pretend it is this UTC time (testing)")
    a = ap.parse_args()

    now = ubin_iso(a.now) if a.now else round(time.time() / BIN_S)
    lat = round(LAT_H * H)
    doc = fetch(path=a.series)
    if a.now:   # replay: hide what didn't exist yet
        for key in ("obs", "model"):
            d = doc[key]
            keep = [i for i, t in enumerate(d["t"]) if round(t / BIN_S) <= now]
            for f in list(d):
                if isinstance(d[f], list) and len(d[f]) == len(d["t"]):
                    d[f] = [d[f][i] for i in keep]
    M = jaf_models.load(a.speed_model)
    MM = jaf_models.load(a.multi_model)
    fcx.FEATURES = M["features"]
    mv.OPTS.update(MM.get("opts", {}))
    S = fcx.Series(doc)
    G = mv.Grid(doc)
    obs = {x: fe.bin_map(doc["obs"]["t"], doc["obs"][x]) for x in VARS}

    runs = {nm: run for nm, run in load_runs(a).items() if run["start"] + lat <= now}
    if not runs:
        print("no run in hand; nothing written")
        return 1
    # newest run in hand; on a tie, production over the JAX backup
    newest = max(runs.values(), key=lambda r: (r["start"], r["src"] == "prod"))
    b0, run_end = newest["start"], newest["start"] + 48 * H

    # --- live forecast: newest run, observations up to now
    v, n, b, start = fe.load_forecast(newest["sat"])
    fc = {"name": newest["name"], "issue_bin": start, "v": v, "n": n, "b": b}
    corr = fa.correct_forecast(S, M, fc, mode="fresh", ahead_h=0.0, latency_h=LAT_H, now_bin=now)
    vspeed = {k: c[2] for k, c in corr.items()}
    targets, multi, full = mv.apply_run(G, MM["models"], newest["sat"], vspeed, lat, 0, now_bin=now)
    fseries = []
    for k in targets:
        if k < now or k not in corr or not full[k]:
            continue
        c, m, raw = corr[k], multi[k], newest["raw"][k]
        fseries.append({"t": iso(k), "v": r(m["v"]), "lo80": r(c[3]["80"][0]), "hi80": r(c[3]["80"][1]),
                        "lo50": r(c[3]["50"][0]), "hi50": r(c[3]["50"][1]),
                        "n": r(m["n"], 2), "b": r(m["b"], 2), "bz": r(m["bz"], 2), "T": r(m["T"], 0),
                        "raw_v": r(raw[0]), "raw_n": r(raw[1], 2), "raw_b": r(raw[2], 2),
                        "raw_bz": r(raw[3], 2), "raw_T": r(raw[4], 0)})

    # --- measured: last 24 h, and "now"
    measured = []
    for k in range(now - 24 * H, now + 1):
        if k in obs["v"]:
            measured.append({"t": iso(k), **{x: r(obs[x].get(k), 2) for x in VARS}})
    last_k = max(k for k in obs["v"] if k <= now)
    now_block = {"t": iso(last_k), "age_min": int((now - last_k) * BIN_S / 60),
                 **{x: r(obs[x].get(last_k), 2) for x in VARS}}

    # --- cards: 12 h blocks from now
    cards = []
    for j in range(4):
        blk = [p for p in fseries if now + j * 12 * H <= ubin_iso(p["t"][:-1]) < now + (j + 1) * 12 * H]
        if len(blk) < 6:
            continue
        vm = float(np.median([p["v"] for p in blk]))
        lo, hi = float(np.median([p["lo80"] for p in blk])), float(np.median([p["hi80"] for p in blk]))
        cards.append({"from": blk[0]["t"], "to": blk[-1]["t"], "v": r(vm, 0), "lo": r(lo, 0), "hi": r(hi, 0),
                      "label": label(vm), "confidence": confidence(hi - lo)})

    # --- next fast stream
    nf = {"status": "none"}
    if (obs["v"].get(last_k) or 0) >= FAST_KMS:
        nf = {"status": "now"}
    else:
        for j in range(len(fseries)):
            seg = fseries[j:j + 18]
            if len(seg) == 18 and all(p["v"] >= FAST_KMS for p in seg):
                peak = max(fseries[j:], key=lambda p: p["v"])
                early = next((p["t"] for p in fseries if p["hi80"] >= FAST_KMS), None)
                late = next((p["t"] for p in fseries if p["lo80"] >= FAST_KMS), None)
                nf = {"status": "coming", "arrival": seg[0]["t"], "peak_v": peak["v"], "peak_t": peak["t"],
                      "earliest": early, "latest": late}
                break

    # --- the operational 24 h-ahead series (archived corrected runs)
    def op_point(k):
        run = operational(runs, k, lat)
        if run is None:
            return None
        j, raw, bd = run["jaf"][k], run["raw"].get(k), run["band"].get(k, {})
        if raw is None:
            return None
        return run, j, raw, bd

    verify, miss_c, miss_r, inside, big = [], [], [], [], None
    for k in range(now - 48 * H, now + 1):
        op = op_point(k)
        o = obs["v"].get(k)
        if op is None:
            continue
        run, j, raw, bd = op
        verify.append({"t": iso(k), "measured": r(o), "corrected": r(j["v"]), "raw": r(raw[0]),
                       "lo80": r(bd.get("lo80")), "hi80": r(bd.get("hi80")), "run": run["name"]})
        if o is not None:
            miss_c.append(abs(j["v"] - o))
            miss_r.append(abs(raw[0] - o))
            if "lo80" in bd:
                inside.append(bd["lo80"] <= o <= bd["hi80"])
            if big is None or abs(j["v"] - o) > big[0]:
                big = (abs(j["v"] - o), k, j["v"] - o)
    verify_stats = {"mean_miss": r(np.mean(miss_c)) if miss_c else None,
                    "mean_miss_raw": r(np.mean(miss_r)) if miss_r else None,
                    "inside80": r(100 * np.mean(inside), 0) if inside else None,
                    "biggest": None if big is None else {"miss": r(big[0], 0), "t": iso(big[1]),
                                                         "direction": "too low" if big[2] < 0 else "too high"}}

    # --- scorecard: last 7 UTC days
    day0 = now - now % (24 * H)
    scorecard = []
    for d in range(6, -1, -1):
        lo_k, hi_k = day0 - d * 24 * H, day0 - d * 24 * H + 24 * H
        rows = []
        for k in range(lo_k, min(hi_k, now + 1)):
            op, o = op_point(k), obs["v"].get(k)
            if op is not None and o is not None:
                rows.append((o, op[2][0], op[1]["v"]))
        if len(rows) < 36:
            scorecard.append({"day": iso(lo_k)[:10], "n": len(rows)})
            continue
        A = np.array(rows)
        scorecard.append({"day": iso(lo_k)[:10], "n": len(rows), "measured": r(A[:, 0].mean(), 0),
                          "raw": r(A[:, 1].mean(), 0), "corrected": r(A[:, 2].mean(), 0),
                          "miss_raw": r(np.abs(A[:, 1] - A[:, 0]).mean(), 0),
                          "miss_corrected": r(np.abs(A[:, 2] - A[:, 0]).mean(), 0)})

    # --- last four runs (their archived corrected speed)
    last4 = []
    for run in sorted((r for r in runs.values() if r["jaf"]), key=lambda r: r["start"])[-4:]:
        s = run["start"]
        last4.append({"run": run["name"],
                      "start": dt.datetime.strptime(run["name"].split("_")[-1], "%y%m%dt%H%M").strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "v": [[iso(k), r(run["jaf"][k]["v"])] for k in sorted(run["jaf"])][::3]})

    # --- skill vs 27-day persistence, 24 h ahead
    rot = round(ROT_S / BIN_S)
    def skill(since):
        err = {x: [] for x in VARS}
        per = {x: [] for x in VARS}
        for k in range(since, now + 1):
            op = op_point(k)
            if op is None:
                continue
            run, j, raw, bd = op
            for x in VARS:
                o, p = obs[x].get(k), obs[x].get(k - rot)
                if o is None or p is None or x not in j:
                    continue
                err[x].append(j[x] - o)
                per[x].append(p - o)
        out = {}
        for x in VARS:
            if len(err[x]) >= 144:
                rm, rp = math.sqrt(np.mean(np.square(err[x]))), math.sqrt(np.mean(np.square(per[x])))
                out[x] = {"rmse": r(rm, 2), "rmse_persistence": r(rp, 2),
                          "skill_pct": r(100 * (1 - rm / rp), 0), "n": len(err[x])}
        return out
    skill_out = {"since_oct1": skill(ubin_iso("2026-10-01")), "last30d": skill(now - 30 * 24 * H)}

    # --- CME watch (DONKI mirror; catalog only)
    cmes = []
    try:
        ev = json.load(open(a.donki)).get("events", [])
    except (OSError, ValueError):
        ev = []
    for e in ev:
        if e.get("kind") != "CME" or not e.get("speed"):
            continue
        sp = float(e["speed"])
        cmd = parse_loc(e.get("loc", ""))
        if sp < 650 or (cmd is not None and abs(cmd) > 60):
            continue
        launch = dt.datetime.fromisoformat(e["time"].rstrip("Z")).replace(tzinfo=dt.timezone.utc).timestamp()
        arr = launch + calibrated_transit_hours(sp) * 3600
        if now * BIN_S - 12 * 3600 <= arr <= now * BIN_S + 72 * 3600:
            cmes.append({"launch": e["time"], "speed": r(sp, 0), "arrival_est": iso(round(arr / BIN_S)),
                         "location": e.get("loc") or "unknown", "link": e.get("link")})

    # --- meta
    hour = (now * BIN_S // 3600) % 24
    nxt = next((h for h in SLOTS_H if h > hour), None)
    nxt_s = (now * BIN_S // 86400) * 86400 + ((nxt if nxt is not None else 24 + SLOTS_H[0]) + 3) * 3600
    meta = {"generated": iso(round(time.time() / BIN_S)) if not a.now else iso(now), "now": iso(now),
            "run": newest["name"], "run_src": newest["src"], "run_end": iso(run_end),
            "run_start": dt.datetime.strptime(newest["name"].split("_")[-1], "%y%m%dt%H%M").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_to": iso(last_k), "next_update_after": iso(round(nxt_s / BIN_S)),
            "fast_kms": FAST_KMS, "labels": [[lim, name] for lim, name in LABELS[:-1]]}

    out = {"meta": meta, "now": now_block, "measured": measured, "forecast": fseries, "cards": cards,
           "next_fast": nf, "verify": verify, "verify_stats": verify_stats, "scorecard": scorecard,
           "runs": last4, "skill": skill_out, "cme": cmes}
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "forecast.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(a.out, "forecast.json"))
    tmp = os.path.join(a.out, "latest.csv.tmp")
    with open(tmp, "w") as f:
        cols = ["t", "v", "lo80", "hi80", "lo50", "hi50", "n", "b", "bz", "T", "raw_v", "raw_n", "raw_b", "raw_bz", "raw_T"]
        f.write("# CLEAR IH-only forecast at Earth, corrected (Justified Adaptive fluX). run "
                f"{newest['name']}, observations to {iso(last_k)}. v km/s, n cm^-3, b nT (incl. dB), "
                "bz nT GSM, T K; lo/hi = 80%/50% speed range; raw_* = uncorrected model.\n")
        f.write(",".join(cols) + "\n")
        for p in fseries:
            f.write(",".join("" if p[c] is None else str(p[c]) for c in cols) + "\n")
    os.replace(tmp, os.path.join(a.out, "latest.csv"))
    for p in ("forecast.json", "latest.csv"):
        try:
            os.chmod(os.path.join(a.out, p), 0o664)
        except OSError:
            pass
    print(f"[{dt.datetime.utcnow():%F %T}Z] forecast.json: run {newest['name']}, {len(fseries)} forecast "
          f"bins, {len(verify)} verify bins, {len(cmes)} CME(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
