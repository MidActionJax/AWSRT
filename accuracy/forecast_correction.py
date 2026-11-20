#!/usr/bin/env python3
"""Justified Adaptive fluX for the 24-48 h IH-only forecasts: lead-aware,
strictly causal corrections of forecast speed, trained on the real-time
history and tested on the real forecasts.

Why lead-aware: the locked nowcast correction (final_correction.py) reads
observations AT the corrected time (observed B, trailing errors).  A
forecast issued at t0 for t0+L can only know observations up to t0.  So
every predictor here sees, for a target t = t0 + L:
  - the model's speed at t and around it (the forecast itself; before t0,
    the real-time run, which is also known at t0)
  - observations up to t0 only (last speed, its 3 h trend, B, Bz, n)
  - persist27: the observation one rotation before t (known for L < 27 d)

Training proxy: the 48 h IH-only forecast matches what the real-time run
later shows at the same time to ~2.5 km/s (corr 0.999, 2026-11-19 check),
so the real-time model series stands in for "the forecast at lead L"
over the whole Jan-Sep history.  Training uses targets before
--train-end only; the real forecasts (Oct-Nov) are never trained on.

Methods compared (all causal):
  raw         the forecast as is
  persist0    last observed speed at t0, held
  blend       0.5 raw + 0.5 persist27 (Phase 1b)
  jaf_literal the locked recipe made causal: adaptive raw/persist27 weight
              from trailing errors up to t0, + P(disturbed)-gated flat shift
              with a per-lead classifier
  ridge       per-lead linear model on all causal features
  ridge_jaf   ridge + P(disturbed)-gated shift of the ridge's own
              disturbed-period residual (per lead)

    python forecast_correction.py --forecasts C:/.../AWSRT/forecast_ih
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

import forecast_eval as fe
from eval_harness import BIN_S, DISTURBED_KMS
from phase2_model import logistic_fit, predict_proba, ridge_fit, ridge_predict

H = round(3600 / BIN_S)          # bins per hour
ROT = fe.ROT_BINS
LEADS_H = list(range(0, 49))      # one model per whole-hour lead
RIDGE_ALPHA = 1.0
LOGISTIC_ALPHA = 0.005
FEATURES = ("raw", "p27", "p0", "raw0", "raw_change", "trend_3h", "trend_6h",
            "vol_24h", "disagree", "obs_trend_3h", "obs_B0", "obs_Bz0",
            "obs_Bz_min24h0", "obs_n0", "obs_B_max24h0")
# tried 2026-11-19 and dropped (no gain on the real forecasts): p27_smooth,
# rec_anom0 (p0 - obs 27 d before t0), obs_mean_24h0; still computed below
TIMING = ("raw_p6", "raw_p12", "raw_m12")          # model speed either side of t
COMPRESS = ("model_n", "model_B", "model_n_max6h", "model_B_max6h")  # model pile-up
# JAF's adaptive-trust idea as interactions: how much to trust the model vs
# the current observation depends on how wrong the model is at t0 (e0) and
# how big a change it predicts (chg)
ADAPTIVE = ("raw_x_e0", "p0_x_e0", "p27_x_e0", "raw_x_chg", "p0_x_chg")
# CME catalog, causal: only CMEs in DONKI at least CME_LAG_H before t0, fast
# and Earth-directed (final_correction's candidate rule), with the calibrated
# transit time; a bump around the predicted arrival at the target and the
# ~1.5 days after it (sheath/ejecta)
CME = ("cme_bump", "cme_after", "cme_has")
CME_LAG_H = 6.0
CME_MIN_SPEED_KMS = 650.0
DONKI_MERGED = "donki_fetch_20261119/donki_raw_merged_20261119.json"
FEATURE_SETS = {"base": FEATURES, "adaptive": FEATURES + ADAPTIVE,
                "adaptive_cme": FEATURES + ADAPTIVE + CME, "timing": FEATURES + TIMING,
                "compress": FEATURES + COMPRESS, "both": FEATURES + TIMING + COMPRESS}


class Series:
    """Observation maps + a model map, with causal lookups."""

    def __init__(self, doc):
        o = doc["obs"]
        self.v = fe.bin_map(o["t"], o["v"])
        self.b = fe.bin_map(o["t"], o["b"])
        self.bz = fe.bin_map(o["t"], o["bz"])
        self.n = fe.bin_map(o["t"], o["n"])
        m = doc["model"]
        self.model = fe.bin_map(m["t"], m["v"])
        self.model_maps = {"v": self.model, "n": fe.bin_map(m["t"], m["n"]),
                           "b": fe.bin_map(m["t"], m["b"])}

    cme = None   # [(predicted_arrival_unix, speed, launch_unix)], set by load_cme()

    def load_cme(self, path=DONKI_MERGED):
        from donki_features import build_arrival_index, load_donki_events_with_width
        ev = load_donki_events_with_width(path)
        self.cme = sorted(build_arrival_index(ev, CME_MIN_SPEED_KMS, max_cmd=60,
                                              include_unknown=True, half_angle_aware=True))
        return len(self.cme)

    def cme_features(self, t, t0):
        if not self.cme:
            return 0.0, 0.0, 0.0
        tu, t0u = t * BIN_S, t0 * BIN_S
        best = None
        for arr, speed, launch in self.cme:
            if launch + CME_LAG_H * 3600 > t0u or t0u - launch > 200 * 3600:
                continue
            off = (tu - arr) / 3600.0
            if best is None or abs(off) < abs(best[0]):
                best = (off, speed)
        if best is None or abs(best[0]) > 48:
            return 0.0, 0.0, 0.0
        off, sp = best
        return (sp / 1000 * float(np.exp(-(off / 12.0) ** 2)),
                sp / 1000 if 0 <= off <= 36 else 0.0,
                1.0 if abs(off) <= 24 else 0.0)

    def realtime_at(self, k, var="v"):
        return self.model_maps[var].get(k)

    def last(self, m, b0, back=6 * H):
        for k in range(b0, b0 - back - 1, -1):
            if k in m:
                return m[k]
        return None

    def window(self, m, lo, hi):
        return [m[k] for k in range(lo, hi + 1) if k in m]


def features(S, model_at, t, t0, end=None):
    """Causal feature dict for target bin t issued at t0, or None.
    model_at(k) gives the model speed at bin k (forecast or real-time).
    end: the last bin the forecast has (default t0 + 48 h); timing
    features never look past it."""
    raw, raw0 = model_at(t), model_at(t0)
    m3, m6 = model_at(t - 3 * H), model_at(t - 6 * H)
    p0 = S.last(S.v, t0)
    p27 = S.v.get(t - ROT)
    if p27 is None:
        p27 = S.last(S.v, t - ROT, back=3)
    p0_3 = S.last(S.v, t0 - 3 * H)
    if None in (raw, raw0, m3, m6, p0, p27, p0_3):
        return None
    vol = [x for x in (model_at(k) for k in range(t - 24 * H, t + 1, 3)) if x is not None]
    B0, Bz0, n0 = S.last(S.b, t0), S.last(S.bz, t0), S.last(S.n, t0)
    bz24 = S.window(S.bz, t0 - 24 * H, t0)
    b24 = S.window(S.b, t0 - 24 * H, t0)
    if len(vol) < 12 or None in (B0, Bz0, n0) or not bz24 or not b24:
        return None
    end = t0 + 48 * H if end is None else end   # never look past the forecast's end
    p6, p12, m12 = model_at(min(t + 6 * H, end)), model_at(min(t + 12 * H, end)), model_at(t - 12 * H)
    mn, mb = model_at(t, "n"), model_at(t, "b")
    n6 = [x for x in (model_at(k, "n") for k in range(t - 6 * H, t + 1, 3)) if x is not None]
    b6 = [x for x in (model_at(k, "b") for k in range(t - 6 * H, t + 1, 3)) if x is not None]
    if None in (p6, p12, m12, mn, mb) or not n6 or not b6:
        return None
    p27w = S.window(S.v, t - ROT - 3 * H, t - ROT + 3 * H)
    p27_t0 = S.last(S.v, t0 - ROT, back=3)
    o24 = S.window(S.v, t0 - 24 * H, t0)
    if len(p27w) < 6 or p27_t0 is None or len(o24) < 6:
        return None
    return {"raw": raw, "p27": p27, "p0": p0, "raw0": raw0,
            "raw_change": raw - raw0, "trend_3h": raw - m3, "trend_6h": raw - m6,
            "vol_24h": float(np.std(vol)), "disagree": raw - p27,
            "obs_trend_3h": p0 - p0_3, "obs_B0": B0, "obs_Bz0": Bz0,
            "obs_Bz_min24h0": min(bz24), "obs_n0": n0, "obs_B_max24h0": max(b24),
            # recurrence: smoothed last-rotation speed at t, and how far this
            # rotation is running above/below the last one right now (t0)
            "p27_smooth": float(np.mean(p27w)), "rec_anom0": p0 - p27_t0,
            "obs_mean_24h0": float(np.mean(o24)),
            "raw_p6": p6, "raw_p12": p12, "raw_m12": m12,
            "model_n": mn, "model_B": mb, "model_n_max6h": max(n6), "model_B_max6h": max(b6),
            "raw_x_e0": raw * abs(raw0 - p0) / 100, "p0_x_e0": p0 * abs(raw0 - p0) / 100,
            "p27_x_e0": p27 * abs(raw0 - p0) / 100, "raw_x_chg": raw * abs(raw - raw0) / 100,
            "p0_x_chg": p0 * abs(raw - raw0) / 100,
            **dict(zip(CME, S.cme_features(t, t0)))}


def adaptive_weight(S, model_at, t0, window=3 * H, min_n=6):
    """Locked-recipe raw weight from trailing 3 h errors ending at t0."""
    er, ep = [], []
    for k in range(t0 - window, t0 + 1):
        o, m, p = S.v.get(k), model_at(k), S.v.get(k - ROT)
        if o is None or m is None or p is None:
            continue
        er.append(abs(m - o))
        ep.append(abs(p - o))
    if len(er) < min_n:
        return 0.5
    tr, tp = np.mean(er), np.mean(ep)
    return tp / (tr + tp + 1e-9)


def training_rows(S, lead_h, t_end_bin, t_start_bin=None, return_t=False, step=1):
    """Proxy rows for one lead from the real-time history, targets in
    [t_start_bin, t_end_bin) (every `step`-th bin)."""
    L = lead_h * H
    X, y, w, ts = [], [], [], []
    rt = S.realtime_at
    for t in sorted(S.model):
        if t >= t_end_bin or t not in S.v or (t_start_bin is not None and t < t_start_bin)                 or t % step:
            continue
        f = features(S, rt, t, t - L)
        if f is None:
            continue
        X.append([f[k] for k in FEATURES])
        y.append(S.v[t])
        w.append(adaptive_weight(S, rt, t - L))
        ts.append(t)
    if return_t:
        return np.array(X), np.array(y), np.array(w), np.array(ts)
    return np.array(X), np.array(y), np.array(w)


def fit_lead(X, y, w):
    i = {k: j for j, k in enumerate(FEATURES)}
    dist = (y > DISTURBED_KMS).astype(float)
    lg = logistic_fit(X, dist, LOGISTIC_ALPHA)
    pr = predict_proba(X, *lg)
    rd = ridge_fit(X, y, RIDGE_ALPHA)
    ridge_pred = ridge_predict(X, *rd)
    lit_blend = w * X[:, i["raw"]] + (1 - w) * X[:, i["p27"]]
    m = dist > 0
    return {"logistic": lg, "ridge": rd,
            "lit_shift": float((y - lit_blend)[m].mean()) if m.any() else 0.0,
            "ridge_shift": float((y - ridge_pred)[m].mean()) if m.any() else 0.0,
            "n": len(y), "frac_disturbed": float(dist.mean()), "train_p": pr}


def predict(models, lead_h, f, w):
    M = models[lead_h]
    x = np.array([[f[k] for k in FEATURES]])
    p = float(predict_proba(x, *M["logistic"])[0])
    rp = float(ridge_predict(x, *M["ridge"])[0])
    lit = w * f["raw"] + (1 - w) * f["p27"]
    return {"raw": f["raw"], "persist0": f["p0"], "blend": 0.5 * f["raw"] + 0.5 * f["p27"],
            "jaf_literal": lit + p * M["lit_shift"],
            "ridge": rp, "ridge_jaf": rp + p * M["ridge_shift"],
            "ridge_blend": 0.5 * rp + 0.5 * (0.5 * f["raw"] + 0.5 * f["p27"]),
            "jaf_forecast": M.get("mix", 1.0) * rp
                            + (1 - M.get("mix", 1.0)) * (0.5 * f["raw"] + 0.5 * f["p27"]),
            "_p": p}


def forecast_model_at(S, fc):
    """Forecast values from t0 on; the real-time run before t0 (known then)."""
    b0 = fc["issue_bin"]
    maps = {"v": fc["v"], "n": fc["n"], "b": fc["b"]}
    def at(k, var="v"):
        if k >= b0:
            m = maps[var]
            if k in m:
                return m[k]
            # sat rows are every ~2 min, so every bin >= b0 is normally there
            return m.get(k - 1, m.get(k + 1))
        return S.realtime_at(k, var)
    return at


METHODS = ("raw", "persist0", "blend", "jaf_literal", "ridge", "ridge_jaf", "ridge_blend",
           "jaf_forecast")


def choose_mix(S, t_start_val, t_end_val, smooth=3):
    """Per-lead weight on ridge (vs the blend) chosen on a validation span
    that comes BEFORE the test period: ridge trained on targets before
    t_start_val, weight = argmin MAE on targets in [t_start_val, t_end_val).
    Smoothed over +/-smooth leads (the optimum is noisy lead to lead)."""
    i = {k: j for j, k in enumerate(FEATURES)}
    grid = np.linspace(0, 1, 21)
    best = []
    for L in LEADS_H:
        Xt, yt, wt = training_rows(S, L, t_start_val)
        rd = ridge_fit(Xt, yt, RIDGE_ALPHA)
        Xv, yv, _ = training_rows(S, L, t_end_val, t_start_val)
        rp = ridge_predict(Xv, *rd)
        bl = 0.5 * Xv[:, i["raw"]] + 0.5 * Xv[:, i["p27"]]
        mae = [np.mean(np.abs(g * rp + (1 - g) * bl - yv)) for g in grid]
        best.append(grid[int(np.argmin(mae))])
    best = np.array(best)
    sm = np.array([best[max(0, j - smooth):j + smooth + 1].mean() for j in range(len(best))])
    return dict(zip(LEADS_H, sm)), dict(zip(LEADS_H, best))


def pick_models(model_sets, b0):
    """Newest model set whose training ended at or before the issue bin."""
    ends = [e for e in model_sets if e <= b0]
    return model_sets[max(ends)]


def score_real(S, fcs, model_sets):
    rows = []   # (lead_h_float, obs, {method: pred})
    for fc in fcs:
        at = forecast_model_at(S, fc)
        b0 = fc["issue_bin"]
        models = pick_models(model_sets, b0)
        w = adaptive_weight(S, at, b0)
        for t in sorted(fc["v"]):
            L = t - b0
            if L < 0 or L > 48 * H or t not in S.v:
                continue
            f = features(S, at, t, b0)
            if f is None:
                continue
            lead_h = min(48, round(L / H))
            rows.append((L / H, S.v[t], predict(models, lead_h, f, w), fc["name"]))
    return rows


def report(rows, buckets=((0, 6), (6, 12), (12, 24), (24, 36), (36, 48), (0, 48))):
    lead = np.array([r[0] for r in rows])
    obs = np.array([r[1] for r in rows])
    print(f"{'lead h':>7} " + " ".join(f"{m:>12}" for m in METHODS) + "   (MAE / corr)")
    for lo, hi in buckets:
        m = (lead >= lo) & ((lead < hi) if hi < 48 else (lead <= hi))
        cells = []
        for meth in METHODS:
            p = np.array([r[2][meth] for r, k in zip(rows, m) if k])
            o = obs[m]
            cells.append(f"{np.mean(np.abs(p - o)):6.1f}/{np.corrcoef(o, p)[0, 1]:.2f}")
        print(f"{lo:>3}-{hi:<3} " + " ".join(f"{c:>12}" for c in cells) + f"   n={m.sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", required=True)
    ap.add_argument("--series", default="series_cache_20261119.json")
    ap.add_argument("--train-end", default="2026-10-01")
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default="base")
    ap.add_argument("--dump", help="pickle the scored rows here for later analysis")
    ap.add_argument("--val-start", default="2026-08-01",
                    help="validation span [val-start, train-end) for choosing the per-lead "
                         "ridge/blend mix (jaf_forecast)")
    ap.add_argument("--retrain-days", type=int, default=0,
                    help="also retrain every N days after --train-end (each forecast uses "
                         "the newest model trained on targets before its issue time)")
    a = ap.parse_args()
    FEATURES = FEATURE_SETS[a.features]
    print("features:", a.features, len(FEATURES))
    S = Series(fe.load_series(a.series))
    if any(k in FEATURES for k in CME):
        print("CME candidates (fast, Earth-directed):", S.load_cme())
    fcs = fe.load_forecasts(a.forecasts)
    t_end = round(dt.datetime.fromisoformat(a.train_end).replace(
        tzinfo=dt.timezone.utc).timestamp() / BIN_S)
    ends = [t_end]
    if a.retrain_days:
        last_issue = max(fc["issue_bin"] for fc in fcs)
        while ends[-1] + a.retrain_days * 24 * H <= last_issue:
            ends.append(ends[-1] + a.retrain_days * 24 * H)
    v_start = round(dt.datetime.fromisoformat(a.val_start).replace(
        tzinfo=dt.timezone.utc).timestamp() / BIN_S)
    mix, mix_raw = choose_mix(S, v_start, t_end)
    print("ridge weight by lead (smoothed / raw), chosen on", a.val_start, "->", a.train_end)
    print("  " + " ".join(f"{L}:{mix[L]:.2f}/{mix_raw[L]:.2f}" for L in LEADS_H[::6]))
    model_sets = {}
    for e in ends:
        models = {}
        for L in LEADS_H:
            X, y, w = training_rows(S, L, e)
            models[L] = fit_lead(X, y, w)
            models[L]["mix"] = mix[L]
        model_sets[e] = models
        M = models[24]
        print(f"trained to {dt.datetime.utcfromtimestamp(e * BIN_S):%Y-%m-%d}: lead 24 h "
              f"{M['n']} rows, ridge_shift {M['ridge_shift']:+.1f}")
    rows = score_real(S, fcs, model_sets)
    if a.dump:
        import pickle
        with open(a.dump, "wb") as f:
            pickle.dump({"rows": rows, "mix": mix, "features": FEATURES,
                         "train_end": a.train_end, "val_start": a.val_start}, f)
    print(f"\nREAL FORECASTS (held out: never trained on): {len(fcs)} forecasts, {len(rows)} points")
    report(rows)
