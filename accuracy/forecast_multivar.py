#!/usr/bin/env python3
"""Justified Adaptive fluX (forecast) for the other variables: density n,
total |B| (SWMF's btotal, sqrt(B^2 + dB^2), the way Keheng scores it), Bz
and temperature T -- same recipe as the speed correction, vectorized:

  per lead hour 0..48, a ridge on causal features (model at/around the
  target, observations up to the issue time, the observation one solar
  rotation before the target, speed context, and the adaptive-trust
  interactions), mixed with the blend 0.5 model + 0.5 persist27 by a
  per-lead weight chosen on Aug-Sep by RMSE; final fit on everything
  before --train-end.  Trained on the real-time history (the IH-only
  forecast reproduces it), applied to the real forecasts with FRESH
  observations: target t is corrected with issue = max(run start +
  latency, t - 24 h), as a forecaster reading the 24 h-ahead value would.

Writes, per run, MY_DIR/RESULTS_<src>_<stamp>/jaf_<stamp>.csv (time, v,
n, b, bz, T) for score_ih_forecasts.py: v is the speed correction (read
from --speed-dir's jaf_band files), the rest from here.

    python forecast_multivar.py --forecasts .../forecast_ih \\
        --speed-dir .../forecasts_jaf_fresh24 --out .../forecasts_jaf_multi
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import math
import os

import numpy as np

import forecast_eval as fe
from eval_harness import BIN_S
from phase2_model import ridge_fit, ridge_predict

H = round(3600 / BIN_S)
ROT = fe.ROT_BINS
LEADS = list(range(0, 49))
VARS = ("n", "b", "bz", "T")
GRID = np.linspace(0, 1, 21)
ALPHA = 1.0


def ubin(day):
    return round(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() / BIN_S)


def ffill(a, limit):
    """Last finite value within `limit` bins back (inclusive)."""
    out = np.full_like(a, np.nan)
    last, age = np.nan, limit + 1
    for i, x in enumerate(a):
        if np.isfinite(x):
            last, age = x, 0
        else:
            age += 1
        if age <= limit:
            out[i] = last
    return out


def shift(a, s):
    """out[i] = a[i - s]"""
    out = np.full_like(a, np.nan)
    if s >= 0:
        out[s:] = a[:len(a) - s]
    else:
        out[:s] = a[-s:]
    return out


def roll(a, w, how):
    """Trailing window of w bins ending at i (inclusive), NaN-aware."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.full(w - 1, np.nan), a])
    v = sliding_window_view(pad, w)
    with np.errstate(all="ignore"):
        cnt = np.isfinite(v).sum(1)
        r = np.nanmean(v, 1) if how == "mean" else np.nanmax(v, 1)
    r[cnt < max(1, w // 6)] = np.nan
    return r


class Grid:
    def __init__(self, doc):
        o, m = doc["obs"], doc["model"]
        ks = [round(t / BIN_S) for t in o["t"]] + [round(t / BIN_S) for t in m["t"]]
        self.g0 = min(ks) - ROT - 48 * H
        self.n = max(ks) - self.g0 + 1 + 48 * H
        self.obs = {x: self._arr(o["t"], o[x if x != "v" else "v"]) for x in ("v",) + VARS}
        mb = self._arr(m["t"], m["b"])
        mdb = self._arr(m["t"], m["db"])
        self.model = {"v": self._arr(m["t"], m["v"]), "n": self._arr(m["t"], m["n"]),
                      "b": np.hypot(mb, np.nan_to_num(mdb)), "bz": self._arr(m["t"], m["bz"]),
                      "T": self._arr(m["t"], m["T"])}
        add_max6(self.model)
        self.ff = {x: ffill(self.obs[x], 6 * H) for x in self.obs}
        self.p27 = {x: shift(ffill(self.obs[x], 3), ROT) for x in self.obs}
        self.m24 = {x: roll(self.obs[x], 24 * H, "mean") for x in self.obs}

    def _arr(self, t, v):
        a = np.full(self.n, np.nan)
        for ti, vi in zip(t, v):
            if vi is not None:
                a[round(ti / BIN_S) - self.g0] = vi
        return a

    def idx(self, b):
        return b - self.g0


def add_max6(model):
    for x in ("v", "n", "b", "bz", "T"):
        model[x + "_max6"] = roll(model[x], 6 * H, "max")


OPTS = {"log": False, "cross": False}   # variants, judged on history (validate())


def feats(G, model, x, t, i):
    """Feature matrix for targets t (grid indices) issued at i; model is
    a dict of model arrays (real-time, or per forecast)."""
    mx, mv = model[x], model["v"]
    mxmax6 = model[x + "_max6"]
    rX, rX0 = mx[t], mx[i]
    pX0, pX27, mX24 = G.ff[x][i], G.p27[x][t], G.m24[x][i]
    rmax6 = mxmax6[t]
    rv, rv0, pv0 = mv[t], mv[i], G.ff["v"][i]
    e0, chg = np.abs(rX0 - pX0), np.abs(rX - rX0)
    cols = [rX, rX0, pX0, pX27, mX24, rmax6, rv, rv0, pv0, rv - rv0,
            rX * e0, pX0 * e0, pX27 * e0, rX * chg, pX0 * chg]
    if OPTS["cross"]:
        # the other plasma variables as measured at the issue time, and the
        # model's at the target: n, |B|, T and v co-vary (compression)
        for o in ("n", "b", "T"):
            if o != x:
                cols += [G.ff[o][i], model[o][t]]
    F = np.column_stack(cols)
    return F, rX, pX27


def to_fit(x, y):
    """Fit in log space for the positive, heavy-tailed variables."""
    return np.log(np.maximum(y, 1e-3)) if OPTS["log"] and x in ("n", "T") else y


def from_fit(x, y):
    return np.exp(y) if OPTS["log"] and x in ("n", "T") else y


def validate(G, x, t_tr, t_mix, t_end, leads=range(0, 49, 6)):
    """History-only check of a variant: ridge on targets < t_tr, mix
    chosen on [t_tr, t_mix), RMSE on [t_mix, t_end) -- no test forecast
    involved. Returns mean RMSE over the leads, and per lead."""
    T_all = np.arange(48 * H + 1, G.n)
    tb = T_all + G.g0
    per = {}
    for L in leads:
        t = T_all
        F, rX, p27 = feats(G, G.model, x, t, t - L * H)
        y = G.obs[x][t]
        ok = np.isfinite(F).all(1) & np.isfinite(y) & (y > 0 if x in ("n", "T") else True)
        tr, mx, te = ok & (tb < t_tr), ok & (tb >= t_tr) & (tb < t_mix), ok & (tb >= t_mix) & (tb < t_end)
        Ft = F[tr] if not (OPTS["log"] and x in ("n", "T")) else F[tr]
        rd = ridge_fit(F[tr], to_fit(x, y[tr]), ALPHA)
        bl = lambda m: 0.5 * rX[m] + 0.5 * p27[m]
        w = best_w(from_fit(x, ridge_predict(F[mx], *rd)), bl(mx), y[mx])
        pred = w * from_fit(x, ridge_predict(F[te], *rd)) + (1 - w) * bl(te)
        per[L] = (float(np.sqrt(np.mean((pred - y[te]) ** 2))),
                  float(np.sqrt(np.mean((bl(te) - y[te]) ** 2))),
                  float(np.sqrt(np.mean((G.p27[x][t][te] - y[te]) ** 2))))
    return np.mean([v[0] for v in per.values()]), per


def best_w(a, b, y):
    return GRID[int(np.argmin([np.sqrt(np.mean((g * a + (1 - g) * b - y) ** 2)) for g in GRID]))]


def smooth(d, s=3):
    v = np.array([d[L] for L in LEADS])
    return {L: v[max(0, j - s):j + s + 1].mean() for j, L in enumerate(LEADS)}


def fit_var(G, x, v0, t_end):
    """Per lead: ridge on targets < t_end, mix chosen on [v0, t_end)."""
    T_all = np.arange(48 * H + 1, G.n)
    tb = T_all + G.g0
    mix, fits = {}, {}
    for L in LEADS:
        t = T_all
        i = t - L * H
        F, rX, p27 = feats(G, G.model, x, t, i)
        y = G.obs[x][t]
        ok = np.isfinite(F).all(1) & np.isfinite(y)
        tr, va = ok & (tb < v0), ok & (tb >= v0) & (tb < t_end)
        if x in ("n", "T"):
            tr, va = tr & (y > 0), va & (y > 0)
        rd = ridge_fit(F[tr], to_fit(x, y[tr]), ALPHA)
        mix[L] = best_w(from_fit(x, ridge_predict(F[va], *rd)), 0.5 * rX[va] + 0.5 * p27[va], y[va])
        al = ok & (tb < t_end) & ((y > 0) if x in ("n", "T") else True)
        fits[L] = ridge_fit(F[al], to_fit(x, y[al]), ALPHA)
    return fits, smooth(mix)


def load_forecast_all(path):
    """sat file -> {bin: (v, n, btotal, bz, T)} (bin medians)."""
    rows = {}
    with open(path) as f:
        for line in f:
            q = line.split()
            if len(q) != 21 or not q[1].isdigit():
                continue
            t = dt.datetime(int(q[1]), int(q[2]), int(q[3]), int(q[4]), int(q[5]), int(q[6]),
                            tzinfo=dt.timezone.utc).timestamp()
            ux, uy, uz = (float(q[j]) for j in (12, 13, 14))
            bx, by, bz = (float(q[j]) * 1e5 for j in (15, 16, 17))
            w = max((float(q[19]) + float(q[20])) * 0.1, 0.0)
            bt = math.hypot(math.sqrt(bx * bx + by * by + bz * bz), math.sqrt(4e-7 * math.pi * w) * 1e9)
            rows.setdefault(round(t / BIN_S), []).append(
                (math.sqrt(ux * ux + uy * uy + uz * uz), float(q[11]), bt, bz, float(q[18])))
    return {k: np.median(np.array(v), 0) for k, v in rows.items()}


def read_speed(speed_dir, name):
    p = glob.glob(os.path.join(speed_dir, name, "IH", "jaf_band_*.csv"))
    if not p:
        return {}
    out = {}
    for r in csv.DictReader(open(p[0])):
        k = round(dt.datetime.fromisoformat(r["time_utc"].replace("Z", "")).replace(
            tzinfo=dt.timezone.utc).timestamp() / BIN_S)
        out[k] = float(r["jaf_v"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", required=True)
    ap.add_argument("--speed-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--series", default="series_cache_20261119.json")
    ap.add_argument("--val-start", default="2026-08-01")
    ap.add_argument("--train-end", default="2026-10-01")
    ap.add_argument("--latency-h", type=float, default=3.6)
    ap.add_argument("--ahead-h", type=float, default=24.0)
    ap.add_argument("--log", action="store_true", help="fit n and T in log space")
    ap.add_argument("--cross", action="store_true", help="add the other variables as features")
    ap.add_argument("--save-models", metavar="PKL",
                    help="fit on targets < --train-end, save the models, exit (no files written)")
    ap.add_argument("--validate", action="store_true",
                    help="history-only comparison of the variants, then exit")
    a = ap.parse_args()
    G = Grid(fe.load_series(a.series))
    if a.validate:
        t_tr, t_mix, t_e = ubin("2026-08-01"), ubin("2026-09-01"), ubin("2026-10-01")
        for x in VARS:
            res = {}
            for log in (False, True):
                for cross in (False, True):
                    if log and x not in ("n", "T"):
                        continue
                    OPTS["log"], OPTS["cross"] = log, cross
                    m, per = validate(G, x, t_tr, t_mix, t_e)
                    res[(log, cross)] = m
                    tag = ("log " if log else "lin ") + ("+cross" if cross else "      ")
                    print(f"{x:3} {tag}: Sep RMSE mean over leads {m:9.3f}   (24 h: {per[24][0]:.3f}, "
                          f"48 h: {per[48][0]:.3f}; blend 24 h {per[24][1]:.3f}, persist27 24 h {per[24][2]:.3f})",
                          flush=True)
        return
    OPTS["log"], OPTS["cross"] = a.log, a.cross
    v0, t_end = ubin(a.val_start), ubin(a.train_end)
    if a.save_models:
        import pickle
        models = {x: fit_var(G, x, v0, t_end) for x in VARS}
        pickle.dump({"models": models, "opts": dict(OPTS), "train_end": a.train_end,
                     "val_start": a.val_start, "vars": VARS}, open(a.save_models, "wb"))
        print("saved", a.save_models)
        return
    v0, t_end = ubin(a.val_start), ubin(a.train_end)
    models = {}
    for x in VARS:
        models[x] = fit_var(G, x, v0, t_end)
        print(f"{x}: blend weight on ridge at 0/12/24/36/48 h: " +
              " ".join(f"{models[x][1][L]:.2f}" for L in (0, 12, 24, 36, 48)), flush=True)

    lat, ahead = round(a.latency_h * H), round(a.ahead_h * H)
    names = sorted(os.path.basename(os.path.dirname(os.path.dirname(p))) for p in
                   glob.glob(os.path.join(a.forecasts, "RESULTS_*", "IH", "sat_earth_*.sat")))
    for name in names:
        sat = glob.glob(os.path.join(a.forecasts, name, "IH", "sat_earth_*.sat"))[0]
        targets, out, _ = apply_run(G, models, sat, read_speed(a.speed_dir, name), lat, ahead)
        stamp = name.split("_")[-1]
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        write_csv(os.path.join(a.out, name, f"jaf_{stamp}.csv"), targets, out)
        print(f"{name}: {len(targets)} bins", flush=True)


def apply_run(G, models, sat, vspeed, lat, ahead, now_bin=None):
    """Correct n, b, bz, T of one run (fresh observations).  vspeed: the
    speed correction {bin: v}.  Returns (targets, {bin: {var: value}},
    {bin: True if every variable was corrected there, not left raw}).
    now_bin: never issue after now (live use, see forecast_apply)."""
    fc = load_forecast_all(sat)
    b0 = min(fc)
    model = {}
    for j, x in enumerate(("v", "n", "b", "bz", "T")):
        model[x] = G.model[x].copy()
        model[x][G.idx(b0):] = np.nan          # nothing from after the run's start...
        for k, vals in fc.items():               # ...except the forecast itself
            model[x][G.idx(k)] = vals[j]
    add_max6(model)
    targets = [k for k in sorted(fc) if b0 + lat <= k <= b0 + 48 * H]
    kk = np.array(targets)
    issue = np.maximum(b0 + lat, kk - ahead)
    if now_bin is not None:
        keep = np.minimum(issue, now_bin) >= b0 + lat
        targets = [k for k, ok in zip(targets, keep) if ok]
        kk, issue = kk[keep], np.minimum(issue[keep], now_bin)
    out = {k: {"v": vspeed.get(k, fc[k][0])} for k in targets}
    full = {k: k in vspeed for k in targets}
    if not targets:
        return targets, out, full
    Ls = np.minimum(48, np.round((kk - issue) / H).astype(int))
    t, i = kk - G.g0, issue - G.g0
    for x in VARS:
        fits, mix = models[x]
        F, rX, p27 = feats(G, model, x, t, i)
        raw = np.array([fc[k][("v", "n", "b", "bz", "T").index(x)] for k in targets])
        pred = raw.copy()
        ok = np.isfinite(F).all(1)
        for L in np.unique(Ls):
            m = (Ls == L) & ok
            if m.any():
                pred[m] = mix[L] * from_fit(x, ridge_predict(F[m], *fits[L])) + (1 - mix[L]) * (0.5 * rX[m] + 0.5 * p27[m])
        if x in ("n", "b", "T"):
            pred = np.maximum(pred, 0.05 * np.abs(raw))
        for j, k in enumerate(targets):
            out[k][x] = float(pred[j])
            full[k] = full[k] and bool(ok[j])
    return targets, out, full


def write_csv(path, targets, out):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("time,v,n,b,bz,T\n")
        for k in targets:
            tt = dt.datetime.utcfromtimestamp(k * BIN_S).strftime("%Y-%m-%dT%H:%M:%SZ")
            r = out[k]
            f.write(f"{tt},{r['v']:.2f},{r['n']:.4f},{r['b']:.4f},{r['bz']:.4f},{r['T']:.1f}\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
