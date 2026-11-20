#!/usr/bin/env python3
"""Hybrid / alternative forecast corrections vs Justified Adaptive fluX
(forecast), all causal, all choices made on Aug-Sep validation history,
scored on the real (held-out) Oct-Nov forecasts.

  jaf      the adopted version: per-lead ridge ("adaptive" features) mixed
           with the 0.5/0.5 blend by a validation-chosen weight
  analog   k-nearest past situations (standardized features), mean outcome
  hyb      g*jaf + (1-g)*analog, g per lead chosen on validation
  regime   jaf, but separate ridges for quiet (p0 < 450) / fast (>= 450)
           wind at the issue time
  jaf_bias jaf minus its own mean error over the trailing BIAS_DAYS of
           proxy targets that were verified before t0

    python forecast_hybrids.py --forecasts C:/.../AWSRT/forecast_ih
"""
from __future__ import annotations

import argparse
import datetime as dt
import pickle

import numpy as np

import forecast_correction as fcx
import forecast_eval as fe
from eval_harness import BIN_S
from phase2_model import ridge_fit, ridge_predict

H = fcx.H
K = 50
REGIME_KMS = 450.0
BIAS_DAYS = 14
GRID = np.linspace(0, 1, 21)


def ubin(day):
    return round(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() / BIN_S)


def knn_predict(Xtr, ytr, Xq, mean, std, k=K, chunk=400):
    A = (Xtr - mean) / std
    out = np.empty(len(Xq))
    for i in range(0, len(Xq), chunk):
        Q = (Xq[i:i + chunk] - mean) / std
        d = (Q * Q).sum(1)[:, None] - 2 * Q @ A.T + (A * A).sum(1)[None, :]
        idx = np.argpartition(d, k, axis=1)[:, :k]
        out[i:i + chunk] = ytr[idx].mean(1)
    return out


class Lead:
    """Everything fitted for one lead on targets before t_end."""

    def __init__(self, X, y, idx):
        self.mean, self.std = X.mean(0), X.std(0)
        self.std[self.std == 0] = 1
        self.X, self.y, self.i = X, y, idx
        self.rd = ridge_fit(X, y, fcx.RIDGE_ALPHA)
        q = X[:, idx["p0"]] >= REGIME_KMS
        self.rd_fast = ridge_fit(X[q], y[q], fcx.RIDGE_ALPHA) if q.sum() > 200 else self.rd
        self.rd_slow = ridge_fit(X[~q], y[~q], fcx.RIDGE_ALPHA) if (~q).sum() > 200 else self.rd

    def parts(self, Xq):
        i = self.i
        blend = 0.5 * Xq[:, i["raw"]] + 0.5 * Xq[:, i["p27"]]
        ridge = ridge_predict(Xq, *self.rd)
        q = Xq[:, i["p0"]] >= REGIME_KMS
        reg = np.where(q, ridge_predict(Xq, *self.rd_fast), ridge_predict(Xq, *self.rd_slow))
        ana = knn_predict(self.X, self.y, Xq, self.mean, self.std)
        return blend, ridge, reg, ana


def best_w(a, b, y):
    return GRID[int(np.argmin([np.mean(np.abs(g * a + (1 - g) * b - y)) for g in GRID]))]


def smooth(d, s=3):
    ks = sorted(d)
    v = np.array([d[k] for k in ks])
    return {k: v[max(0, j - s):j + s + 1].mean() for j, k in enumerate(ks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", required=True)
    ap.add_argument("--series", default="series_cache_20261119.json")
    ap.add_argument("--val-start", default="2026-08-01")
    ap.add_argument("--train-end", default="2026-10-01")
    ap.add_argument("--dump", default="rows_hybrids.pkl")
    a = ap.parse_args()
    fcx.FEATURES = fcx.FEATURE_SETS["adaptive"]
    idx = {k: j for j, k in enumerate(fcx.FEATURES)}
    S = fcx.Series(fe.load_series(a.series))
    v0, t_end = ubin(a.val_start), ubin(a.train_end)
    fcs = fe.load_forecasts(a.forecasts)
    last = max(fc["issue_bin"] for fc in fcs) + 48 * H

    # real-forecast feature rows, grouped by lead hour
    q = {L: [] for L in fcx.LEADS_H}
    for fc in fcs:
        at = fcx.forecast_model_at(S, fc)
        b0 = fc["issue_bin"]
        for t in sorted(fc["v"]):
            Lb = t - b0
            if Lb < 0 or Lb > 48 * H or t not in S.v:
                continue
            f = fcx.features(S, at, t, b0)
            if f is not None:
                q[min(48, round(Lb / H))].append((Lb / H, S.v[t], [f[k] for k in fcx.FEATURES],
                                                   fc["name"], b0))

    mix, wh, rows = {}, {}, []
    fits = {}
    for L in fcx.LEADS_H:
        # validation: fit on < val-start, choose weights on [val-start, train-end)
        Xt, yt, _ = fcx.training_rows(S, L, v0)
        Xv, yv, _ = fcx.training_rows(S, L, t_end, v0, step=6)
        m = Lead(Xt, yt, idx)
        bl, rd, rg, an = m.parts(Xv)
        mix[L] = best_w(rd, bl, yv)
        jv = mix[L] * rd + (1 - mix[L]) * bl
        wh[L] = best_w(jv, an, yv)
        # final fit on everything < train-end
        X, y, _ = fcx.training_rows(S, L, t_end)
        fits[L] = Lead(X, y, idx)
        print(f"lead {L:2d}: ridge mix {mix[L]:.2f}, jaf-vs-analog {wh[L]:.2f}", flush=True)
    mix, wh = smooth(mix), smooth(wh)

    # proxy predictions over the test span, for the rolling bias (targets
    # verified before each forecast's t0)
    proxy = {}
    for L in fcx.LEADS_H:
        Xp, yp, _, tp = fcx.training_rows(S, L, last, t_end - BIAS_DAYS * 24 * H, return_t=True)
        i = idx
        blp = 0.5 * Xp[:, i["raw"]] + 0.5 * Xp[:, i["p27"]]
        rdp = ridge_predict(Xp, *fits[L].rd)
        proxy[L] = (tp, mix[L] * rdp + (1 - mix[L]) * blp - yp)

    for L in fcx.LEADS_H:
        if not q[L]:
            continue
        Xq = np.array([r[2] for r in q[L]])
        bl, rd, rg, an = fits[L].parts(Xq)
        jaf = mix[L] * rd + (1 - mix[L]) * bl
        reg = mix[L] * rg + (1 - mix[L]) * bl
        hyb = wh[L] * jaf + (1 - wh[L]) * an
        tp, res = proxy[L]
        for j, r in enumerate(q[L]):
            b0 = r[4]
            sel = (tp < b0) & (tp >= b0 - BIAS_DAYS * 24 * H)
            bias = float(res[sel].mean()) if sel.sum() > 50 else 0.0
            raw = Xq[j, idx["raw"]]
            rows.append((r[0], r[1], {"raw": raw, "blend": bl[j], "jaf": jaf[j], "analog": an[j],
                                      "hyb": hyb[j], "regime": reg[j], "jaf_bias": jaf[j] - bias,
                                      "jaf_forecast": jaf[j]}, r[3]))
    pickle.dump({"rows": rows, "mix": mix, "wh": wh}, open(a.dump, "wb"))

    meths = ("raw", "blend", "jaf", "analog", "hyb", "regime", "jaf_bias")
    lead = np.array([r[0] for r in rows]); obs = np.array([r[1] for r in rows])
    print(f"\n{'lead h':>7} " + " ".join(f"{m:>11}" for m in meths) + "  (MAE/corr)")
    for lo, hi in ((0, 6), (6, 12), (12, 24), (24, 36), (36, 48), (0, 48)):
        msk = (lead >= lo) & ((lead < hi) if hi < 48 else (lead <= hi))
        cells = []
        for mm in meths:
            p = np.array([r[2][mm] for r, k in zip(rows, msk) if k])
            cells.append(f"{np.mean(np.abs(p - obs[msk])):5.1f}/{np.corrcoef(obs[msk], p)[0, 1]:.2f}")
        print(f"{lo:>3}-{hi:<3} " + " ".join(f"{c:>11}" for c in cells))
    print("\nchosen jaf-vs-analog weight by lead:",
          " ".join(f"{L}:{wh[L]:.2f}" for L in fcx.LEADS_H[::6]))


if __name__ == "__main__":
    main()
