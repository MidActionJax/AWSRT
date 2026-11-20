#!/usr/bin/env python3
"""Confidence band for Justified Adaptive fluX (forecast): how far off is
today's corrected forecast likely to be, lead by lead.

  1. Out-of-sample residuals of the adopted correction on the Jan-Sep
     history by MONTHLY cross-fitting: for each month, the per-lead ridge
     is refit on all OTHER months (before --train-end) and scored on that
     month.  The ridge/blend mix stays the adopted, validation-chosen one.
  2. Per lead, an error-size model: ridge of |residual| on uncertainty
     signals known at t0 (UNC below), floored at MIN_SCALE.
  3. Band = prediction + per-lead quantiles of residual / predicted size
     (pooled over +-POOL leads), so it is asymmetric where the errors are
     (e.g. the forecast's underprediction bias).
  Tested on the real Oct-Nov forecasts: coverage, width, and whether the
  width flags the bad forecasts.

    python forecast_confidence.py --forecasts C:/.../AWSRT/forecast_ih
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
UNC = ("e0", "chg", "vol_24h", "obs_trend_abs", "p0", "raw", "disagree_abs", "obs_B_max24h0")
MIN_SCALE = 5.0
POOL = 3
BANDS = {"80": (0.10, 0.90), "50": (0.25, 0.75)}


def ubin(day):
    return round(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() / BIN_S)


def unc_matrix(X, idx):
    g = lambda k: X[:, idx[k]]
    return np.column_stack([np.abs(g("raw0") - g("p0")), np.abs(g("raw") - g("raw0")),
                            g("vol_24h"), np.abs(g("obs_trend_3h")), g("p0"), g("raw"),
                            np.abs(g("disagree")), g("obs_B_max24h0")])


def jaf_predict(rd, mix, X, idx):
    return mix * ridge_predict(X, *rd) + (1 - mix) * (0.5 * X[:, idx["raw"]] + 0.5 * X[:, idx["p27"]])


def month_of(tbins):
    return np.array([dt.datetime.utcfromtimestamp(t * BIN_S).month for t in tbins])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", required=True)
    ap.add_argument("--series", default="series_cache_20261119.json")
    ap.add_argument("--mix-from", default="rows_adaptive.pkl",
                    help="the adopted run's dump (its per-lead mix is reused)")
    ap.add_argument("--train-end", default="2026-10-01")
    ap.add_argument("--dump", default="rows_confidence.pkl")
    ap.add_argument("--model-out", default="jaf_forecast_model.pkl")
    a = ap.parse_args()
    fcx.FEATURES = fcx.FEATURE_SETS["adaptive"]
    idx = {k: j for j, k in enumerate(fcx.FEATURES)}
    mix = pickle.load(open(a.mix_from, "rb"))["mix"]
    S = fcx.Series(fe.load_series(a.series))
    t_end = ubin(a.train_end)
    fcs = fe.load_forecasts(a.forecasts)

    # real-forecast rows by lead hour
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
                                                   fc["name"]))

    models, ratios = {}, {}
    for L in fcx.LEADS_H:
        X, y, _, tb = fcx.training_rows(S, L, t_end, return_t=True)
        mon = month_of(tb)
        res = np.empty(len(y))
        for m in np.unique(mon):
            k = mon == m
            rd = ridge_fit(X[~k], y[~k], fcx.RIDGE_ALPHA)
            res[k] = y[k] - jaf_predict(rd, mix[L], X[k], idx)
        U = unc_matrix(X, idx)
        sc = ridge_fit(U, np.abs(res), 1.0)
        s_hat = np.maximum(ridge_predict(U, *sc), MIN_SCALE)
        models[L] = {"rd": ridge_fit(X, y, fcx.RIDGE_ALPHA), "sc": sc,
                     "cv_mae": float(np.mean(np.abs(res)))}
        ratios[L] = res / s_hat
        if L % 12 == 0 and L > 0:
            print(f"lead {L:2d}: cross-fit MAE {models[L]['cv_mae']:.1f}, corr(|res|, size) "
                  f"{np.corrcoef(np.abs(res), s_hat)[0, 1]:.2f}", flush=True)
    quant = {}
    for L in fcx.LEADS_H:
        pool = np.concatenate([ratios[k] for k in fcx.LEADS_H if abs(k - L) <= POOL])
        quant[L] = {b: np.quantile(pool, qs) for b, qs in BANDS.items()}

    rows = []
    for L in fcx.LEADS_H:
        if not q[L]:
            continue
        Xq = np.array([r[2] for r in q[L]])
        pred = jaf_predict(models[L]["rd"], mix[L], Xq, idx)
        s_hat = np.maximum(ridge_predict(unc_matrix(Xq, idx), *models[L]["sc"]), MIN_SCALE)
        for j, r in enumerate(q[L]):
            band = {b: (pred[j] + quant[L][b][0] * s_hat[j], pred[j] + quant[L][b][1] * s_hat[j])
                    for b in BANDS}
            rows.append({"lead": r[0], "obs": r[1], "pred": pred[j], "size": s_hat[j],
                         "band": band, "name": r[3], "raw": Xq[j, idx["raw"]]})
    pickle.dump({"rows": rows, "quant": quant, "mix": mix}, open(a.dump, "wb"))
    # the frozen model forecast_apply.py uses
    pickle.dump({"features": fcx.FEATURES, "unc": UNC, "min_scale": MIN_SCALE,
                 "train_end": a.train_end, "mix": mix, "quant": quant, "bands": BANDS,
                 "leads": {L: {"rd": models[L]["rd"], "sc": models[L]["sc"]} for L in fcx.LEADS_H}},
                open(a.model_out, "wb"))
    print("saved model to", a.model_out)

    # --- scoring on the real forecasts
    lead = np.array([r["lead"] for r in rows])
    obs = np.array([r["obs"] for r in rows])
    print(f"\nREAL FORECASTS: {len(rows)} points")
    print(f"{'lead h':>7} {'MAE':>6} {'in80%':>6} {'w80':>6} {'in50%':>6} {'w50':>6}")
    for lo, hi in ((0, 6), (6, 12), (12, 24), (24, 36), (36, 48), (0, 48)):
        m = (lead >= lo) & ((lead < hi) if hi < 48 else (lead <= hi))
        rr = [r for r, k in zip(rows, m) if k]
        cov = {b: np.mean([r["band"][b][0] <= r["obs"] <= r["band"][b][1] for r in rr]) for b in BANDS}
        wid = {b: np.mean([r["band"][b][1] - r["band"][b][0] for r in rr]) for b in BANDS}
        mae = np.mean([abs(r["pred"] - r["obs"]) for r in rr])
        print(f"{lo:>3}-{hi:<3} {mae:6.1f} {cov['80']:6.2f} {wid['80']:6.1f} {cov['50']:6.2f} {wid['50']:6.1f}")

    # does the band's width flag the bad forecasts?
    names = sorted({r["name"] for r in rows})
    per = []
    for n in names:
        rr = [r for r in rows if r["name"] == n]
        per.append((n, np.mean([r["band"]["80"][1] - r["band"]["80"][0] for r in rr]),
                    np.mean([abs(r["pred"] - r["obs"]) for r in rr])))
    w = np.array([p[1] for p in per]); e = np.array([p[2] for p in per])
    rank = lambda x: np.argsort(np.argsort(x))
    print(f"\nper forecast: corr(mean 80% width, actual MAE) = {np.corrcoef(w, e)[0, 1]:.2f}, "
          f"rank corr = {np.corrcoef(rank(w), rank(e))[0, 1]:.2f}")
    worst = sorted(per, key=lambda p: -p[2])[:8]
    wr = {n: i + 1 for i, (n, _, _) in enumerate(sorted(per, key=lambda p: -p[1]))}
    print("worst 8 forecasts by actual error, with their width rank (1 = widest band):")
    for n, wi, ei in worst:
        print(f"  {n:26} MAE {ei:6.1f}  mean 80% width {wi:6.1f}  width rank {wr[n]}/{len(per)}")


if __name__ == "__main__":
    main()
