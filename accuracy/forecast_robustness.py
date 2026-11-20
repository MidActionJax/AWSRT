#!/usr/bin/env python3
"""Per-forecast robustness of the forecast corrections (reads the rows
forecast_correction.py --dump wrote).

  - each forecast's own 0-48 h MAE: raw vs blend vs jaf_forecast, and how
    many forecasts each one wins
  - the scores Keheng will compute: speed at exactly 24 h and 48 h lead
    (the point within +-30 min of that lead, one per forecast)

    python forecast_robustness.py rows_base.pkl
"""
import collections
import pickle
import sys

import numpy as np

COMPARE = ("raw", "blend", "jaf_forecast")


def main(path):
    d = pickle.load(open(path, "rb"))
    rows = d["rows"]
    by = collections.defaultdict(list)
    for lead, obs, pred, name in rows:
        by[name].append((lead, obs, pred))

    print(f"{'forecast':26} " + " ".join(f"{m:>12}" for m in COMPARE) + "   (0-48 h MAE, km/s)")
    per = {m: [] for m in COMPARE}
    for name in sorted(by):
        r = by[name]
        o = np.array([x[1] for x in r])
        cells = []
        for m in COMPARE:
            e = float(np.mean(np.abs(np.array([x[2][m] for x in r]) - o)))
            per[m].append(e)
            cells.append(f"{e:12.1f}")
        print(f"{name:26} " + " ".join(cells))
    raw, bl, jf = (np.array(per[m]) for m in COMPARE)
    n = len(raw)
    print(f"\njaf_forecast better than raw on {np.sum(jf < raw)}/{n} forecasts, "
          f"better than blend on {np.sum(jf < bl)}/{n}")
    print(f"median per-forecast MAE: raw {np.median(raw):.1f}, blend {np.median(bl):.1f}, "
          f"jaf_forecast {np.median(jf):.1f}; worst: raw {raw.max():.1f}, "
          f"blend {bl.max():.1f}, jaf_forecast {jf.max():.1f}")

    for target in (24, 48):
        pts = {}
        for name, r in by.items():
            c = [x for x in r if abs(x[0] - target) <= 0.5]
            if c:
                pts[name] = min(c, key=lambda x: abs(x[0] - target))
        o = np.array([p[1] for p in pts.values()])
        print(f"\nAt {target} h lead ({len(pts)} forecasts, one point each):")
        for m in COMPARE:
            p = np.array([x[2][m] for x in pts.values()])
            e = p - o
            print(f"  {m:13} MAE {np.mean(np.abs(e)):6.1f}  RMSE {np.sqrt(np.mean(e * e)):6.1f}  "
                  f"bias {np.mean(e):+6.1f}  corr {np.corrcoef(o, p)[0, 1]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
