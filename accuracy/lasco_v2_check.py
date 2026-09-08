#!/usr/bin/env python3
"""Cheap, no-pipeline-risk check before doing anything else: does either of
LASCO's two 2nd-order speed measurements (at final height, at 20 Rs --
these account for CME acceleration/deceleration near the Sun, unlike the
linear-fit speed already used) correlate with V_max at Earth better than
the linear speed does? Uses the already-cached raw HTML from
lasco_catalog.py's run -- no new network fetches.

Reuses parse_rc_lasco_refs() for the (v_max, lasco_dt) side; re-parses the
cached monthly pages here for the two 2nd-order columns specifically
(fetch_month() in lasco_catalog.py only kept the linear one).

    python lasco_v2_check.py
"""
import os
import re

import numpy as np
from bs4 import BeautifulSoup

from lasco_catalog import CACHE_DIR, parse_rc_lasco_refs


def _parse_speed(cell):
    if cell is None:
        return None
    txt = cell.get_text(strip=True)
    m = re.search(r"-?\d+", txt)
    if not m or "-" in txt.replace(m.group(), ""):
        return None
    return float(m.group())


def fetch_month_v2(y, m):
    """[(datetime_utc, v1, v2_final, v2_20rs), ...] for one cached month."""
    path = os.path.join(CACHE_DIR, f"{y}_{m:02d}.html")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    out = []
    import datetime as dtmod
    for tr in soup.find_all("tr"):
        date_time_cells = tr.find_all("td", headers="hd_date_time")
        if len(date_time_cells) != 2:
            continue
        date_txt = date_time_cells[0].get_text(strip=True)
        time_txt = date_time_cells[1].get_text(strip=True)
        try:
            dt = dtmod.datetime.strptime(f"{date_txt} {time_txt}", "%Y/%m/%d %H:%M:%S")
            dt = dt.replace(tzinfo=dtmod.timezone.utc)
        except ValueError:
            continue
        v1_cell = tr.find("td", headers="hd_v1")
        v2_cells = tr.find_all("td", headers="hd_v2")
        v1 = _parse_speed(v1_cell)
        v2f = _parse_speed(v2_cells[0]) if len(v2_cells) > 0 else None
        v2t = _parse_speed(v2_cells[1]) if len(v2_cells) > 1 else None
        out.append((dt, v1, v2f, v2t))
    return out


def main():
    refs = parse_rc_lasco_refs()
    months_needed = sorted({(dt.year, dt.month) for _, dt in refs})
    month_cache = {(y, m): fetch_month_v2(y, m) for y, m in months_needed}

    pairs_v1, pairs_v2f, pairs_v2t = [], [], []
    for v_max, dt in refs:
        entries = month_cache.get((dt.year, dt.month), [])
        best = None
        best_diff = None
        for edt, v1, v2f, v2t in entries:
            diff = abs((edt - dt).total_seconds()) / 60.0
            if diff <= 60.0 and (best_diff is None or diff < best_diff):
                best, best_diff = (v1, v2f, v2t), diff
        if best is None:
            continue
        v1, v2f, v2t = best
        if v1 is not None:
            pairs_v1.append((v1, v_max))
        if v2f is not None:
            pairs_v2f.append((v2f, v_max))
        if v2t is not None:
            pairs_v2t.append((v2t, v_max))

    for name, pairs in (("linear (already used)", pairs_v1),
                        ("2nd-order @ final height", pairs_v2f),
                        ("2nd-order @ 20 Rs", pairs_v2t)):
        if len(pairs) < 10:
            print(f"{name:28s}: n={len(pairs):4d} -- too few to trust")
            continue
        arr = np.array(pairs)
        corr = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
        print(f"{name:28s}: n={len(pairs):4d}  corr={corr:.3f}  "
              f"speed range {arr[:,0].min():.0f}-{arr[:,0].max():.0f}")


if __name__ == "__main__":
    main()
