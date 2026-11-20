#!/usr/bin/env python3
"""Before designing any mechanism: does LASCO's near-sun speed (known at
launch, before the disturbance arrives) actually predict HELIO4CAST's
richer structure (Bzmin, duration) on the doubly-matched event set? If
not, there's no genuine lead-time value to build a feature around --
better to find that out now than after wiring something in.

Reuses:
  - lasco_catalog.parse_rc_lasco_refs() -- (v_max, lasco_dt) per R&C row
  - lasco_catalog.fetch_month() -- cached LASCO monthly pages -> speed
  - helio4cast_matched.pkl -- built by helio4cast_match.py

    python helio4cast_diagnose.py
"""
import datetime
import pickle
import re

import numpy as np
from bs4 import BeautifulSoup

from lasco_catalog import fetch_month
from lasco_width_check import fetch_month_width


def parse_rc_triples(path="icme_raw.html"):
    """(rc_disturbance_dt, v_max, lasco_dt) for every real row with a
    usable LASCO reference -- same parsing as lasco_catalog.
    parse_rc_lasco_refs(), but keeping the disturbance date too so this
    can join to helio4cast_matched.pkl's rc_date exactly, not by
    re-matching on v_max (which can collide across rows)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) != 18:
            continue
        rc_date_txt = cells[0].get_text(strip=True)
        v_max_txt = cells[12].get_text(strip=True)
        lasco_txt = cells[17].get_text(strip=True)
        if not lasco_txt or lasco_txt in ("...", "-", "----"):
            continue
        try:
            rc_dt = datetime.datetime.strptime(rc_date_txt, "%Y/%m/%d %H%M").replace(tzinfo=datetime.timezone.utc)
            v_max = float(v_max_txt)
        except ValueError:
            continue
        first = re.split(r"[,\t]", lasco_txt)[0].strip()
        m = re.match(r"(\d{4}/\d{2}/\d{2})\s+(\d{4})", first)
        if not m:
            continue
        date_part, hhmm = m.groups()
        try:
            lasco_dt = datetime.datetime.strptime(f"{date_part} {hhmm}", "%Y/%m/%d %H%M").replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        out.append((rc_dt, v_max, lasco_dt))
    return out


def main():
    with open("helio4cast_matched.pkl", "rb") as f:
        matched = pickle.load(f)
    print(f"{len(matched)} R&C rows matched to a real HELIO4CAST Wind entry")

    triples = parse_rc_triples()
    triple_by_date = {rc_dt: (v_max, lasco_dt) for rc_dt, v_max, lasco_dt in triples}

    rows = []
    cache = {}
    for m in matched:
        rc_dt = m["rc_date"]
        if rc_dt not in triple_by_date:
            continue
        v_max, lasco_dt = triple_by_date[rc_dt]
        key = (lasco_dt.year, lasco_dt.month)
        if key not in cache:
            cache[key] = fetch_month_width(*key)
        entries = cache[key]
        best, best_width, best_diff = None, None, None
        for edt, speed, width in entries:
            diff = abs((edt - lasco_dt).total_seconds()) / 60.0
            if diff <= 60 and (best_diff is None or diff < best_diff):
                best, best_width, best_diff = speed, width, diff
        if best is None:
            continue
        rows.append({"lasco_speed": best, "width": best_width, "v_max": v_max,
                      "duration_h": m["duration_h"], "bzmin": m["bzmin"],
                      "bzmean": m["bzmean"], "bmax": m["bmax"]})

    print(f"{len(rows)} rows with BOTH a real LASCO near-sun speed AND HELIO4CAST structure\n")

    lasco = np.array([r["lasco_speed"] for r in rows])
    for field in ("v_max", "duration_h", "bzmin", "bzmean", "bmax"):
        vals = np.array([r[field] for r in rows if r[field] is not None])
        ls = np.array([r["lasco_speed"] for r in rows if r[field] is not None])
        if len(vals) < 15:
            print(f"lasco_speed -> {field:12s}: n={len(vals):3d} too few")
            continue
        corr = np.corrcoef(ls, vals)[0, 1]
        print(f"lasco_speed -> {field:12s}: n={len(vals):3d}  corr={corr:+.3f}")

    print()
    width_rows = [r for r in rows if r["width"] is not None]
    print(f"{len(width_rows)}/{len(rows)} rows also have a usable angular-width value")
    for field in ("v_max", "duration_h", "bzmin", "bzmean", "bmax"):
        vals = np.array([r[field] for r in width_rows if r[field] is not None])
        ws = np.array([r["width"] for r in width_rows if r[field] is not None])
        if len(vals) < 15:
            print(f"width       -> {field:12s}: n={len(vals):3d} too few")
            continue
        corr = np.corrcoef(ws, vals)[0, 1]
        print(f"width       -> {field:12s}: n={len(vals):3d}  corr={corr:+.3f}")

    import pickle as pkl
    with open("helio4cast_lasco_pairs.pkl", "wb") as f:
        pkl.dump(rows, f)
    print(f"\nsaved {len(rows)} full rows to helio4cast_lasco_pairs.pkl")


if __name__ == "__main__":
    main()
