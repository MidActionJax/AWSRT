#!/usr/bin/env python3
"""Speed doesn't predict duration/Bz structure (helio4cast_diagnose.py).
Before giving up on a richer-input mechanism: does LASCO's ANGULAR WIDTH
-- a genuinely different measurement, physically tied to how large/complex
the eruption is, not how fast it's moving -- predict what speed couldn't?
Reuses the already-cached LASCO monthly pages, no new network fetches.

    python lasco_width_check.py
"""
import os
import pickle
import re

import numpy as np
from bs4 import BeautifulSoup

from lasco_catalog import CACHE_DIR


def fetch_month_width(y, m):
    """[(datetime_utc, linear_speed, width_deg), ...] for one cached month."""
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

        def parse_num(cell):
            if cell is None:
                return None
            txt = cell.get_text(strip=True)
            m2 = re.search(r"-?\d+", txt)
            if not m2 or "-" in txt.replace(m2.group(), ""):
                return None
            return float(m2.group())

        v1 = parse_num(tr.find("td", headers="hd_v1"))
        width = parse_num(tr.find("td", headers="hd_wd"))
        out.append((dt, v1, width))
    return out


def main():
    with open("helio4cast_lasco_pairs.pkl", "rb") as f:
        rows = pickle.load(f)
    print(f"{len(rows)} rows with lasco_speed + helio4cast structure already matched")

    # need the LASCO datetime for each row to look up its width -- re-derive
    # via the same R&C triple join used in helio4cast_diagnose.py
    import datetime
    from bs4 import BeautifulSoup as BS

    def parse_rc_triples(path="icme_raw.html"):
        with open(path, encoding="utf-8", errors="replace") as f:
            soup = BS(f.read(), "html.parser")
        out = {}
        for tr in soup.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) != 18:
                continue
            v_max_txt = cells[12].get_text(strip=True)
            lasco_txt = cells[17].get_text(strip=True)
            if not lasco_txt or lasco_txt in ("...", "-", "----"):
                continue
            try:
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
            out.setdefault(round(v_max, 1), []).append(lasco_dt)
        return out

    triples = parse_rc_triples()
    cache = {}
    full_rows = []
    for r in rows:
        candidates = triples.get(round(r["v_max"], 1), [])
        if not candidates:
            continue
        lasco_dt = candidates[0]
        key = (lasco_dt.year, lasco_dt.month)
        if key not in cache:
            cache[key] = fetch_month_width(*key)
        entries = cache[key]
        best_w, best_diff = None, None
        for edt, v1, width in entries:
            diff = abs((edt - lasco_dt).total_seconds()) / 60.0
            if diff <= 60 and v1 is not None and abs(v1 - r["lasco_speed"]) < 1.0 and \
                    (best_diff is None or diff < best_diff):
                best_w, best_diff = width, diff
        if best_w is None:
            continue
        full_rows.append({**r, "width": best_w})

    print(f"{len(full_rows)} rows with width also resolved\n")
    widths = np.array([r["width"] for r in full_rows])
    for field in ("v_max", "duration_h", "bzmin", "bzmean", "bmax"):
        vals = np.array([r[field] for r in full_rows if r[field] is not None])
        ws = np.array([r["width"] for r in full_rows if r[field] is not None])
        if len(vals) < 15:
            print(f"width -> {field:12s}: n={len(vals):3d} too few")
            continue
        corr = np.corrcoef(ws, vals)[0, 1]
        print(f"width -> {field:12s}: n={len(vals):3d}  corr={corr:+.3f}")

    with open("helio4cast_lasco_width_pairs.pkl", "wb") as f:
        pickle.dump(full_rows, f)


if __name__ == "__main__":
    main()
