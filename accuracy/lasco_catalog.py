#!/usr/bin/env python3
"""Fetch and parse the real, public SOHO/LASCO CME catalog
(cdaw.gsfc.nasa.gov/CME_list) -- the standard, continuous, near-sun CME
speed record since 1996, unlike DONKI which only starts ~2010. Used to
massively expand estimate_earth_vmax()'s calibration sample: Richardson &
Cane's own catalog (icme_raw.html) already records an exact LASCO CME
timestamp for 374 of its 633 verified ICME entries (see the "LASCO CME
Y/M/D (UT)" column, index 17 of the 18 real per-row cells) -- we do not
need to time-match anything ourselves, R&C already did that. We only need
to look up each referenced LASCO CME's own measured speed.

    python lasco_catalog.py    # builds and caches the (lasco_speed, v_max) pairs
"""
import datetime
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
from bs4 import BeautifulSoup

LASCO_URL = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/{y}_{m:02d}/univ{y}_{m:02d}.html"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "lasco_cache")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _get(url, timeout=30, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if i == retries - 1:
                raise
            time.sleep(2)


def fetch_month(y, m):
    """[(datetime_utc, linear_speed_kms_or_None), ...] for one LASCO monthly page."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{y}_{m:02d}.html")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            html = f.read()
    else:
        html = _get(LASCO_URL.format(y=y, m=m))
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(html)

    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue
        date_cell = tr.find("td", headers="hd_date_time")
        date_time_cells = tr.find_all("td", headers="hd_date_time")
        if len(date_time_cells) != 2:
            continue
        date_txt = date_time_cells[0].get_text(strip=True)
        time_txt = date_time_cells[1].get_text(strip=True)
        speed_cell = tr.find("td", headers="hd_v1")
        if speed_cell is None:
            continue
        speed_txt = speed_cell.get_text(strip=True)
        try:
            dt = datetime.datetime.strptime(f"{date_txt} {time_txt}", "%Y/%m/%d %H:%M:%S")
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        speed = None
        m_speed = re.search(r"-?\d+", speed_txt)
        if m_speed and "-" not in speed_txt.replace(m_speed.group(), ""):
            speed = float(m_speed.group())
        out.append((dt, speed))
    return out


def parse_rc_lasco_refs(path=os.path.join(os.path.dirname(__file__), "icme_raw.html")):
    """(v_max, lasco_dt) for every Richardson/Cane row with a usable LASCO
    CME reference -- takes the FIRST candidate date when a row lists
    several (ambiguous match), strips trailing halo/quality letter codes."""
    with open(path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    out = []
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
            dt = datetime.datetime.strptime(f"{date_part} {hhmm}", "%Y/%m/%d %H%M")
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        out.append((v_max, dt))
    return out


def build_pairs(tolerance_min=15):
    """(lasco_speed, v_max) for every R&C row whose referenced LASCO CME
    is found in the real catalog within tolerance_min of the recorded
    time (should be near-exact -- R&C recorded this timestamp FROM the
    LASCO catalog itself)."""
    refs = parse_rc_lasco_refs()
    print(f"{len(refs)} Richardson/Cane rows have a usable LASCO CME reference")

    months_needed = sorted({(dt.year, dt.month) for _, dt in refs})
    print(f"fetching {len(months_needed)} LASCO monthly catalog pages "
          f"({months_needed[0]} .. {months_needed[-1]})...")
    month_cache = {}
    for i, (y, m) in enumerate(months_needed):
        try:
            month_cache[(y, m)] = fetch_month(y, m)
        except Exception as e:
            print(f"  {y}-{m:02d}  FAILED ({type(e).__name__}: {e})")
            month_cache[(y, m)] = []
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(months_needed)} months fetched")

    pairs = []
    matched = unmatched = no_speed = 0
    for v_max, dt in refs:
        entries = month_cache.get((dt.year, dt.month), [])
        best, best_diff = None, None
        for edt, speed in entries:
            diff = abs((edt - dt).total_seconds()) / 60.0
            if diff <= tolerance_min and (best_diff is None or diff < best_diff):
                best, best_diff = speed, diff
        if best is None:
            # widen once to +/-60min for genuine near-misses (rare rounding
            # in either catalog's recorded minute) before giving up
            for edt, speed in entries:
                diff = abs((edt - dt).total_seconds()) / 60.0
                if diff <= 60.0 and (best_diff is None or diff < best_diff):
                    best, best_diff = speed, diff
        if best is None:
            unmatched += 1
            continue
        if best is None or np.isnan(best):
            no_speed += 1
            continue
        matched += 1
        pairs.append((best, v_max))

    print(f"\nmatched {matched}/{len(refs)} to a real LASCO catalog speed "
          f"({unmatched} not found in catalog, {no_speed} found but no usable speed value)")
    return np.array(pairs)


if __name__ == "__main__":
    pairs = build_pairs()
    out_path = os.path.join(os.path.dirname(__file__), "lasco_magnitude_pairs.npy")
    np.save(out_path, pairs)
    print(f"saved {len(pairs)} (lasco_speed, v_max) pairs to {out_path}")
    if len(pairs) > 2:
        corr = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
        print(f"correlation(lasco_speed, v_max) = {corr:.3f}  (n={len(pairs)})")
