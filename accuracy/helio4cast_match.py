#!/usr/bin/env python3
"""Tier 3 item 9, phase 2: match HELIO4CAST's real, actively-maintained
ICMECAT (Wind entries -- same L1 vantage point as our own obs data) to
the R&C/LASCO-matched candidates already built in lasco_catalog.py, by
date. Both catalogs describe the SAME real ICME occurrences at Earth
(independently curated by different research groups), so this is a
close date match, not a fresh heuristic candidate-selection problem --
avoids re-introducing the automated-matching pitfall this project hit
repeatedly in Tier 0 (0a/0b/0c).

Richardson & Cane's own "Disturbance Y/M/D (UT)" date (icme_raw.html,
column 0) is the shock/disturbance ONSET; HELIO4CAST's icme_start_time is
also the ICME/shock onset at the same spacecraft -- expect these to be
close (same physical event, different manual analysis), not identical to
the minute, so match with a real tolerance window.

    python helio4cast_match.py
"""
import datetime
import json
import urllib.request

import numpy as np
from bs4 import BeautifulSoup

ICMECAT_URL = "https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.json"
MATCH_TOLERANCE_H = 24  # generous but bounded -- independent manual onset calls can differ by hours


def fetch_icmecat_wind():
    req = urllib.request.Request(ICMECAT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    keys = sorted(data["icmecat_id"].keys(), key=lambda x: int(x))
    out = []
    for k in keys:
        if data["sc_insitu"][k] != "Wind":
            continue
        t = data["icme_start_time"][k]
        if not t:
            continue
        try:
            dt = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M%z") if "+" in t or t.endswith("Z") is False \
                else datetime.datetime.strptime(t.rstrip("Z"), "%Y-%m-%dT%H:%M").replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        out.append({
            "id": data["icmecat_id"][k], "start": dt,
            "duration_h": data["mo_duration"][k], "bmax": data["mo_bmax"][k],
            "bmean": data["mo_bmean"][k], "bzmin": data["mo_bzmin"][k],
            "bzmean": data["mo_bzmean"][k], "speed_mean": data["mo_speed_mean"][k],
            "density_mean": data["mo_density_mean"][k],
        })
    return out


def parse_rc_disturbance_dates(path="icme_raw.html"):
    """(disturbance_start_dt, v_max) for every real 18-cell R&C row --
    reused convention from climatology_mc.py's column mapping."""
    with open(path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) != 18:
            continue
        date_txt = cells[0].get_text(strip=True)
        v_max_txt = cells[12].get_text(strip=True)
        try:
            dt = datetime.datetime.strptime(date_txt, "%Y/%m/%d %H%M").replace(tzinfo=datetime.timezone.utc)
            v_max = float(v_max_txt)
        except ValueError:
            continue
        out.append((dt, v_max))
    return out


def main():
    print("Fetching HELIO4CAST ICMECAT v23 (Wind entries)...")
    wind_events = fetch_icmecat_wind()
    print(f"  {len(wind_events)} real Wind ICME entries, "
          f"{wind_events[0]['start'].date()} .. {wind_events[-1]['start'].date()}")

    print("\nParsing Richardson & Cane's own disturbance dates (already have LASCO matches for these)...")
    rc_events = parse_rc_disturbance_dates()
    print(f"  {len(rc_events)} R&C rows total")

    print(f"\nMatching by date, tolerance +/-{MATCH_TOLERANCE_H}h...")
    matched, unmatched = [], 0
    for rc_dt, v_max in rc_events:
        best, best_diff = None, None
        for ev in wind_events:
            diff_h = abs((ev["start"] - rc_dt).total_seconds()) / 3600.0
            if diff_h <= MATCH_TOLERANCE_H and (best_diff is None or diff_h < best_diff):
                best, best_diff = ev, diff_h
        if best is None:
            unmatched += 1
            continue
        matched.append({"rc_date": rc_dt, "v_max": v_max, "diff_h": best_diff, **best})

    print(f"  matched {len(matched)}/{len(rc_events)} R&C rows to a real HELIO4CAST Wind entry")
    print(f"  ({unmatched} unmatched -- either not in HELIO4CAST's Wind subset, outside its "
          f"1995-2025 coverage, or a bigger date gap than {MATCH_TOLERANCE_H}h)")

    diffs = np.array([m["diff_h"] for m in matched])
    print(f"  match time-diff: median {np.median(diffs):.1f}h, "
          f"mean {diffs.mean():.1f}h, max {diffs.max():.1f}h "
          f"({int((diffs<=1).sum())}/{len(diffs)} within 1h -- confirms these are the same real events)")

    import pickle
    with open("helio4cast_matched.pkl", "wb") as f:
        pickle.dump(matched, f)
    print(f"\nsaved {len(matched)} matched events to helio4cast_matched.pkl")

    # quick sanity check: does duration/structure correlate with v_max at all?
    durs = np.array([m["duration_h"] for m in matched if m["duration_h"] is not None])
    vmaxs = np.array([m["v_max"] for m in matched if m["duration_h"] is not None])
    if len(durs) > 10:
        corr = np.corrcoef(durs, vmaxs)[0, 1]
        print(f"\nquick check: corr(mo_duration, v_max) = {corr:.3f}  (n={len(durs)})")
    bzmins = np.array([m["bzmin"] for m in matched if m["bzmin"] is not None])
    vmaxs2 = np.array([m["v_max"] for m in matched if m["bzmin"] is not None])
    if len(bzmins) > 10:
        corr = np.corrcoef(bzmins, vmaxs2)[0, 1]
        print(f"quick check: corr(mo_bzmin, v_max) = {corr:.3f}  (n={len(bzmins)})")


if __name__ == "__main__":
    main()
