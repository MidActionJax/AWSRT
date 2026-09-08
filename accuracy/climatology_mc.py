#!/usr/bin/env python3
"""Tier 2 item 6: redo the event climatology stratified by Richardson/Cane's
own magnetic-cloud (MC) classification instead of raw event duration.

Duration-stratification (climatology.py) proved the *shape* insight was
real -- a naive blanket composite washes out because a 20-min spike and a
74h extended event share no common profile -- but every magnitude-scaling
mechanism on top of it failed the same way (see accuracy_improvement_
tracking.txt). MC classification is a physically different split: true
magnetic clouds (smooth, coherent, rotating field) vs. other ICMEs (messier,
more variable) are a real physical distinction, not just a duration proxy,
so they may compose into more coherent, more usefully-scaled profiles.

icme_raw.html's real column layout (verified directly, not assumed --
header labels don't map 1:1 to <td> cells since several labels span two
columns each):
    0 disturbance start (Y/M/D HHMM), ... 14 "MC?" (0/1/2/2H)
MC codes: 2 = clear MC, 2H = MC-like for half the interval, 1 = possible/
weak MC, 0 = not an MC. Buckets here: "mc" (2, 2H), "non_mc" (0); "1" is
left out of both -- it's an ambiguous middle category, not clearly either.

    python climatology_mc.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from bs4 import BeautifulSoup

from eval_harness import BIN_S, fetch as fetch_live
import phase2_model as p2
from climatology import (ONSET_HOURS_BEFORE, PROFILE_HOURS_AFTER,
                          bin_omni, build_composite, find_events)

ICME_HTML = "icme_raw.html"
MATCH_TOLERANCE_H = 18   # our own threshold-crossing start vs. the catalog's
                         # assigned disturbance start won't line up exactly


def parse_mc_catalog(path=ICME_HTML):
    """(start_datetime, mc_code) for every real data row in the catalog."""
    with open(path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) != 18:
            continue   # header rows (repeated periodically) have 15
        date_txt = cells[0].get_text(strip=True)
        mc_txt = cells[14].get_text(strip=True)
        try:
            dt = datetime.strptime(date_txt, "%Y/%m/%d %H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            continue   # a few rows have partial/uncertain timestamps
        out.append((dt, mc_txt))
    return out


def bucket_by_mc(events, catalog, tolerance_h=MATCH_TOLERANCE_H):
    """Match each (start_b, end_b, peak) OMNI event to the nearest catalog
    entry within tolerance_h and bucket by its MC code. Events with no
    catalog match within tolerance, or matching the ambiguous "1" code,
    are dropped rather than guessed into a bucket."""
    cat_times = [dt for dt, _ in catalog]
    out = {"mc": [], "non_mc": []}
    unmatched = 0
    for start_b, end_b, peak in events:
        start_dt = datetime.fromtimestamp(start_b * BIN_S, tz=timezone.utc)
        best_dt_diff, best_code = None, None
        for dt, code in catalog:
            diff_h = abs((dt - start_dt).total_seconds()) / 3600
            if diff_h <= tolerance_h and (best_dt_diff is None or diff_h < best_dt_diff):
                best_dt_diff, best_code = diff_h, code
        if best_code is None:
            unmatched += 1
            continue
        if best_code in ("2", "2H"):
            out["mc"].append((start_b, end_b, peak))
        elif best_code == "0":
            out["non_mc"].append((start_b, end_b, peak))
        # code "1" (ambiguous) deliberately excluded from both buckets
    print(f"  matched {len(events) - unmatched}/{len(events)} events to the "
          f"catalog within {tolerance_h}h; {len(out['mc'])} MC, "
          f"{len(out['non_mc'])} non-MC, rest ambiguous/unmatched")
    return out


def score_bucket(name, profile, offsets_h, events_2026, obs_bins, blend_by_bin):
    before_bins = round(ONSET_HOURS_BEFORE * 3600 / BIN_S)
    results_obs, results_clim = [], []
    for start_b, end_b, peak in events_2026:
        if start_b not in obs_bins:
            continue
        baseline = obs_bins.get(start_b - before_bins)
        if baseline is None:
            continue
        hist_baseline = profile[before_bins]
        for i, bi in enumerate(range(start_b - before_bins,
                                      start_b + round(PROFILE_HOURS_AFTER * 3600 / BIN_S) + 1)):
            if bi not in obs_bins or bi not in blend_by_bin:
                continue
            clim_pred = baseline + (profile[i] - hist_baseline)
            results_obs.append(obs_bins[bi])
            results_clim.append(clim_pred)
    if not results_obs:
        print(f"  {name:8s}: no scoreable rows")
        return
    s = p2.stats(np.array(results_obs), np.array(results_clim))
    print(f"  {name:8s}: n={s['n']:5d}  rmse={s['rmse']:6.1f}  "
          f"mae={s['mae']:6.1f}  bias={s['bias']:7.1f}  corr={s['corr']:6.3f}")


def main():
    print("Loading 2023-2024 OMNI history and the Richardson/Cane catalog...")
    hist_bins = bin_omni("omni_2023_2024.npz")
    hist_events = find_events(hist_bins, threshold=600.0)
    catalog = parse_mc_catalog()
    print(f"  {len(hist_events)} OMNI events (600+ km/s), "
          f"{len(catalog)} catalog entries total\n")

    print("Matching events to MC classification...")
    buckets = bucket_by_mc(hist_events, catalog)
    print()

    print("Loading 2026 test data...")
    doc = fetch_live()
    model, obs = doc["model"], doc["obs"]
    from eval_harness import to_bin_map
    model_bins = to_bin_map(model["t"], model["v"])
    obs_bins = to_bin_map(obs["t"], obs["v"])
    events_2026 = p2.event_windows(obs_bins, threshold=600.0)
    bins_arr, data = p2.build_dataset(model_bins, obs_bins)
    blend_by_bin = dict(zip(bins_arr, data["blend_v"]))
    print(f"  {len(events_2026)} 2026 event windows to test against\n")

    print("Per-bucket composites, scored against the same 2026 events:")
    for name, evs in buckets.items():
        if len(evs) < 3:
            print(f"  {name:8s}: too few events ({len(evs)}) to build a composite")
            continue
        offsets_h, profile, n_used = build_composite(hist_bins, evs)
        if profile is None:
            print(f"  {name:8s}: no usable composite (all events failed coverage check)")
            continue
        print(f"  {name:8s}: composite from {n_used}/{len(evs)} usable events, "
              f"peak {np.nanmax(profile):.0f} km/s")
        score_bucket(name, profile, offsets_h, events_2026, obs_bins, blend_by_bin)
        print()

    print("For reference, the Phase 1b blend on the same rows:")
    before_bins = round(ONSET_HOURS_BEFORE * 3600 / BIN_S)
    results_obs, results_blend = [], []
    for start_b, end_b, peak in events_2026:
        if start_b not in obs_bins or (start_b - before_bins) not in obs_bins:
            continue
        for bi in range(start_b - before_bins,
                         start_b + round(PROFILE_HOURS_AFTER * 3600 / BIN_S) + 1):
            if bi in obs_bins and bi in blend_by_bin:
                results_obs.append(obs_bins[bi])
                results_blend.append(blend_by_bin[bi])
    s_blend = p2.stats(np.array(results_obs), np.array(results_blend))
    print(f"  blend   : n={s_blend['n']:5d}  rmse={s_blend['rmse']:6.1f}  "
          f"mae={s_blend['mae']:6.1f}  bias={s_blend['bias']:7.1f}  corr={s_blend['corr']:6.3f}")


if __name__ == "__main__":
    main()
