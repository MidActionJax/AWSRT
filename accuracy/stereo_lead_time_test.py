#!/usr/bin/env python3
"""Tier 3 item 7 (never attempted until now): does STEREO-A's real-time
measured solar wind speed carry genuine LEAD TIME on what Earth will see,
via corotation (both spacecraft sample the same rotating coronal structure
at different heliographic longitudes)? Unlike 27-day persistence (a month-
old guess) or anything else in this project, this would be a direct,
current, real measurement of the actual structure heading toward Earth --
genuinely new information, not statistics on old information.

Works around a real bug found in CLEAR's own fetch_insitu.py: the beacon
CDF's "Bulk_Speed" variable does not exist in the 2026 (V14) format --
speed is instead the 4th element of the "Velocity_RTN" array (the first
3 R/T/N components are unpopulated fill in this quicklook product). Not
fixed in fetch_insitu.py itself (shared production code, out of scope for
this test); worked around here only.

    python stereo_lead_time_test.py
"""
import datetime
import tempfile
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\jaxon\OneDrive\Desktop\Github\CLEAR\Website\MagConnectivity\scripts")
import fetch_insitu as fi  # noqa: E402
from eval_harness import BIN_S, fetch, to_bin_map  # noqa: E402

V_SANE = fi.V_SANE
LAG_DAYS = range(-10, 11)  # sweep both directions -- sign of the real lag isn't assumed


def fetch_sta_speed(year, months):
    """(t_unix, v_kms) for STEREO-A beacon speed across the given months,
    using the corrected Velocity_RTN[:,3] extraction (see module docstring)."""
    all_t, all_v = [], []
    for m in months:
        base = fi.STEREO_BEACON.format(craft="ahead", y=year, m=m)
        try:
            names = fi._listing(base)
        except Exception as e:
            print(f"  {year}-{m:02d}  no listing ({type(e).__name__})")
            continue
        day_t, day_v = [], []
        bad = 0
        for name in names:
            try:
                raw = fi.get(base + name, timeout=60)
                import cdflib
                with tempfile.NamedTemporaryFile(suffix=".cdf", delete=False) as f:
                    f.write(raw)
                    tmp = f.name
                try:
                    c = cdflib.CDF(tmp)
                    vel = np.asarray(c.varget("Velocity_RTN"), dtype=np.float64)
                    epoch = c.varget("Epoch1")
                    tu = np.asarray(cdflib.cdfepoch.unixtime(epoch), dtype=np.float64)
                finally:
                    del c
                    os.unlink(tmp)
                speed = vel[:, 3]
                ok = np.isfinite(speed) & (speed > V_SANE[0]) & (speed < V_SANE[1])
                day_t.append(tu[ok])
                day_v.append(speed[ok])
            except Exception:
                bad += 1
        if day_t:
            t = np.concatenate(day_t)
            v = np.concatenate(day_v)
            all_t.append(t)
            all_v.append(v)
            print(f"  {year}-{m:02d}  +{t.size:,} samples from {len(names)} files"
                  + (f"  [{bad} unreadable]" if bad else ""))
        else:
            print(f"  {year}-{m:02d}  no usable samples ({bad}/{len(names)} unreadable)")
    if not all_t:
        return np.array([]), np.array([])
    return np.concatenate(all_t), np.concatenate(all_v)


def daily_median(t, v):
    """(day_int, median_speed) -- integer calendar-day keys throughout, no
    float round-trip (an earlier version converted back to a noon-centered
    unix timestamp and re-derived the day via division+round, which silently
    hit numpy's round-half-to-even on the exact .5 fraction and collapsed
    every key to an even day -- caught by every odd lag mysteriously
    returning zero pairs). Coarse daily binning matches the beacon product's
    own noisy/sparse quicklook nature (per fetch_insitu.py's own doc: ~15%
    of rows carry plasma, spikes are common)."""
    day = np.floor(t / 86400.0).astype(np.int64)
    days = np.unique(day)
    out_d, out_v = [], []
    for d in days:
        m = day == d
        if m.sum() < 3:
            continue
        out_d.append(int(d))
        out_v.append(float(np.median(v[m])))
    return np.array(out_d, dtype=np.int64), np.array(out_v)


CACHE_PATH = os.path.join(os.path.dirname(__file__), "stereo_test_data", "sta_2026_speed_cache.npz")


def main():
    if os.path.exists(CACHE_PATH):
        d = np.load(CACHE_PATH)
        sta_t, sta_v = d["t"], d["v"]
        print(f"Loaded cached STEREO-A data: {sta_t.size:,} samples")
    else:
        print("Fetching real STEREO-A beacon speed, 2026-01..2026-09...")
        months = list(range(1, 10))
        sta_t, sta_v = fetch_sta_speed(2026, months)
        if sta_t.size == 0:
            raise SystemExit("no STEREO-A data fetched -- check network/host reachability")
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez_compressed(CACHE_PATH, t=sta_t, v=sta_v)
    sta_day, sta_day_v = daily_median(sta_t, sta_v)
    print(f"\nSTEREO-A: {sta_t.size:,} raw samples -> {sta_day.size} daily medians, "
          f"median speed {np.median(sta_v):.0f} km/s\n")

    print("Fetching real Earth obs data (same live feed used throughout this project)...")
    doc = fetch()
    obs = doc["obs"]
    obs_bins = to_bin_map(obs["t"], obs["v"])
    earth_bins_sorted = sorted(obs_bins.keys())
    earth_t = np.array(earth_bins_sorted, dtype=np.float64) * BIN_S  # bin index -> real unix seconds
    earth_v = np.array([obs_bins[b] for b in earth_bins_sorted])
    earth_day, earth_day_v = daily_median(earth_t, earth_v)
    print(f"Earth: {len(obs_bins):,} raw bins -> {earth_day.size} daily medians, "
          f"median speed {np.median(earth_v):.0f} km/s\n")

    # align both to a common daily grid -- integer day keys throughout, no float round-trip
    sta_map = dict(zip(sta_day.tolist(), sta_day_v))
    earth_map = dict(zip(earth_day.tolist(), earth_day_v))

    print("Cross-correlation at each lag (positive lag = STEREO-A leads Earth "
          "by that many days, i.e. STEREO-A(t) vs Earth(t+lag)):")
    print(f"{'lag(days)':>10s}{'n_pairs':>10s}{'corr':>10s}")
    best_lag, best_corr, best_n = None, -2.0, 0
    for lag in LAG_DAYS:
        pairs = [(sta_map[d], earth_map[d + lag]) for d in sta_map
                 if (d + lag) in earth_map]
        if len(pairs) < 20:
            print(f"{lag:>10d}{len(pairs):>10d}{'--':>10s}")
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        print(f"{lag:>10d}{len(pairs):>10d}{corr:>10.3f}")
        if not np.isnan(corr) and corr > best_corr:
            best_lag, best_corr, best_n = lag, corr, len(pairs)

    print(f"\nBest lag: {best_lag} days, corr={best_corr:.3f}, n={best_n}")
    zero_lag_pairs = [(sta_map[d], earth_map[d]) for d in sta_map if d in earth_map]
    if zero_lag_pairs:
        a = np.array([p[0] for p in zero_lag_pairs])
        b = np.array([p[1] for p in zero_lag_pairs])
        zero_corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        print(f"Zero-lag corr (for reference): {zero_corr:.3f}, n={len(zero_lag_pairs)}")


if __name__ == "__main__":
    main()
