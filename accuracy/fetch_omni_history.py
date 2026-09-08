#!/usr/bin/env python3
"""Fetch real, multi-year historical OMNI B/Bz/density/speed data directly
from NASA/SPDF -- item #6 of the "11 things" (accuracy_improvement_
tracking.txt, 2026-09-17 LOG): attack the small-N problem (only ~10-15
independent major disturbance events in AWSoM-R's own ~7.7-month
real-time record) with a much larger, completely independent, real
historical sample.

Source: NASA/SPDF OMNI HRO 1-minute monthly ASCII
(https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min/),
continuous since 1981 -- the SAME source, same documented column indices
and fill values, CLEAR's own fetch_insitu.py already uses for the live
RTSW dashboard's observed overlays (Website/MagConnectivity/scripts/
fetch_insitu.py, OMNI_SPEED_COL=21, OMNI_B_COL=13, OMNI_BZ_COL=18,
OMNI_DEN_COL=25). Re-derived here (not imported) to keep accuracy/
self-contained, but the column mapping is copied verbatim from that
already-verified, real, in-production script rather than re-guessed.

Deliberately NO overlap with this project's own 2026 test window (Feb
2026 onward) -- default range 2021-01 through 2023-12, a real, completely
independent 3-year period.

    python fetch_omni_history.py --start 2021-01 --end 2023-12
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import io
import urllib.error
import urllib.request

import numpy as np

OMNI_URL = ("https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni"
            "/monthly_1min/omni_min{ym}.asc")

# Same column indices/fills as CLEAR's own fetch_insitu.py (verbatim, not
# re-derived) -- see that script's own header comment for provenance.
OMNI_SPEED_COL = 21
OMNI_FILL = 99999.0
OMNI_B_COL, OMNI_BZ_COL, OMNI_DEN_COL = 13, 18, 25
OMNI_B_FILL, OMNI_DEN_FILL = 9999.0, 999.0


def months(start, end):
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def get(url, timeout=120, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if i == retries - 1:
                raise
            print(f"    retry {i + 1}/{retries - 1} after {type(e).__name__}", flush=True)


def parse_month(raw):
    """OMNI HRO 1-min ASCII -> (t_unix, v_kms, b_nT, bz_nT, n_cc), rows
    without a plasma speed dropped (matches fetch_insitu.py's own
    contract); B/Bz/n individually NaN'd on their own fill value.
    """
    t, v, b, bz, n = [], [], [], [], []
    for line in io.BytesIO(raw):
        f = line.split()
        if len(f) <= OMNI_DEN_COL:
            continue
        speed = float(f[OMNI_SPEED_COL])
        if speed >= OMNI_FILL:
            continue
        year, doy, hour, minute = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
        when = (dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
                + dt.timedelta(days=doy - 1, hours=hour, minutes=minute))
        t.append(when.timestamp())
        v.append(speed)
        bval = float(f[OMNI_B_COL])
        b.append(bval if bval < OMNI_B_FILL else np.nan)
        bzval = float(f[OMNI_BZ_COL])
        bz.append(bzval if abs(bzval) < OMNI_B_FILL else np.nan)
        nval = float(f[OMNI_DEN_COL])
        n.append(nval if nval < OMNI_DEN_FILL else np.nan)
    return (np.array(t, dtype=np.float64), np.array(v, dtype=np.float32),
            np.array(b, dtype=np.float32), np.array(bz, dtype=np.float32),
            np.array(n, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default="2023-12")
    ap.add_argument("--out", default="omni_history_bfield.npz")
    args = ap.parse_args()

    all_t, all_v, all_b, all_bz, all_n = [], [], [], [], []
    got = miss = 0
    for y, m in months(args.start, args.end):
        ym = f"{y}{m:02d}"
        url = OMNI_URL.format(ym=ym)
        try:
            raw = get(url)
        except Exception as e:  # noqa: BLE001
            print(f"  {y}-{m:02d}  MISSING ({type(e).__name__})")
            miss += 1
            continue
        t, v, b, bz, n = parse_month(raw)
        if t.size == 0:
            print(f"  {y}-{m:02d}  no usable samples")
            miss += 1
            continue
        all_t.append(t)
        all_v.append(v)
        all_b.append(b)
        all_bz.append(bz)
        all_n.append(n)
        got += 1
        print(f"  {y}-{m:02d}  +{t.size:,} samples  "
              f"(median v={np.nanmedian(v):.0f} km/s, "
              f"median |B|={np.nanmedian(b):.1f} nT, min Bz={np.nanmin(bz):.1f} nT)",
              flush=True)

    if not all_t:
        raise SystemExit("nothing fetched")

    t = np.concatenate(all_t)
    v = np.concatenate(all_v)
    b = np.concatenate(all_b)
    bz = np.concatenate(all_bz)
    n = np.concatenate(all_n)
    order = np.argsort(t)
    t, v, b, bz, n = t[order], v[order], b[order], bz[order], n[order]

    np.savez_compressed(args.out, t_unix=t, v_kms=v, b_nT=b, bz_nT=bz, n_cc=n)
    print(f"\n{got} months fetched, {miss} missing")
    print(f"wrote {t.size:,} total 1-minute samples to {args.out} "
          f"({dt.datetime.fromtimestamp(t[0], dt.timezone.utc):%Y-%m-%d} .. "
          f"{dt.datetime.fromtimestamp(t[-1], dt.timezone.utc):%Y-%m-%d})")


if __name__ == "__main__":
    main()
