#!/usr/bin/env python3
"""Score candidate solar wind speed predictors against the real RTSW record.

    python eval_harness.py
    python eval_harness.py --url https://clear-dev-clear.apps.campus.web.umich.edu/rtsw/api/series/?t=earth
    python eval_harness.py --json path/to/earth.json

Fetches the same model+observed series the live dashboard plots
(rtsw_products.py's earth.json: 10-minute model output alongside measured
OMNI/SWPC wind speed) and scores several candidate predictors against the
observed series using identical matching/statistics, so results are directly
comparable to each other and to the numbers in
AWSRT/RTSW_Accuracy_Improvement_Scoping.md.

Candidates scored:
    raw          the model series as-is (today's production behavior)
    lag<H>       the model series shifted by H hours, REPORTED FOR DIAGNOSTIC
                 PURPOSES ONLY. This is not a deployable real-time fix: the
                 correlation-improving direction requires model output from
                 its own future, which does not exist yet at serving time
                 (the model's latest point runs ~1-2h behind live obs, not
                 ahead of it). Kept here specifically so this stays visible
                 and doesn't get re-proposed as a cheap fix later.
    persist27    27-day solar-rotation recurrence: predict today's speed as
                 the observed speed one rotation ago (no model input at all)
    blend        simple average of the RAW model and persist27 (both
                 causally available at serving time, unlike the lag variant)

Matching: each observed sample is paired with the model/candidate value at
the nearest 10-minute bin (tolerance 1 bin), the same rule
rtsw_products.py's own read_sat/read_obs binning implies.
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request

DEFAULT_URL = ("https://clear-dev-clear.apps.campus.web.umich.edu"
               "/rtsw/api/series/?t=earth")
BIN_S = 600  # model cadence, seconds
ROTATION_S = 27 * 86400
DISTURBED_KMS = 500.0  # obs speed above this = "disturbed" segment


def fetch(url=None, path=None):
    if path:
        with open(path) as f:
            return json.load(f)
    with urllib.request.urlopen(url or DEFAULT_URL, timeout=60) as r:
        return json.load(r)


def to_bin_map(t, v):
    """{bin_index: value}, later duplicate wins (shouldn't occur in practice)."""
    return {round(ti / BIN_S): vi for ti, vi in zip(t, v) if vi is not None}


def lag_shift(model_bins, lag_hours):
    """model_bins shifted so bin B now holds what was at B - lag_bins.

    Shifting the model *forward* in time means: at real time T, use the
    model's value from T - lag (the model's forecast for "36h from now",
    read out 36h early) -- equivalently, re-key each entry to bin + lag_bins.
    """
    if lag_hours == 0:
        return model_bins
    lag_bins = round(lag_hours * 3600 / BIN_S)
    return {b + lag_bins: v for b, v in model_bins.items()}


def persistence_predict(obs_bins):
    """{bin: obs value from one 27-day rotation earlier}, only where available."""
    rot_bins = round(ROTATION_S / BIN_S)
    out = {}
    for b in obs_bins:
        prior = obs_bins.get(b - rot_bins)
        if prior is not None:
            out[b] = prior
    return out


def match(obs_bins, pred_bins, tolerance=1):
    """[(obs, pred)] for every obs bin with a pred within `tolerance` bins."""
    pairs = []
    for b, ov in obs_bins.items():
        pv = pred_bins.get(b)
        if pv is None:
            for d in range(1, tolerance + 1):
                pv = pred_bins.get(b - d, pred_bins.get(b + d))
                if pv is not None:
                    break
        if pv is not None:
            pairs.append((ov, pv))
    return pairs


def stats(pairs):
    n = len(pairs)
    if n == 0:
        return None
    err = [p - o for o, p in pairs]
    bias = sum(err) / n
    rmse = math.sqrt(sum(e * e for e in err) / n)
    mae = sum(abs(e) for e in err) / n
    mo = sum(o for o, _ in pairs) / n
    mp = sum(p for _, p in pairs) / n
    cov = sum((o - mo) * (p - mp) for o, p in pairs)
    vo = sum((o - mo) ** 2 for o, _ in pairs)
    vp = sum((p - mp) ** 2 for _, p in pairs)
    corr = cov / math.sqrt(vo * vp) if vo > 0 and vp > 0 else float("nan")
    return {"n": n, "bias": round(bias, 1), "rmse": round(rmse, 1),
            "mae": round(mae, 1), "corr": round(corr, 3)}


def report(name, pairs):
    all_s = stats(pairs)
    quiet = stats([(o, p) for o, p in pairs if o <= DISTURBED_KMS])
    dist = stats([(o, p) for o, p in pairs if o > DISTURBED_KMS])
    print(f"\n{name}")
    for label, s in (("all", all_s), ("quiet", quiet), ("disturbed", dist)):
        if s is None:
            print(f"  {label:10s}  no data")
        else:
            print(f"  {label:10s}  n={s['n']:6d}  rmse={s['rmse']:6.1f}  "
                  f"mae={s['mae']:6.1f}  bias={s['bias']:7.1f}  "
                  f"corr={s['corr']:6.3f}")
    return all_s


def find_events(obs_bins, threshold=DISTURBED_KMS, min_bins=3, merge_gap_bins=6):
    """Distinct disturbed events: obs > threshold, sustained >= min_bins
    consecutive bins, gaps of <= merge_gap_bins below threshold merged into
    the same event (a brief dip mid-event doesn't split it in two).

    Returns [(start_bin, end_bin, peak_kms)], ascending by start_bin. Used
    to size how much independent training/validation signal actually
    exists for Phase 2 -- a handful of long events is a very different
    ML problem than thousands of independent points, even if the point
    count within those events is large.
    """
    bins_sorted = sorted(obs_bins)
    events, cur = [], []
    for b in bins_sorted:
        if obs_bins[b] > threshold:
            if cur and b - cur[-1] > merge_gap_bins:
                if len(cur) >= min_bins:
                    events.append(cur)
                cur = []
            cur.append(b)
        elif cur and b - cur[-1] > merge_gap_bins:
            if len(cur) >= min_bins:
                events.append(cur)
            cur = []
    if cur and len(cur) >= min_bins:
        events.append(cur)
    return [(ev[0], ev[-1], max(obs_bins[b] for b in ev)) for ev in events]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="override the live feed URL")
    ap.add_argument("--json", default=None, help="local earth.json instead of fetching")
    ap.add_argument("--lag-hours", type=float, default=36.0,
                     help="timing correction to test (default: 36, the measured lag)")
    ap.add_argument("--blend-weight", type=float, default=0.5,
                     help="weight on the lag-corrected model in the blend "
                          "(default 0.5 = simple average with persist27)")
    ap.add_argument("--events", action="store_true",
                     help="list distinct disturbed events instead of scoring "
                          "candidates (for Phase 2 planning: how many "
                          "independent events actually exist to train/"
                          "validate against)")
    ap.add_argument("--event-threshold", type=float, default=DISTURBED_KMS,
                     help="km/s threshold defining a disturbed event")
    args = ap.parse_args()

    doc = fetch(args.url, args.json)
    model, obs = doc.get("model"), doc.get("obs")
    if not model or not obs:
        raise SystemExit("earth.json is missing model or obs -- nothing to score")

    obs_bins = to_bin_map(obs["t"], obs["v"])

    if args.events:
        import datetime
        events = find_events(obs_bins, threshold=args.event_threshold)
        print(f"target={doc.get('target')}  threshold={args.event_threshold:g} km/s  "
              f"{len(events)} distinct event(s)\n")
        for start_b, end_b, peak in events:
            start = datetime.datetime.utcfromtimestamp(start_b * BIN_S)
            end = datetime.datetime.utcfromtimestamp(end_b * BIN_S)
            dur_h = (end_b - start_b) * BIN_S / 3600
            print(f"  {start}  ..  {end}   ({dur_h:6.1f}h, peak {peak:5.0f} km/s)")
        return

    model_bins = to_bin_map(model["t"], model["v"])
    lagged_bins = lag_shift(model_bins, args.lag_hours)
    persist_bins = persistence_predict(obs_bins)
    blend_bins = {b: args.blend_weight * model_bins[b]
                       + (1 - args.blend_weight) * persist_bins[b]
                  for b in set(model_bins) & set(persist_bins)}

    print(f"target={doc.get('target')}  run={doc.get('run')}  "
          f"model span=[{model['t'][0]}, {model['t'][-1]}]  "
          f"obs span=[{obs['t'][0]}, {obs['t'][-1]}]")

    results = {}
    results["raw"] = report("raw (current production model)",
                             match(obs_bins, model_bins))
    results[f"lag{args.lag_hours:g}h"] = report(
        f"lag{args.lag_hours:g}h (DIAGNOSTIC ONLY -- not causally deployable, "
        f"see module docstring)",
        match(obs_bins, lagged_bins))
    results["persist27"] = report("persist27 (27-day recurrence, no model input)",
                                   match(obs_bins, persist_bins))
    results["blend"] = report(
        f"blend ({args.blend_weight:g}*raw + {1 - args.blend_weight:g}*persist27)",
        match(obs_bins, blend_bins))

    print("\nsummary (overall, all conditions):")
    for name, s in results.items():
        if s:
            print(f"  {name:12s}  rmse={s['rmse']:6.1f}  mae={s['mae']:6.1f}  "
                  f"bias={s['bias']:7.1f}  corr={s['corr']:6.3f}")
    deployable = {n: s for n, s in results.items()
                  if s and not n.startswith("lag")}
    best = min(deployable.values(), key=lambda s: s["rmse"])
    winner = [n for n, s in deployable.items() if s is best][0]
    print(f"\nlowest RMSE among deployable candidates (lag excluded, "
          f"diagnostic only): {winner}")


if __name__ == "__main__":
    main()
