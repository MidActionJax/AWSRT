"""Turn the DONKI CME catalog into per-bin features: is a cataloged CME
predicted to be arriving at Earth right now, and how fast is it.

Real fetch (2026-09-11): CLEAR's own DONKI mirror (fetch_donki.py) only
keeps a 30-day trailing window in production, so this pulls the FULL
history for the analysis window directly from the same public NASA/CCMC
API it uses (kauai.ccmc.gsfc.nasa.gov/DONKI), via fetch_donki.fetch()/
norm_cme() -- see donki_cme_2026.json for the saved pull (1183 CMEs,
2026-01-10 through 2026-09-11, 1182 with a usable speed estimate).

Arrival model is deliberately the simplest physically-defensible one
(constant-speed ballistic transit, no drag): given how little independent
disturbance data exists (~10-15 major events, see phase2_model.py), a
fancier arrival model isn't worth the extra parameters/assumptions yet.
Good enough to test whether "a cataloged CME is expected to be arriving
right now" carries real predictive signal at all before investing further.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

AU_KM = 149_597_870.0

# Empirically fit transit_hours = A + B*v_max, from Richardson/Cane
# verified near-Earth ICME entries with both a matched source CME and a
# V_max (icmetable2.htm, izw1.caltech.edu/ACE/ASC/DATA/level3/). Two
# fittings so far:
#   2026-09-11, first pass: 14 pairs, 2026 only. R^2=0.62, residual std
#   ~10h, transit_h = 109.53 - 0.07507*v_max.
#   2026-09-11, refit: 255 pairs, 1999-2026 (full catalog history pulled,
#   not just 2026). correlation(v_max, transit_h) = -0.725 (consistent
#   with the 14-point sample's -0.79 -- confirms the relationship is real
#   and stable across decades, not a small-sample fluke). R^2=0.526,
#   residual std ~14h. THIS is the current fit in use: the coefficients
#   changed meaningfully from the 14-point version (steeper slope, higher
#   intercept) because the small sample was skewed toward a narrow speed
#   range; 255 points spanning the real speed distribution is the more
#   trustworthy fit. See accuracy_improvement_tracking.txt LOG for the
#   full comparison and how this changed (or didn't) the downstream
#   Phase 2 result.
# Both fittings used ONLY EXPERT-VERIFIED CME-to-ICME matches -- this
# project's own heuristic matching (picking the fastest DONKI CME in a
# multi-day window) gave correlation 0.045 on the same kind of
# relationship, confirming the earlier DONKI-only failure was the
# matching step, not the underlying physics.
CALIBRATED_TRANSIT_A = 125.42
CALIBRATED_TRANSIT_B = -0.09942

# NOTED, NOT FIXED HERE (found 2026-09-14 while building dbm_transit()
# below): this regression's OWN fitting variable was V_max (the ICME's
# speed AT EARTH, per the comment above), but build_calibrated_index()
# calls this with e["speed"] -- the DONKI catalog's NEAR-SUN launch
# speed, a different (correlated but not identical) quantity. A CME
# decelerates via drag between launch and arrival, so launch speed
# systematically overstates V_max, especially for fast CMEs -- meaning
# this function is applied to an input it wasn't fit on. Left as-is
# (still used by uncertainty_model.py's calibrated_arrival mode; not
# touched today, out of scope) but this mismatch is exactly what
# dbm_transit() below is designed to avoid -- it takes launch speed as
# its actual input and derives transit time AND arrival speed together,
# self-consistently, from one physical model instead of two separately-
# fit regressions on different variables.


def calibrated_transit_hours(speed_kms):
    return CALIBRATED_TRANSIT_A + CALIBRATED_TRANSIT_B * speed_kms


# Drag-based model (DBM), Vrsnak & Zic 2007: a CME/ICME decelerates (or
# accelerates) toward the ambient solar wind speed w via aerodynamic-like
# drag, at a rate proportional to the square of its speed relative to the
# wind. For constant w and drag parameter gamma, the relative speed
# u(t) = v(t) - w has the closed form u(t) = u0 / (1 + gamma*|u0|*t)
# (this single form is correct whether u0 is positive -- CME faster than
# wind, decelerating -- or negative -- CME slower than wind, being
# dragged along/accelerated -- since drag always pushes |u| toward 0).
# Integrating v(t) = w + u(t) gives distance traveled:
#   r(t) - r0 = w*t + sign(u0)/gamma * ln(1 + gamma*|u0|*t)
# Transit time T solves r(T) = R (target distance, e.g. 1 AU) -- solved
# numerically (r(t) is monotonic in t for w > 0, so a bracketed root
# find is well-posed). Arrival speed is then v(T) = w + u(T).
#
# GAMMA IS NOT AN EXTERNAL LITERATURE VALUE HERE -- calibrated locally
# against this project's own event set instead (see
# calibrate_dbm_gamma() and accuracy_improvement_tracking.txt LOG,
# "TIER 0c"), specifically because Tier 0a already found that an
# externally-calibrated magnitude relationship (estimate_earth_vmax, a
# real, honestly-validated fit on ITS OWN 12-pair sample) did not
# transfer its absolute scale to this project's specific sample of
# unusually severe events. Same risk applies to any literature gamma
# (typically quoted ~0.1-2e-7 km^-1 from different, unrelated event
# samples) -- so it is deliberately not used, fit locally instead.
R0_KM = 20.0 * 696_000.0  # ~20 solar radii, the conventional DBM/coronagraph starting distance


def dbm_transit(v0_kms, w_kms, gamma, r0_km=R0_KM, r_km=AU_KM, max_hours=400.0):
    """(transit_hours, arrival_speed_kms) for a CME launched at v0_kms into
    ambient wind w_kms, under drag parameter gamma (km^-1). Returns
    (None, None) if no physically sensible transit is found within
    max_hours (e.g. gamma so large the CME never covers the distance)."""
    import numpy as _np
    from scipy.optimize import brentq

    u0 = v0_kms - w_kms
    au0 = abs(u0)
    sign = 1.0 if u0 >= 0 else -1.0

    def r_at(t_s):
        if au0 < 1e-9:
            return r0_km + w_kms * t_s
        return r0_km + w_kms * t_s + (sign / gamma) * _np.log1p(gamma * au0 * t_s)

    target = r_km - r0_km
    hi = max_hours * 3600.0
    if r_at(hi) - r0_km < target:
        return None, None
    lo = 1.0
    if r_at(lo) - r0_km >= target:
        return 0.0, v0_kms
    t_s = brentq(lambda t: (r_at(t) - r0_km) - target, lo, hi, xtol=1.0)
    if au0 < 1e-9:
        arrival_speed = w_kms
    else:
        arrival_speed = w_kms + u0 / (1.0 + gamma * au0 * t_s)
    return t_s / 3600.0, arrival_speed


def dbm_speed_at(v0_kms, w_kms, gamma, t_s):
    """v(t) under the same DBM drag physics as dbm_transit(), evaluated at
    an ARBITRARY elapsed time t_s since launch -- not solved for reaching
    any particular distance. dbm_transit() already computes exactly this
    expression internally (its own `arrival_speed` line) but only ever
    returns it at the one specific t_s that solves r(t)=1AU, discarding
    the fact that the same closed-form velocity curve is valid at any t.

    Added 2026-09-17 specifically to test a genuinely different idea from
    the already-tried (and already-rejected, Tier 0c, 2026-09-14) single-
    point DBM arrival-speed replacement: instead of one flat magnitude
    number for "when this CME arrives," use the CONTINUOUS speed profile
    to answer "what does this CME's own physics say the speed will be at
    THIS SPECIFIC hour" -- which may be meaningfully different from the
    nominal arrival speed if the target hour falls before or after the
    CME's own predicted transit time (the CME is still decelerating
    toward w if evaluated early, or has already relaxed toward w if
    evaluated late). This directly targets the timing-precision failure
    diagnosed in the (also rejected) cme_arrival_signal feature above: a
    single magnitude number applied regardless of exactly which hour
    within the disturbance is being predicted.

    TESTED 2026-09-17, KEPT AS A CORRECT UTILITY DESPITE A REAL NEGATIVE
    RESULT (see accuracy_improvement_tracking.txt LOG): evaluating this at
    the exact forecast target hour, gated by a tight match-window (a few
    hours) against the candidate's own DBM-predicted arrival, turned out
    to be numerically almost IDENTICAL to just using the flat single-point
    arrival speed -- a tight window forces the target hour to already be
    close to the nominal arrival time by construction, so there's no real
    "elsewhere on the profile" left to evaluate. Widening the match window
    to actually exercise this function's real advantage (evaluating well
    before/after the nominal arrival) reintroduces the exact candidate-
    dilution problem this project diagnosed on day one (loose windows
    match almost everything, carrying no discriminating information). Not
    wired into early_warning.py as a result -- checked cheaply at the
    diagnostic/correlation level before investing in full ridge-regression
    integration, and the underlying correlation was never there to find.
    """
    u0 = v0_kms - w_kms
    au0 = abs(u0)
    if au0 < 1e-9 or t_s <= 0:
        return v0_kms if t_s <= 0 else w_kms
    return w_kms + u0 / (1.0 + gamma * au0 * t_s)


# Empirically fit v_max_earth = A + B*donki_source_speed, from 12 2026
# Richardson/Cane verified ICME entries matched to their real DONKI CME
# catalog entry by exact launch time (see accuracy_improvement_tracking.txt
# LOG, 2026-09-12). This is a DIFFERENT relationship than
# CALIBRATED_TRANSIT_A/B above: that one uses V_max (the ICME's own speed
# AT EARTH) to predict transit time; this one uses the DONKI CME's speed
# NEAR THE SUN (what's actually available in real time before the ICME
# arrives) to predict what V_max at Earth will turn out to be -- the
# actual magnitude estimate needed for a live forecast. correlation=0.807
# on the matched pairs -- strong, and decouples magnitude entirely from
# the OMNI composite's own (proven unreliable, see LOG) internal scale.
MAGNITUDE_FIT_A = 354.30
MAGNITUDE_FIT_B = 0.3324


def estimate_earth_vmax(donki_source_speed_kms):
    return MAGNITUDE_FIT_A + MAGNITUDE_FIT_B * donki_source_speed_kms


# 2026-09-16: MAGNITUDE_FIT_A/B above is fit on only 12 pairs (2026-only,
# DONKI-matched). Found a way to expand this 25x while staying fully
# external (not tuned to this project's own small test window, avoiding
# the overfitting risk found the same day in a locally-recalibrated
# version -- see accuracy_improvement_tracking.txt): Richardson & Cane's
# own catalog already records an exact LASCO CME timestamp for 374 of its
# 633 verified rows (icme_raw.html, "LASCO CME Y/M/D (UT)" column) --
# matched against the real, public SOHO/LASCO CME catalog
# (cdaw.gsfc.nasa.gov/CME_list, continuous since 1996) via
# lasco_catalog.py, 303/327 references resolved to a real measured speed,
# spanning 1996-2026 (~2.5 solar cycles). Refit: correlation 0.596 (n=303)
# -- LOWER than the 12-pair fit's claimed 0.807, and more trustworthy for
# it (small samples inflate correlation by luck).
LASCO_MAGNITUDE_A = 403.97
LASCO_MAGNITUDE_B = 0.2177


def estimate_earth_vmax_lasco(donki_source_speed_kms):
    """Same relationship as estimate_earth_vmax(), refit on the 303-pair
    LASCO-matched sample instead of the original 12 pairs. Tested directly
    (lasco_magnitude_test.py): using this fit's absolute scale as-is on
    our own data is a clean NEGATIVE (our own test window's disturbances
    run more severe than the general 1996-2026 population this represents)
    -- but its SLOPE is robust (fit on 25x the data, multiple solar
    cycles, does not collapse to flat the way a locally-refit slope did)
    and is what final_correction.py actually uses, paired with a
    LOO-shifted local intercept rather than this function's own intercept.
    """
    return LASCO_MAGNITUDE_A + LASCO_MAGNITUDE_B * donki_source_speed_kms


import re

_LOC_RE = re.compile(r"^([NS])(\d{1,2})([EW])(\d{1,3})$")


def parse_loc(loc):
    """DONKI sourceLocation (e.g. "S16W04") -> signed central-meridian
    distance in degrees (west positive, east negative), or None if blank/
    unparseable. Values beyond +/-90 mean the source was on the far side
    of the Sun (as seen from Earth) -- definitely not Earth-directed,
    not just a low-confidence guess.
    """
    if not loc:
        return None
    m = _LOC_RE.match(loc.strip())
    if not m:
        return None
    ns, lat, ew, lon = m.groups()
    lon = int(lon)
    return lon if ew == "W" else -lon


def load_events(path):
    with open(path) as f:
        raw = json.load(f)
    out = []
    for e in raw:
        if not e.get("speed") or not e.get("time"):
            continue
        t = e["time"].rstrip("Z")
        try:
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append({"launch_unix": dt.timestamp(), "speed": float(e["speed"]),
                     "cmd": parse_loc(e.get("loc"))})
    out.sort(key=lambda e: e["launch_unix"])
    return out


def transit_hours(speed_kms):
    return AU_KM / speed_kms / 3600.0


def build_index(events, max_lookback_hours=200):
    """Precompute (predicted_arrival_unix, speed) for every event, so
    per-bin lookups don't repeat the transit calculation."""
    return [(e["launch_unix"] + transit_hours(e["speed"]) * 3600.0, e["speed"],
             e["launch_unix"])
            for e in events]


def build_calibrated_index(events):
    """Same as build_index, but using the empirically-calibrated transit
    formula (see CALIBRATED_TRANSIT_A/B) instead of naive constant-speed
    physics."""
    return [(e["launch_unix"] + calibrated_transit_hours(e["speed"]) * 3600.0,
             e["speed"], e["launch_unix"])
            for e in events]


def filter_earth_directed(events, max_cmd=60, include_unknown=False):
    """Keep only CMEs plausibly Earth-directed: known location within
    max_cmd degrees of central meridian. include_unknown=True also keeps
    blank-location entries (often halo CMEs, ambiguous but not
    necessarily off-target) as a separate lower-confidence inclusion
    rather than silently treating "unknown" the same as "known and
    off-target" (a location >90 deg is on the far side of the Sun and
    is excluded either way, unknown or not)."""
    out = []
    for e in events:
        cmd = e.get("cmd")
        if cmd is None:
            if include_unknown:
                out.append(e)
            continue
        if abs(cmd) <= max_cmd:
            out.append(e)
    return out


def activity_feature(t_unix, events, window_hours=72):
    """(count, max_speed) for CMEs launched in the trailing window_hours
    before t_unix -- a coarse "how much CME activity has there been
    recently" signal, sidestepping arrival-time modeling entirely (no
    transit-time/drag assumption at all, just launch recency).

    ``events`` is the raw list from load_events (unix launch time + speed),
    not the arrival index build_index() produces.
    """
    window_s = window_hours * 3600.0
    count, max_speed = 0, 0.0
    for e in events:
        if e["launch_unix"] > t_unix:
            continue
        if t_unix - e["launch_unix"] > window_s:
            continue
        count += 1
        max_speed = max(max_speed, e["speed"])
    return float(count), max_speed


def load_enlil_earth_arrivals(path):
    """Real, already-computed Earth-arrival predictions from NASA's own
    OPERATIONAL WSA-Enlil model runs, embedded directly in DONKI's raw API
    response (each CME's cmeAnalyses[].enlilList[] entries) -- added
    2026-09-17 after Jaxon pointed out that this project cannot get
    CLEAR's own CME-injection pipeline fixed (Koban/Biro don't work with
    us) and asked what's actually within reach alone. This is: NASA/NOAA
    SWPC already run their own real, independent, operational CME-arrival
    simulation (WSA-Enlil, a genuinely different model from AWSoM-R/EEGGL)
    for a large fraction of catalogued CMEs, and its Earth-arrival verdict
    (isEarthGB -- glancing blow, or isEarthMinorImpact) plus a real
    estimatedShockArrivalTime is sitting in the SAME public API this
    project already pulls from -- no new access, no supercomputer, no
    one else's cooperation needed. Confirmed directly (not assumed): 97
    distinct real CME events in a fresh Jan-Sept 2026 pull have a genuine
    WSA-Enlil Earth-impact verdict, 144 total enlil runs (some events were
    modeled more than once), 143/144 also carry a real Kp-index estimate
    (kp_90/kp_135/kp_180 -- different ensemble members/assumptions, not
    always all three populated).

    Unlike every other arrival source in this file, the arrival time here
    is NOT computed from a transit-time formula at all -- it's WSA-Enlil's
    own simulated estimate, directly. This isolates a single, real
    hypothesis: is this project's own biggest recurring bottleneck
    (candidate TIMING precision, see accuracy_improvement_tracking.txt
    LOG's four same-day negatives) actually a limitation of the transit-
    time FORMULA, fixable by using a real, independent, event-specific
    simulation's own timing instead of a population-average empirical fit?

    Returns a list of (arrival_unix, speed, launch_unix) tuples -- the
    same shape build_calibrated_index()/build_index() produce, usable
    directly with arrival_feature() and this project's existing candidate-
    matching machinery, but built from a real second model's own answer
    rather than derived by this project at all.
    """
    with open(path) as f:
        raw = json.load(f)
    return [(e["arrival_unix"], e["speed"], e["launch_unix"])
            for e in enlil_earth_events_detailed(raw)]


def enlil_earth_events_detailed(raw_events):
    """Same real WSA-Enlil Earth-impact extraction as
    load_enlil_earth_arrivals() above, but (a) takes already-parsed raw
    DONKI JSON directly instead of only reading from a cached file --
    shared by the file-based loader above and fetch_donki_raw_live()
    below -- and (b) returns richer dicts (real Kp estimate, DONKI link,
    glancing-blow vs minor-impact flag) instead of bare tuples, for
    display use (added 2026-09-17 for a real live "second model"
    dashboard comparison, not just the candidate-matching test above).
    """
    out = []
    for e in raw_events:
        launch_t = e.get("startTime")
        if not launch_t:
            continue
        try:
            launch_dt = datetime.fromisoformat(launch_t.rstrip("Z")).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        for a in (e.get("cmeAnalyses") or []):
            speed = a.get("speed")
            if not speed:
                continue
            for enlil in (a.get("enlilList") or []):
                if not (enlil.get("isEarthGB") or enlil.get("isEarthMinorImpact")):
                    continue
                arrival_t = enlil.get("estimatedShockArrivalTime")
                if not arrival_t:
                    continue
                try:
                    arrival_dt = datetime.fromisoformat(arrival_t.rstrip("Z")).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                kp = enlil.get("kp_90")
                if kp is None:
                    kp = enlil.get("kp_135")
                if kp is None:
                    kp = enlil.get("kp_180")
                out.append({
                    "launch_unix": launch_dt.timestamp(),
                    "arrival_unix": arrival_dt.timestamp(),
                    "speed": float(speed),
                    "kp": kp,
                    "is_glancing": bool(enlil.get("isEarthGB")),
                    "is_minor": bool(enlil.get("isEarthMinorImpact")),
                    "activity_id": e.get("activityID"),
                    "link": e.get("link"),
                })
    out.sort(key=lambda t: t["launch_unix"])
    return out


def fetch_donki_raw_live(days_back=60, days_forward=7, timeout=30):
    """Live fetch of DONKI's raw CME API for a recent trailing window --
    added 2026-09-17 for the dashboard's live "second model" comparison
    (see enlil_earth_events_detailed above). Unlike donki_raw_full.json
    (a one-time snapshot fetched earlier today), this hits the real API
    fresh every time it's called, so a currently-active WSA-Enlil
    Earth-impact prediction shows up without needing to re-fetch and
    commit a new snapshot file by hand.
    """
    import urllib.request
    from datetime import timedelta
    end = datetime.now(timezone.utc) + timedelta(days=days_forward)
    start = end - timedelta(days=days_back + days_forward)
    url = (f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"
           f"?startDate={start:%Y-%m-%d}&endDate={end:%Y-%m-%d}")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def load_donki_events_with_width(path):
    """Load DONKI's RAW, un-normalized API response (kauai.ccmc.gsfc.nasa.gov/
    DONKI/WS/get/CME?startDate=...&endDate=...) directly, instead of this
    project's older normalized cache (donki_cme_2026.json, built by a
    since-lost fetch_donki.norm_cme() step that only kept kind/id/time/
    speed/type/loc/link). Added 2026-09-17 after Experiment 1's finding
    (see accuracy_improvement_tracking.txt LOG) that swapping to CLEAR's
    much smaller real production candidate list (71 events) traded away
    critical COVERAGE just to get a real per-event half_angle -- checking
    the raw DONKI response directly (not assumed, fetched and inspected)
    confirmed halfAngle has been available in DONKI's own API all along,
    on 1239/1240 events in a fresh Jan-Sept 2026 pull, i.e. the SAME dense
    coverage as the existing cache PLUS real width almost everywhere.

    Each raw DONKI event can carry multiple `cmeAnalyses` revisions (119 of
    1240 in the 2026-09-17 pull); DONKI itself flags which one to trust via
    `isMostAccurate` (present on 1239/1240) -- used here in preference to
    just taking the first entry.

    DONKI's `longitude` field on a cmeAnalysis is already in this
    project's own established sign convention (west positive, east
    negative -- confirmed directly against production catalog entries
    where both a location string and longitude/latitude were present and
    agreed, e.g. "N9E166" <-> longitude=-166.0), so it's used directly as
    `cmd` here rather than re-parsing a location string (which DONKI's
    top-level `sourceLocation` is frequently blank for anyway, even when
    the analysis itself has real longitude/latitude/halfAngle).
    """
    with open(path) as f:
        raw = json.load(f)
    out = []
    for e in raw:
        analyses = e.get("cmeAnalyses") or []
        if not analyses:
            continue
        chosen = next((a for a in analyses if a.get("isMostAccurate")), analyses[0])
        speed = chosen.get("speed")
        t = e.get("startTime")
        if not speed or not t:
            continue
        t = t.rstrip("Z")
        try:
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        lon = chosen.get("longitude")
        out.append({"launch_unix": dt.timestamp(), "speed": float(speed),
                     "cmd": float(lon) if lon is not None else None,
                     "half_angle": chosen.get("halfAngle")})
    out.sort(key=lambda e: e["launch_unix"])
    return out


def load_production_cme_events(path):
    """Load CLEAR's own real, production CME-selection output (synced to
    solsticedisk at data/PIPELINE/CME_pipeline/checking_CME.json -- pulled
    2026-09-17 via the same SSH key CLEAR_web's own refresh-local-data.sh
    uses). This is NOT DONKI's raw catalog: it is the already speed-
    filtered (every entry checked >=800 km/s, matching CLEAR's real
    RealTimeCME/cmetrigger selection rule) set of CMEs that CLEAR's own
    real-time pipeline judged worth simulating -- a materially more
    selective, already-vetted source than this project's own
    donki_cme_2026.json pull, and (from 2026-04-01 onward) it carries a
    real per-event half_angle the DONKI pull never had (see
    filter_earth_directed_variable_width below for why that matters).

    Uses "liftofftime" (the source's own refined liftoff estimate) rather
    than the "id" timestamp (which is this catalog's detection time, not
    the physical launch time transit-time math needs) -- same physical
    quantity load_events() above reads from DONKI's "time" field.

    Same output shape as load_events(): a list of dicts with launch_unix,
    speed, cmd -- plus "half_angle" (None on older entries, pre-2026-04).
    """
    with open(path) as f:
        raw = json.load(f)
    out = []
    for e in raw:
        if not e.get("speed") or not e.get("liftofftime"):
            continue
        t = e["liftofftime"].rstrip("Z")
        try:
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append({"launch_unix": dt.timestamp(), "speed": float(e["speed"]),
                     "cmd": parse_loc(e.get("location")),
                     "half_angle": e.get("half_angle")})
    out.sort(key=lambda e: e["launch_unix"])
    return out


def filter_earth_directed_variable_width(events, default_max_cmd=60, margin=0.0,
                                          include_unknown=False):
    """Like filter_earth_directed, but uses each CME's OWN half_angle (its
    real angular half-width, when known) as the Earth-directedness cutoff
    instead of one flat degree threshold applied to every CME regardless of
    width. This is the actual physically-correct test (a narrow ~20-degree
    jet centered 55 degrees off disk-center genuinely may not reach Earth;
    a wide ~50-degree halo centered 55 degrees off-center very well might)
    -- filter_earth_directed's flat max_cmd is a reasonable proxy when
    width isn't known, but a real per-event width is strictly better
    information once it's available (see load_production_cme_events).
    Falls back to default_max_cmd for any event with no half_angle (e.g.
    a pre-2026-04 production-catalog entry, or any DONKI-sourced event
    passed in here by mistake -- DONKI events never carry half_angle, so
    they always fall back). A location beyond +/-90 degrees is excluded
    unconditionally either way (far side of the Sun, not a width question).
    """
    out = []
    for e in events:
        cmd = e.get("cmd")
        if cmd is None:
            if include_unknown:
                out.append(e)
            continue
        if abs(cmd) > 90:
            continue
        half_angle = e.get("half_angle")
        limit = (half_angle + margin) if half_angle is not None else default_max_cmd
        if abs(cmd) <= limit:
            out.append(e)
    return out


def build_arrival_index(events, min_speed_kms, max_cmd=60, include_unknown=True,
                         half_angle_aware=False, margin=0.0):
    """Filter `events` (from load_events or load_production_cme_events) to
    fast, Earth-directed candidates and build a calibrated-transit arrival
    index -- the ONE shared implementation of "which CMEs count as
    candidates and when do they arrive," used by both final_correction.py
    and early_warning.py. Added 2026-09-17 specifically so a second script
    reusing this logic doesn't re-derive it independently -- final_
    correction.py already hit this exact duplication-bug class twice in
    one day on 2026-09-14 (apply_to_live() and export_dashboard_data.py
    each drifted out of sync with the canonical candidate logic); this
    exists to prevent a third occurrence as a new caller gets added.
    """
    fast = [e for e in events if e["speed"] >= min_speed_kms]
    if half_angle_aware:
        filtered = filter_earth_directed_variable_width(
            fast, default_max_cmd=max_cmd, margin=margin, include_unknown=include_unknown)
    else:
        filtered = filter_earth_directed(fast, max_cmd=max_cmd, include_unknown=include_unknown)
    return build_calibrated_index(filtered)


def cme_arrival_signal(t_now_unix, t_target_unix, arrivals, tolerance_hours=3.0,
                        lookback_hours=200):
    """For a FORECAST issued at t_now_unix predicting the value at
    t_target_unix (t_target_unix = t_now_unix + horizon): among CME
    candidates launched before t_now_unix (i.e. causally available at the
    moment the forecast is issued -- a CME that launches AFTER t_now_unix
    obviously can't be known about yet), is any one's predicted arrival
    time within tolerance_hours of t_target_unix?

    This is a different question from arrival_feature() above (which asks
    "what's arriving closest to right now") -- here we're asking "is a
    real physics-based CME arrival expected to land inside THIS specific
    forecast's target window." Returns (has_signal, donki_speed):
    donki_speed is the candidate's own catalog/launch speed (NOT yet an
    Earth-arrival magnitude estimate -- the caller applies
    estimate_earth_vmax_lasco or similar), 0.0 when has_signal is 0.0, so
    "no candidate" and "a candidate with a nominally-zero speed" can never
    be confused (same convention as arrival_feature's has_candidate).
    """
    lookback_s = lookback_hours * 3600.0
    best = None
    for arrival_unix, speed, launch_unix in arrivals:
        if launch_unix > t_now_unix or (t_now_unix - launch_unix) > lookback_s:
            continue
        offset_h = abs(arrival_unix - t_target_unix) / 3600.0
        if best is None or offset_h < best[0]:
            best = (offset_h, speed)
    if best is None or best[0] > tolerance_hours:
        return 0.0, 0.0
    return 1.0, best[1]


def arrival_feature(t_unix, arrivals, max_offset_hours=48.0, lookback_hours=200):
    """(offset_hours, speed, has_candidate) for the CME whose predicted
    arrival is closest in time to t_unix, among CMEs launched within
    lookback_hours before t_unix.

    offset_hours = t_unix - predicted_arrival, in hours (0 = predicted to
    arrive exactly now; negative = still inbound; positive = already past
    its predicted arrival). Clipped to +/-max_offset_hours and has_candidate
    set to 0 when nothing is close enough to be a plausible current signal,
    so "no CME nearby" and "a CME arriving right now" don't collide at the
    same feature value.
    """
    lookback_s = lookback_hours * 3600.0
    best = None
    for arrival_unix, speed, launch_unix in arrivals:
        if launch_unix > t_unix or (t_unix - launch_unix) > lookback_s:
            continue
        offset_h = (t_unix - arrival_unix) / 3600.0
        if best is None or abs(offset_h) < abs(best[0]):
            best = (offset_h, speed)
    if best is None or abs(best[0]) > max_offset_hours:
        return 0.0, 0.0, 0.0
    offset_h, speed = best
    return offset_h, speed, 1.0
