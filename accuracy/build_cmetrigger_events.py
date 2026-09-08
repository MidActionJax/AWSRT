#!/usr/bin/env python3
"""Extract the 41 real, production-selected CME events (proper DONKI
selection + GONG magnetogram + ROI-pair flux-rope fit, not our own crude
speed+earth-directed heuristic) from cmetrigger's reference/baseline_
nbiro_2026 dataset, into the same {time, speed, loc} schema
donki_features.load_events() already expects -- so the existing pipeline
can use this candidate set as a drop-in replacement for donki_cme_2026.json,
no new matching code needed.

    python build_cmetrigger_events.py
"""
import glob
import json
import os

CMETRIGGER_EVENTS_DIR = (
    r"C:\Users\jaxon\OneDrive\Desktop\Github\CLEAR\RealTimeCME\cmetrigger"
    r"\reference\baseline_nbiro_2026\events"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "cmetrigger_events_2026.json")


def main():
    out = []
    for run_json in sorted(glob.glob(os.path.join(CMETRIGGER_EVENTS_DIR, "*", "run.json"))):
        with open(run_json) as f:
            d = json.load(f)
        e = d.get("event", {})
        cme_id, speed, loc = e.get("id"), e.get("speed"), e.get("location")
        if not cme_id or not speed:
            continue
        time_str = cme_id.split("-CME-")[0]  # "2026-01-18T18:09:00"
        out.append({"time": time_str, "speed": speed, "loc": loc})
    out.sort(key=lambda x: x["time"])
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} events to {OUT_PATH}")
    print(f"span: {out[0]['time']} to {out[-1]['time']}")


if __name__ == "__main__":
    main()
