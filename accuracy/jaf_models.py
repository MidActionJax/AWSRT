#!/usr/bin/env python3
"""Portable model files for Justified Adaptive fluX: the fitted models as
plain JSON (arrays, tuples and integer keys tagged), so they load on any
numpy version -- a pickle made with numpy 2 does not load on numpy 1.

    python jaf_models.py export jaf_forecast_model_ops_20261119.pkl     -> .json
    python jaf_models.py check  jaf_forecast_model_ops_20261119.pkl     (round-trip test)
"""
import json
import pickle
import sys

import numpy as np


def _enc(o):
    if isinstance(o, np.ndarray):
        return {"__nd__": o.tolist()}
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, tuple):
        return {"__tuple__": [_enc(x) for x in o]}
    if isinstance(o, dict):
        return {(f"__int__{k}" if isinstance(k, (int, np.integer)) else str(k)): _enc(v)
                for k, v in o.items()}
    if isinstance(o, list):
        return [_enc(x) for x in o]
    return o


def _dec(o):
    if isinstance(o, dict):
        if "__nd__" in o:
            return np.array(o["__nd__"], dtype=float)
        if "__tuple__" in o:
            return tuple(_dec(x) for x in o["__tuple__"])
        return {(int(k[7:]) if k.startswith("__int__") else k): _dec(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_dec(x) for x in o]
    return o


def save(obj, path):
    with open(path, "w") as f:
        json.dump(_enc(obj), f)


def load(path):
    """A model file: .json (portable) or .pkl (same machine's numpy)."""
    if path.endswith(".json"):
        with open(path) as f:
            return _dec(json.load(f))
    with open(path, "rb") as f:
        return pickle.load(f)


def _same(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and np.allclose(a, b, equal_nan=True)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return abs(a - float(b)) < 1e-12 or (a != a and b != b)
    return a == b


if __name__ == "__main__":
    cmd, src = sys.argv[1], sys.argv[2]
    obj = pickle.load(open(src, "rb"))
    out = src[:-4] + ".json"
    save(obj, out)
    ok = _same(obj, load(out))
    print(f"{out}: round trip {'OK' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)
