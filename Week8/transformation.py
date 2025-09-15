##### transformation.py

"""
Transformation detection & application utilities for ARC-style grids.

Conventions (image/grid coordinates):
- rotate_90   == 90° clockwise
- rotate_180  == 180°
- rotate_270  == 90° counterclockwise
- mirror_x    == left-right flip (vertical axis)
- mirror_y    == up-down flip (horizontal axis)
- translate_dr_dc == shift by (dr rows, dc cols); e.g. translate_0_1 or translate(0,1)

Exports (backward compatible):
- detect_transformation(input_grid, expected_output, prefer_non_identity=True)
Extra helpers:
- apply_rule_to_grid(grid, rule)
- detect_rule_from_pairs(pairs, allowed=None, prefer_non_identity=True)
- list_supported_rules()
"""
from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import re
import numpy as np

Grid = np.ndarray

# --- Core transforms on 2D numpy arrays ---
def rotate_90(g: Grid) -> Grid:     # 90° clockwise
    return np.rot90(g, k=3)

def rotate_180(g: Grid) -> Grid:
    return np.rot90(g, k=2)

def rotate_270(g: Grid) -> Grid:    # 90° counterclockwise
    return np.rot90(g, k=1)

def mirror_x(g: Grid) -> Grid:      # left-right flip
    return np.fliplr(g)

def mirror_y(g: Grid) -> Grid:      # up-down flip
    return np.flipud(g)

# Optional extras (kept low priority)
def transpose_main(g: Grid) -> Grid:      # main diagonal
    return np.transpose(g)

def transpose_anti(g: Grid) -> Grid:      # anti-diagonal
    return np.fliplr(np.transpose(np.fliplr(g)))

def translate(g: Grid, dr: int, dc: int, fill: int = 0) -> Grid:
    """
    Shift grid by (dr, dc). Positive dr -> down, positive dc -> right.
    Cells shifted out are dropped; new cells filled with `fill`.
    """
    arr = np.asarray(g)
    H, W = arr.shape
    out = np.full_like(arr, fill)

    # source window
    src_r0 = max(0, -dr)
    src_r1 = min(H, H - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(W, W - dc)

    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out  # everything shifted out

    # destination window
    dst_r0 = src_r0 + dr
    dst_r1 = src_r1 + dr
    dst_c0 = src_c0 + dc
    dst_c1 = src_c1 + dc

    out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out

# Ordered registry for tie-breaking (identity last)
_TRANSFORMS: Dict[str, Callable[[Grid], Grid]] = {
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "rotate_270": rotate_270,
    "mirror_x": mirror_x,
    "mirror_y": mirror_y,
    "transpose_main": transpose_main,
    "transpose_anti": transpose_anti,
    "identity": lambda g: g,
}
# Note: translations are parameterized (translate_dr_dc) and therefore not in the fixed registry.

# Synonyms → canonical
_SYNONYMS = {
    "rotate_cw_90": "rotate_90",
    "rotate_ccw_90": "rotate_270",
    "flip_lr": "mirror_x",
    "flip_ud": "mirror_y",
    # Friendly translation names
    "shift_right_1": "translate_0_1",
    "shift_left_1":  "translate_0_-1",
    "shift_up_1":    "translate_-1_0",
    "shift_down_1":  "translate_1_0",
}

# --------- parsing & canonicalization for translation ----------
_TRANSLATE_RE_RC = re.compile(r"^translate\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$")
_TRANSLATE_RE_US = re.compile(r"^translate_(-?\d+)_(-?\d+)$")

def _parse_translate(rule: str) -> Optional[Tuple[int, int]]:
    n = (rule or "").strip().lower()
    m = _TRANSLATE_RE_RC.match(n)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _TRANSLATE_RE_US.match(n)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def canonical_rule_name(name: str) -> str:
    n = (name or "").strip().lower()
    # normalize translate(r,c) -> translate_r_c
    t = _parse_translate(n)
    if t is not None:
        dr, dc = t
        return f"translate_{dr}_{dc}"
    return _SYNONYMS.get(n, n)

def list_supported_rules() -> List[str]:
    # Keep the fixed list; translations are parametric and not enumerable here.
    return list(_TRANSFORMS.keys())

def apply_rule_to_grid(grid: Grid, rule: str) -> Grid:
    rule = canonical_rule_name(rule)
    t = _parse_translate(rule)
    if t is not None:
        dr, dc = t
        return translate(grid, dr, dc, fill=0)
    if rule not in _TRANSFORMS:
        raise ValueError(f"Unknown rule: {rule}")
    return _TRANSFORMS[rule](grid)

# --------- translation detection helpers ----------
def _detect_pure_translation(input_grid: Grid, expected_output: Grid) -> Optional[Tuple[int, int]]:
    """
    Return (dr, dc) if expected_output == translate(input_grid, dr, dc), else None.
    Fast check using nonzero bounding boxes, then verify by applying.
    """
    a = np.asarray(input_grid)
    b = np.asarray(expected_output)
    if a.shape != b.shape:
        return None
    if np.array_equal(a, b):
        return (0, 0)

    # if count of nonzeros differs, can't be a pure translation
    a_nz = np.nonzero(a)
    b_nz = np.nonzero(b)
    if len(a_nz[0]) != len(b_nz[0]):
        return None

    # both empty -> identity
    if len(a_nz[0]) == 0:
        return (0, 0)

    a_rmin, a_rmax = a_nz[0].min(), a_nz[0].max()
    a_cmin, a_cmax = a_nz[1].min(), a_nz[1].max()
    b_rmin, b_rmax = b_nz[0].min(), b_nz[0].max()
    b_cmin, b_cmax = b_nz[1].min(), b_nz[1].max()

    dr1 = b_rmin - a_rmin
    dr2 = b_rmax - a_rmax
    dc1 = b_cmin - a_cmin
    dc2 = b_cmax - a_cmax

    if dr1 != dr2 or dc1 != dc2:
        return None

    dr, dc = dr1, dc1
    if np.array_equal(translate(a, dr, dc, fill=0), b):
        return (dr, dc)
    return None

# ---------- main detectors ----------
def detect_transformation(input_grid: Grid,
                          expected_output: Grid,
                          prefer_non_identity: bool = True) -> str:
    """
    Backward-compatible single-pair detection.
    Returns canonical rule name or "unknown".
    """
    # 0) try pure translation first
    tr = _detect_pure_translation(input_grid, expected_output)
    if tr is not None:
        dr, dc = tr
        if (dr, dc) == (0, 0):
            return "identity" if prefer_non_identity is False else "identity"
        return f"translate_{dr}_{dc}"

    # 1) try fixed transforms
    candidates: List[str] = []
    for name, fn in _TRANSFORMS.items():
        try:
            if np.array_equal(fn(input_grid), expected_output):
                candidates.append(name)
        except Exception:
            pass

    if not candidates:
        return "unknown"
    if prefer_non_identity and len(candidates) > 1 and "identity" in candidates:
        candidates.remove("identity")
    return candidates[0]

def detect_rule_from_pairs(pairs: Sequence[Tuple[Grid, Grid]],
                           allowed: Optional[Iterable[str]] = None,
                           prefer_non_identity: bool = True) -> str:
    """
    Multi-pair detection: find a single rule that explains all pairs.
    Returns canonical rule name or "unknown".
    """
    tests = list(pairs)
    if not tests:
        return "unknown"

    # 0) If allowed explicitly contains a translate(...), prefer testing that exact one.
    if allowed is not None:
        for a in allowed:
            ca = canonical_rule_name(a)
            t = _parse_translate(ca)
            if t is not None:
                dr, dc = t
                ok = True
                for inp, out in tests:
                    try:
                        if not np.array_equal(translate(inp, dr, dc, fill=0), out):
                            ok = False; break
                    except Exception:
                        ok = False; break
                if ok:
                    if (dr, dc) == (0, 0) and prefer_non_identity and len(allowed) > 1:
                        pass  # keep searching
                    else:
                        return f"translate_{dr}_{dc}"

    # 1) Try to infer a single translation from the first pair and validate across all
    drdc = _detect_pure_translation(tests[0][0], tests[0][1])
    if drdc is not None:
        dr, dc = drdc
        all_ok = True
        for inp, out in tests:
            if not np.array_equal(translate(inp, dr, dc, fill=0), out):
                all_ok = False; break
        if all_ok:
            if (dr, dc) == (0, 0) and prefer_non_identity and len(tests) > 1:
                pass
            else:
                return f"translate_{dr}_{dc}"

    # 2) Fall back to fixed transforms (optionally filtered by 'allowed')
    names = list(_TRANSFORMS.keys()) if allowed is None else [canonical_rule_name(a) for a in allowed]
    for nm in names:
        fn = _TRANSFORMS.get(nm)
        if fn is None:
            continue
        ok = True
        for inp, out in tests:
            try:
                if not np.array_equal(fn(inp), out):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            if prefer_non_identity and nm == "identity" and len(names) > 1:
                continue
            return nm
    return "unknown"

if __name__ == "__main__":
    # Quick sanity checks
    a = np.array([[1,2,0],
                  [0,3,4],
                  [5,0,6]])
    b = np.array([[5,0,1],
                  [0,3,2],
                  [6,4,0]])  # rotate_90
    assert detect_transformation(a, b) == "rotate_90"
    assert (apply_rule_to_grid(a, "rotate_90") == b).all()

    # Translation tests
    g = np.array([[0,0,0,0,0],
                  [0,2,0,0,0],
                  [0,2,2,0,0],
                  [0,0,0,0,0],
                  [0,0,0,3,0]])
    g_r = translate(g, 0, 1)
    assert detect_transformation(g, g_r) == "translate_0_1"
    assert (apply_rule_to_grid(g, "translate(0,1)") == g_r).all()
    print("transformation.py OK")
