"""
3D target mapping for ARC rules + robot-friendly rotation/translation helpers.

- execute_rule(scene, rule, grid_shape=None, strict=True)
    -> grid mapping supporting quarter-turn rotations, mirrors, transposes, and translations.
- rotation_hint_rad(rule)
    -> minimal signed angle in radians (e.g., 270° -> -pi/2)
- split_rotation_for_dual_knob(angle_rad, max_seg=pi/2)
    -> list of <=90° segments for dual-knob execution
- translation_hint_cells(rule)
    -> (dr, dc) if rule is a translation; else None
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import math
import re
import numpy as np

# Synonyms (mirror transformation.py)
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

def canonical_rule_name(rule: str) -> str:
    n = (rule or "").strip().lower()
    # normalize translate(r,c) -> translate_r_c
    t = _parse_translate(n)
    if t is not None:
        dr, dc = t
        return f"translate_{dr}_{dc}"
    return _SYNONYMS.get(n, n)

def _infer_shape_from_scene(scene: Dict[str, Dict[str, int]]) -> Tuple[int, int]:
    rows = [int(o["row"]) for o in scene.values()] or [0]
    cols = [int(o["col"]) for o in scene.values()] or [0]
    H = max(rows) + 1
    W = max(cols) + 1
    return H, W

# ------------------------ robot helpers (rotation) ------------------------

def _wrap_to_pi(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

def _deg_to_rad_minimal(deg: float) -> float:
    """Degrees -> radians, wrapped to (-pi, pi]."""
    return _wrap_to_pi(math.radians(deg))

def rotation_hint_rad(rule: str) -> Optional[float]:
    """
    Return a minimal signed angle (radians) that achieves the same logical rotation,
    choosing the *shortest* direction:
      - rotate_90  -> +pi/2
      - rotate_180 ->  +pi
      - rotate_270 -> -pi/2   (shortest: -90)
    Also supports 'rotate_<deg>' (any integer degrees) and 'rotate_rad_<float>'.

    Returns None for non-rotation rules.
    """
    if not rule:
        return None
    r = canonical_rule_name(rule)

    # Exact tokens
    if r == "rotate_90":   return  _deg_to_rad_minimal( 90)
    if r == "rotate_180":  return  _deg_to_rad_minimal(180)
    if r == "rotate_270":  return  _deg_to_rad_minimal(270)  # -> -90°
    if r == "identity":    return  0.0
    if r.startswith("mirror_") or r.startswith("transpose_") or r.startswith("translate_"):
        return None

    # Generic: rotate_<deg> or rotate_rad_<float>
    m = re.fullmatch(r"rotate_(-?\d+)", r)
    if m:
        deg = float(m.group(1))
        return _deg_to_rad_minimal(deg)

    m = re.fullmatch(r"rotate_rad_(-?\d*\.?\d+)", r)
    if m:
        ang = float(m.group(1))
        return _wrap_to_pi(ang)

    return None

def split_rotation_for_dual_knob(angle_rad: float, max_seg: float = math.pi/2) -> List[float]:
    """
    Split any angle into segments whose magnitudes are <= max_seg (default: 90°),
    after first wrapping to the minimal signed angle (-pi, pi].
    """
    a = _wrap_to_pi(float(angle_rad))
    if a == 0.0:
        return [0.0]

    segs: List[float] = []
    sgn = 1.0 if a > 0 else -1.0
    rem = abs(a)
    cap = abs(float(max_seg))
    cap = max(1e-6, min(cap, math.pi))  # sanity

    while rem > 1e-9:
        step = min(rem, cap)
        segs.append(sgn * step)
        rem -= step
    return segs

# ------------------------ robot helpers (translation) ------------------------

_TRANSLATE_RE_RC = re.compile(r"^translate\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$")
_TRANSLATE_RE_US = re.compile(r"^translate_(-?\d+)_(-?\d+)$")

def _parse_translate(rule: str) -> Optional[Tuple[int, int]]:
    """Parse 'translate(r,c)' or 'translate_r_c' -> (dr, dc). Else None."""
    n = (rule or "").strip().lower()
    m = _TRANSLATE_RE_RC.match(n)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _TRANSLATE_RE_US.match(n)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def translation_hint_cells(rule: str) -> Optional[Tuple[int, int]]:
    """
    If rule encodes a translation, return (dr, dc) in *grid cells*.
    Positive dr -> down; positive dc -> right. Otherwise None.
    """
    r = canonical_rule_name(rule)
    return _parse_translate(r)

# ------------------------ ARC grid mapping ------------------------

def execute_rule(scene: Dict[str, Dict[str, int]],
                 rule: str,
                 grid_shape: Optional[Tuple[int, int]] = None,
                 *,
                 strict: bool = True) -> Dict[str, Dict[str, int]]:
    """
    Return new (row, col) for each object after applying rule.

    - Skips objects whose INPUT (row,col) are out-of-bounds (e.g., -1,-1 from extractor).
    - Still enforces bounds on OUTPUT when `strict=True`.
    """
    H, W = grid_shape if grid_shape is not None else _infer_shape_from_scene(scene)
    max_r, max_c = H - 1, W - 1

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r <= max_r and 0 <= c <= max_c

    def within(r: int, c: int) -> Tuple[int, int]:
        if 0 <= r <= max_r and 0 <= c <= max_c:
            return r, c
        if strict:
            raise ValueError(f"Target (r={r}, c={c}) out of bounds for grid {(H, W)}")
        return max(0, min(r, max_r)), max(0, min(c, max_c))

    rule = canonical_rule_name(rule)

    # --- handle translation rules first (translate_<dr>_<dc>)
    m = re.fullmatch(r"translate_(-?\d+)_(-?\d+)", rule)
    if m:
        dr = int(m.group(1)); dc = int(m.group(2))
        out: Dict[str, Dict[str, int]] = {}
        for name, obj in scene.items():
            r = int(obj.get("row", -1)); c = int(obj.get("col", -1))
            # Skip off-grid inputs (e.g., -1,-1)
            if not in_bounds(r, c):
                continue
            nr, nc = r + dr, c + dc
            nr, nc = within(nr, nc)  # may raise if strict and OOB
            out[name] = {"row": int(nr), "col": int(nc)}
        return out

    out: Dict[str, Dict[str, int]] = {}
    for name, obj in scene.items():
        r = int(obj.get("row", -1)); c = int(obj.get("col", -1))
        # Skip off-grid inputs (e.g., -1,-1)
        if not in_bounds(r, c):
            continue

        if rule == "mirror_x":            # left-right
            nr, nc = r, max_c - c
        elif rule == "mirror_y":          # up-down
            nr, nc = max_r - r, c
        elif rule == "rotate_90":         # 90° clockwise
            nr, nc = c, max_r - r
        elif rule == "rotate_180":        # 180°
            nr, nc = max_r - r, max_c - c
        elif rule == "rotate_270":        # 90° counterclockwise
            nr, nc = max_c - c, r
        elif rule == "transpose_main":
            nr, nc = c, r
        elif rule == "transpose_anti":
            nr, nc = max_c - c, max_r - r
        elif rule == "identity":
            nr, nc = r, c
        else:
            # try generic rotate_<deg> normalization to quarter turns
            m2 = re.fullmatch(r"rotate_(-?\d+)", rule)
            if m2:
                deg = int(m2.group(1)) % 360
                opts = {0: "identity", 90: "rotate_90", 180: "rotate_180", 270: "rotate_270"}
                if deg in opts:
                    # recurse with normalized
                    return execute_rule(scene, opts[deg], grid_shape, strict=strict)
            raise ValueError(f"Unknown rule: {rule}")

        nr, nc = within(nr, nc)
        out[name] = {"row": int(nr), "col": int(nc)}

    return out


# ------------------------ supply/cleanup reasoning helpers ------------------------

def color_counts(grid: np.ndarray) -> Dict[int, int]:
    """
    Count nonzero cell values (ARC colors) in a grid.
    Returns {color: count}. Zeros are ignored.
    """
    arr = np.asarray(grid)
    nz = arr[arr != 0]
    if nz.size == 0:
        return {}
    vals, cnts = np.unique(nz, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}

def color_deltas_from_grids(input_grid: np.ndarray,
                            expected_output: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Compare input vs. expected grids and compute:
      - need:   colors with positive deficit (must PICK from dispenser),   e.g., {6: +3}
      - surplus:colors with positive surplus (can BIN),                    e.g., {2: +1}
    """
    a = color_counts(input_grid)
    b = color_counts(expected_output)
    keys = set(a.keys()) | set(b.keys())
    need    = {k: b.get(k, 0) - a.get(k, 0) for k in keys if (b.get(k, 0) - a.get(k, 0)) > 0}
    surplus = {k: a.get(k, 0) - b.get(k, 0) for k in keys if (a.get(k, 0) - b.get(k, 0)) > 0}
    return need, surplus


# ------------------------ quick self-test ------------------------
if __name__ == "__main__":
    scene = {"A": {"row": 0, "col": 0}, "B": {"row": 2, "col": 1}}
    res = execute_rule(scene, "rotate_90", (3,3))
    assert res["A"] == {"row": 0, "col": 2}
    assert res["B"] == {"row": 1, "col": 0}
    # Rotation helpers sanity
    assert abs(rotation_hint_rad("rotate_270") - (-math.pi/2)) < 1e-9
    assert split_rotation_for_dual_knob(3*math.pi/4) == [math.pi/2, math.pi/4]

    # NEW: deltas sanity
    g0 = np.zeros((3,3), int); g1 = g0.copy(); g1[0,0] = 6; g1[1,1] = 2
    need, surplus = color_deltas_from_grids(g0, g1)
    assert need == {6: 1, 2: 1} and surplus == {}
    print("Ruleto3D.py OK")
