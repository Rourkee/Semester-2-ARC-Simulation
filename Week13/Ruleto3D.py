"""
3D target mapping for ARC rules + robot-friendly rotation helpers.

- execute_rule(scene, rule, grid_shape=None, strict=True)
    -> projects a 2D rule onto named scene objects (row/col → row/col).
- rotation_hint_rad(rule)
    -> minimal signed angle in radians (robot-friendly shortest turn)
- split_rotation_for_dual_knob(angle_rad, max_seg=pi/2)
    -> segments a rotation into <=90° chunks for robust execution
- collapse_to_cells(scene)
    -> keep one representative per cell; prefer _s1 if present (for stacks)
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import math
import re
import numpy as np

# Single source of truth for symbolic bits (names & translate parsing)
from transformation import canonical_rule_name, translation_hint_cells


# ------------------------ angle helpers ------------------------

def _wrap_to_pi(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

def _deg_to_rad_minimal(deg: float) -> float:
    return _wrap_to_pi(math.radians(deg))

def rotation_hint_rad(rule: str) -> Optional[float]:
    """
    Minimal signed angle (radians) that achieves the same logical rotation,
    choosing the shortest direction:
      - rotate_90  -> +pi/2
      - rotate_180 -> +pi
      - rotate_270 -> -pi/2
    Returns None for non-rotation rules.
    """
    if not rule:
        return None
    r = canonical_rule_name(rule)
    if r == "rotate_90":   return  _deg_to_rad_minimal( 90)
    if r == "rotate_180":  return  _deg_to_rad_minimal(180)
    if r == "rotate_270":  return  _deg_to_rad_minimal(270)
    if r == "identity":    return  0.0
    if r.startswith(("mirror_", "transpose_", "translate_")):
        return None

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
    """Split any rotation into <=90° segments (after minimal wrapping)."""
    a = _wrap_to_pi(float(angle_rad))
    if a == 0.0:
        return [0.0]
    segs: List[float] = []
    sgn = 1.0 if a > 0 else -1.0
    rem = abs(a)
    cap = max(1e-6, min(abs(float(max_seg)), math.pi))
    while rem > 1e-9:
        step = min(rem, cap)
        segs.append(sgn * step)
        rem -= step
    return segs


# ------------------------ grid helpers ------------------------

def _infer_shape_from_scene(scene: Dict[str, Dict[str, int]]) -> Tuple[int, int]:
    rows = [int(o.get("row", -1)) for o in scene.values() if "row" in o and "col" in o]
    cols = [int(o.get("col", -1)) for o in scene.values() if "row" in o and "col" in o]
    if not rows or not cols:
        return (0, 0)
    return max(rows) + 1, max(cols) + 1

def execute_rule(scene: Dict[str, Dict[str, int]],
                 rule: str,
                 grid_shape: Optional[Tuple[int, int]] = None,
                 *,
                 strict: bool = True) -> Dict[str, Dict[str, int]]:
    """
    Map each object's (row,col) under a 2D rule. No reasoning here—just projection.
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

    rname = canonical_rule_name(rule)
    out: Dict[str, Dict[str, int]] = {}

    # translations (explicit)
    t = translation_hint_cells(rname)
    if t is not None:
        dr, dc = t
        for name, obj in scene.items():
            r = int(obj.get("row", -1)); c = int(obj.get("col", -1))
            if not in_bounds(r, c):   # skip off-grid inputs
                continue
            nr, nc = within(r + dr, c + dc)
            out[name] = {"row": int(nr), "col": int(nc)}
        return out

    for name, obj in scene.items():
        r = int(obj.get("row", -1)); c = int(obj.get("col", -1))
        if not in_bounds(r, c):
            continue

        if rname == "mirror_x":            nr, nc = r, max_c - c
        elif rname == "mirror_y":          nr, nc = max_r - r, c
        elif rname == "rotate_90":         nr, nc = c, max_r - r
        elif rname == "rotate_180":        nr, nc = max_r - r, max_c - c
        elif rname == "rotate_270":        nr, nc = max_c - c, r
        elif rname == "transpose_main":    nr, nc = c, r
        elif rname == "transpose_anti":    nr, nc = max_c - c, max_r - r
        elif rname == "identity":          nr, nc = r, c
        else:
            m = re.fullmatch(r"rotate_(-?\d+)", rname)
            if m:
                deg = int(m.group(1)) % 360
                table = {0:"identity",90:"rotate_90",180:"rotate_180",270:"rotate_270"}
                if deg in table:
                    return execute_rule(scene, table[deg], grid_shape, strict=strict)
            raise ValueError(f"Unknown rule: {rname}")

        nr, nc = within(nr, nc)
        out[name] = {"row": int(nr), "col": int(nc)}
    return out


# ------------------------ cell representative picker ------------------------

_SLOT = re.compile(r"^(G\d+\d+)(?:_s\d+)?$")

def collapse_to_cells(scene: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str,int]]:
    """
    Keep one representative per cell; prefer _s1 if present, else the lexicographically first.
    Keys remain the chosen geom names.
    Works for both flat 'Grc' and stacked 'Grc_sK' geoms.
    """
    per_cell: Dict[str, Dict[str,int]] = {}
    by_root: Dict[str, List[str]] = {}
    for nm, obj in scene.items():
        m = _SLOT.match(nm or "")
        if not m:
            continue
        root = m.group(1)  # e.g., "G23"
        by_root.setdefault(root, []).append(nm)

    for root, names in by_root.items():
        pick = None
        # prefer root_s1, else smallest suffix, else first
        for nm in names:
            if nm.endswith("_s1"):
                pick = nm
                break
        if pick is None:
            pick = sorted(names, key=lambda x: (x.endswith("_s"), x))[0]
        per_cell[pick] = scene[pick]
    return per_cell


# ------------------------ self-test ------------------------

if __name__ == "__main__":
    scene = {"A": {"row": 0, "col": 0}, "B": {"row": 2, "col": 1}}
    res = execute_rule(scene, "rotate_90", (3,3))
    assert res["A"] == {"row": 0, "col": 2}
    assert res["B"] == {"row": 1, "col": 0}

    # collapse_to_cells quick check
    s = {"G11": {"row":1,"col":1}, "G11_s1": {"row":1,"col":1}, "G23_s2": {"row":2,"col":3}}
    c = collapse_to_cells(s)
    assert "G11_s1" in c and len(c) == 2

    print("Ruleto3D.py OK")
