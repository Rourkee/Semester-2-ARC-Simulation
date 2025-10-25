"""
Transformation + Program detection for ARC-style grids (SINGLE SOURCE OF TRUTH).

Conventions (image/grid coordinates):
- rotate_90   == 90° clockwise
- rotate_180  == 180°
- rotate_270  == 90° counterclockwise
- mirror_x    == left-right flip (vertical axis)
- mirror_y    == up-down flip (horizontal axis)
- translate_dr_dc == shift by (dr rows, dc cols); e.g. translate_0_1 or translate(0,1)

Exports (legacy, kept compatible):
- detect_transformation(input_grid, expected_output, prefer_non_identity=True)
- apply_rule_to_grid(grid, rule)
- detect_rule_from_pairs(pairs, allowed=None, prefer_non_identity=True)
- list_supported_rules()

Single-source "reasoning" API:
- detect_program(input_grid, expected_output) -> DetectedProgram(geom_rule, edits[, bulk_recolor])
- program_to_dsl(program) -> "seq(...)" string
- canonical_rule_name(name), translation_hint_cells(rule)

Parametric program templates (for TRAIN->TEST generalization):
- learn_program_template_from_pair(input_grid, expected_output) -> ProgramTemplate
- apply_program_template(input_grid, program_template) -> synthesized_output
- program_template_to_dsl(program_template) -> "seq(...)" string

This module now owns:
- Canonical names & parsing
- Geometry detection & best-alignment
- Residual edit program (adds/removes/recolors)
- Optional compression (bulk recolor)
- Parametric add/remove (relative to anchors) + absolute fallbacks
"""
from __future__ import annotations
from dataclasses import dataclass
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
    """Shift grid by (dr, dc). Positive dr -> down, positive dc -> right. Fill vacated with `fill`."""
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
# Note: translations are parametric (translate_dr_dc) and therefore not in the fixed registry.

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

def translation_hint_cells(rule: str) -> Optional[Tuple[int, int]]:
    """If rule encodes a translation, return (dr, dc); else None."""
    r = canonical_rule_name(rule)
    return _parse_translate(r)

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
    Allows off-grid losses caused by the shift (fill=0). Shapes must match.
    """
    a = np.asarray(input_grid)
    b = np.asarray(expected_output)
    if a.shape != b.shape:
        return None
    if np.array_equal(a, b):
        return (0, 0)

    # Find displacement using ANY color that appears in both grids.
    # (Works when colors are unique; if duplicates exist, the bbox trick below usually suffices.)
    a_nz = np.argwhere(a != 0)
    b_nz = np.argwhere(b != 0)
    if a_nz.size == 0 and b_nz.size == 0:
        return (0, 0)

    # 1) Try bounding-box displacement (robust, O(1))
    a_rmin, a_rmax = (a_nz[:,0].min(), a_nz[:,0].max()) if a_nz.size else (0, -1)
    a_cmin, a_cmax = (a_nz[:,1].min(), a_nz[:,1].max()) if a_nz.size else (0, -1)
    b_rmin, b_rmax = (b_nz[:,0].min(), b_nz[:,0].max()) if b_nz.size else (0, -1)
    b_cmin, b_cmax = (b_nz[:,1].min(), b_nz[:,1].max()) if b_nz.size else (0, -1)
    dr_bb = b_rmin - a_rmin
    dc_bb = b_cmin - a_cmin
    # If both ends shift consistently, that’s a good hint
    if (b_rmax - a_rmax) == dr_bb and (b_cmax - a_cmax) == dc_bb:
        if np.array_equal(translate(a, dr_bb, dc_bb, fill=0), b):
            return (dr_bb, dc_bb)

    # 2) Fallback: try a few displacements derived from first matches
    #    (Pick the first overlapping colored cell across both)
    H, W = a.shape
    # Build color→positions (keep at most a few samples per color)
    posA = {int(v): tuple(p) for v, p in [(int(a[r,c]), (r,c)) for r,c in a_nz] if v != 0}
    posB = {int(v): tuple(p) for v, p in [(int(b[r,c]), (r,c)) for r,c in b_nz] if v != 0}

    shared = [v for v in posA.keys() if v in posB]
    for v in shared[:4]:
        r0,c0 = posA[v]; r1,c1 = posB[v]
        dr = r1 - r0; dc = c1 - c0
        if np.array_equal(translate(a, dr, dc, fill=0), b):
            return (dr, dc)

    return None


# ---------- legacy single- & multi-pair detectors (kept for compatibility) ----------
def detect_transformation(input_grid: Grid,
                          expected_output: Grid,
                          prefer_non_identity: bool = True) -> str:
    """Backward-compatible single-pair detection of a single geometric transform."""
    tr = _detect_pure_translation(input_grid, expected_output)
    if tr is not None:
        dr, dc = tr
        if (dr, dc) == (0, 0):
            return "identity" if prefer_non_identity is False else "identity"
        return f"translate_{dr}_{dc}"

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
    """Legacy multi-pair single rule detection."""
    tests = list(pairs)
    if not tests:
        return "unknown"

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
                        pass
                    else:
                        return f"translate_{dr}_{dc}"

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

# ------------------------ supply/cleanup helpers ------------------------
def color_counts(grid: Grid) -> Dict[int, int]:
    """Count nonzero cell values (ARC colors) in a grid."""
    arr = np.asarray(grid)
    nz = arr[arr != 0]
    if nz.size == 0:
        return {}
    vals, cnts = np.unique(nz, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}

def color_deltas(input_grid: Grid,
                 expected_output: Grid) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Compare input vs. expected and compute:
      need:    colors with positive deficit (pick from dispenser), e.g., {6: +3}
      surplus: colors with positive surplus (send to bin),          e.g., {2: +1}
    """
    a = color_counts(input_grid)
    b = color_counts(expected_output)
    keys = set(a.keys()) | set(b.keys())
    need    = {k: b.get(k, 0) - a.get(k, 0) for k in keys if (b.get(k, 0) - a.get(k, 0)) > 0}
    surplus = {k: a.get(k, 0) - b.get(k, 0) for k in keys if (a.get(k, 0) - b.get(k, 0)) > 0}
    return need, surplus

# ======================= DetectedProgram (flat diffs) =========================
@dataclass
class EditOp:
    op: str                        # "add" | "remove" | "recolor"
    args: Tuple[int, ...]          # add(r,c,color) / remove(r,c) / recolor(r,c,from,to)

@dataclass
class DetectedProgram:
    geom_rule: str                 # canonical geometry name or "identity"
    edits: List[EditOp]            # residual inventory edits
    bulk_recolor: Optional[Tuple[int,int]] = None  # (from,to) if uniform recolor detected

def best_geom_alignment(a: Grid, b: Grid) -> str:
    """
    Choose one geometry (translate OR fixed transform OR identity) that best aligns A to B.
    - Prefer exact translation if possible.
    - Otherwise, score each fixed transform by matches (including zeros) and pick best.
    """
    tr = _detect_pure_translation(a, b)
    if tr is not None:
        dr, dc = tr
        return f"translate_{dr}_{dc}"

    best = ("identity", -1)
    for name, fn in _TRANSFORMS.items():
        try:
            aa = fn(a)
            score = int(np.sum(aa == b))  # include zeros to avoid degenerate ties
            if score > best[1]:
                best = (name, score)
        except Exception:
            pass
    return best[0]

def diff_edits_flat(aligned_inp: Grid, expected: Grid) -> List[EditOp]:
    """Produce a minimal per-cell edit list for flat grids (no stacking)."""
    H, W = expected.shape
    edits: List[EditOp] = []
    for r in range(H):
        for c in range(W):
            x = int(aligned_inp[r, c])
            y = int(expected[r, c])
            if x == y:
                continue
            if x == 0 and y != 0:
                edits.append(EditOp("add", (r, c, y)))
            elif x != 0 and y == 0:
                edits.append(EditOp("remove", (r, c)))
            else:
                edits.append(EditOp("recolor", (r, c, x, y)))
    return edits

def maybe_bulk_recolor(edits: List[EditOp]) -> Optional[Tuple[int,int]]:
    """
    If all edits are recolors with the same (from,to), compress to a bulk recolor.
    Returns (from,to) or None.
    """
    f = None; t = None
    if not edits:
        return None
    for e in edits:
        if e.op != "recolor":
            return None
        _, _, fr, to = e.args
        if f is None: f = fr
        if t is None: t = to
        if fr != f or to != t:
            return None
    return (f, t)

def detect_program(input_grid: Grid, expected_output: Grid) -> DetectedProgram:
    a = np.asarray(input_grid, int)
    b = np.asarray(expected_output, int)

    # 1) Pure translation?
    tr = _detect_pure_translation(a, b)
    if tr is not None:
        dr, dc = tr
        geom = f"translate_{dr}_{dc}"
        return DetectedProgram(geom_rule=geom, edits=[], bulk_recolor=None)

    # 2) Any fixed transform that matches exactly?
    for name, fn in _TRANSFORMS.items():
        try:
            if np.array_equal(fn(a), b):
                return DetectedProgram(geom_rule=name, edits=[], bulk_recolor=None)
        except Exception:
            pass

    # 3) Otherwise, pick the fixed transform with the fewest edits
    best = None  # (score, name, edits)
    for name, fn in _TRANSFORMS.items():
        try:
            al = fn(a)
            edits = diff_edits_flat(al, b)
            score = len(edits)  # you can weight adds/removes/recolors if you like
            if best is None or score < best[0]:
                best = (score, name, edits)
        except Exception:
            pass

    if best is None:
        # fallback: identity vs expected
        edits = diff_edits_flat(a, b)
        bulk = maybe_bulk_recolor(edits)
        return DetectedProgram(geom_rule="identity", edits=edits, bulk_recolor=bulk)

    _, name, edits = best
    bulk = maybe_bulk_recolor(edits)
    return DetectedProgram(geom_rule=name, edits=edits, bulk_recolor=bulk)


def program_to_dsl(p: DetectedProgram) -> str:
    """Pretty printer for logs/HUD, e.g., seq(rotate_90; recolor_where(value==1,to=6))."""
    parts: List[str] = []
    if p.geom_rule and p.geom_rule != "identity":
        parts.append(p.geom_rule)
    if p.bulk_recolor:
        f, t = p.bulk_recolor
        parts.append(f"recolor_where(value=={f},to={t})")
    else:
        for e in p.edits:
            if e.op == "add":
                r,c,color = e.args; parts.append(f"add({r},{c},{color})")
            elif e.op == "remove":
                r,c = e.args; parts.append(f"remove({r},{c})")
            elif e.op == "recolor":
                r,c,fr,to = e.args; parts.append(f"recolor({r},{c},{fr},{to})")
    return "seq(" + "; ".join(parts) + ")"

# ======================= ProgramTemplate (parametric) =========================
@dataclass
class AddRelative:
    anchor_color: int
    offsets: List[Tuple[int,int]]   # e.g., [(0,1),(0,2)]
    color: int                      # color to place

@dataclass
class RemoveRelative:
    anchor_color: int
    offsets: List[Tuple[int,int]]

@dataclass
class AddAbsolute:
    coords: List[Tuple[int,int]]
    color: int

@dataclass
class RemoveAbsolute:
    coords: List[Tuple[int,int]]

@dataclass
class ProgramTemplate:
    geom_rule: str
    bulk_recolor: Optional[Tuple[int,int]]         # (from,to) or None
    adds_rel: List[AddRelative]
    removes_rel: List[RemoveRelative]
    adds_abs: List[AddAbsolute]
    removes_abs: List[RemoveAbsolute]

# ---- helpers for template learning/apply ----
def positions_of_color(grid: np.ndarray, color: int) -> List[Tuple[int,int]]:
    rr, cc = np.where(grid == int(color))
    return list(zip(rr.tolist(), cc.tolist()))

def _fit_relative_offsets(
    aligned_inp: np.ndarray,
    expected: np.ndarray,
    adds: List[Tuple[int,int,int]],    # (r,c,color)
    removes: List[Tuple[int,int]],     # (r,c)
    max_unique_offsets: int = 3
) -> Tuple[List[AddRelative], List[RemoveRelative], List[Tuple[int,int,int]], List[Tuple[int,int]]]:
    """
    Try to explain adds/removes as offsets from an anchor color present in aligned_inp.
    Returns (adds_rel, removes_rel, leftover_adds, leftover_removes).
    """
    adds_rel: List[AddRelative] = []
    removes_rel: List[RemoveRelative] = []
    leftover_adds = adds[:]
    leftover_removes = removes[:]

    H, W = aligned_inp.shape
    anchor_colors = [int(c) for c in np.unique(aligned_inp) if c != 0]

    def nearest_anchor_offsets(anchor_color: int, points: List[Tuple[int,int]]):
        anchors = positions_of_color(aligned_inp, anchor_color)
        if not anchors:
            return None
        offsets = []
        for (r,c) in points:
            # nearest anchor by L1
            a_r, a_c = min(anchors, key=lambda p: abs(p[0]-r) + abs(p[1]-c))
            offsets.append((r - a_r, c - a_c))
        return offsets

    # === Adds ===
    if leftover_adds:
        # group by placed color for compact templates
        add_colors = list({col for (_,_,col) in leftover_adds})
        for ac in anchor_colors:
            for add_col in add_colors:
                pts = [(r,c) for (r,c,col) in leftover_adds if col == add_col]
                offs = nearest_anchor_offsets(ac, pts)
                if offs is None:
                    continue
                uniq = sorted(set(offs))
                if len(uniq) <= max_unique_offsets:
                    adds_rel.append(AddRelative(anchor_color=ac, offsets=uniq, color=add_col))
                    covered = set(pts)
                    leftover_adds = [(r,c,col) for (r,c,col) in leftover_adds if (r,c) not in covered]
                    # continue searching to maybe cover more colors/anchors

    # === Removes ===
    if leftover_removes:
        for ac in anchor_colors:
            pts = list(leftover_removes)
            offs = nearest_anchor_offsets(ac, pts)
            if offs is None:
                continue
            uniq = sorted(set(offs))
            if len(uniq) <= max_unique_offsets:
                removes_rel.append(RemoveRelative(anchor_color=ac, offsets=uniq))
                covered = set(pts)
                leftover_removes = [(r,c) for (r,c) in leftover_removes if (r,c) not in covered]
                break

    return adds_rel, removes_rel, leftover_adds, leftover_removes

def learn_program_template_from_pair(input_grid: np.ndarray, expected_output: np.ndarray) -> ProgramTemplate:
    """
    Learn a parametric program (geom + relative edits + absolute fallbacks) from a single pair.
    Works best when additions/removals are consistently related to anchors in the input.
    """
    a = np.asarray(input_grid, int)
    b = np.asarray(expected_output, int)

    # 1) choose geometry
    geom = best_geom_alignment(a, b)
    aligned = apply_rule_to_grid(a, geom)

    # 2) compute flat diffs (adds/removes/recolors)
    H, W = b.shape
    adds: List[Tuple[int,int,int]] = []
    removes: List[Tuple[int,int]] = []
    recolors: List[Tuple[int,int,int,int]] = []
    for r in range(H):
        for c in range(W):
            x = int(aligned[r,c]); y = int(b[r,c])
            if x == y: continue
            if x == 0 and y != 0: adds.append((r,c,y))
            elif x != 0 and y == 0: removes.append((r,c))
            else: recolors.append((r,c,x,y))

    # 3) compress recolors to bulk if possible
    bulk = None
    if recolors:
        frs = {fr for (_,_,fr,_) in recolors}
        tos = {to for (_,_,_,to) in recolors}
        if len(frs) == 1 and len(tos) == 1:
            bulk = (list(frs)[0], list(tos)[0])
            recolors.clear()  # handled by bulk recolor

    # 4) try parametric relative templates for adds/removes
    adds_rel, removes_rel, leftover_adds, leftover_removes = _fit_relative_offsets(
        aligned, b, adds, removes, max_unique_offsets=3
    )

    # 5) fall back to absolute for leftovers
    adds_abs: List[AddAbsolute] = []
    removes_abs: List[RemoveAbsolute] = []
    if leftover_adds:
        byc: Dict[int, List[Tuple[int,int]]] = {}
        for r,c,col in leftover_adds:
            byc.setdefault(col, []).append((r,c))
        for col, coords in byc.items():
            adds_abs.append(AddAbsolute(coords=coords, color=col))
    if leftover_removes:
        removes_abs.append(RemoveAbsolute(coords=leftover_removes))

    return ProgramTemplate(
        geom_rule=geom,
        bulk_recolor=bulk,
        adds_rel=adds_rel,
        removes_rel=removes_rel,
        adds_abs=adds_abs,
        removes_abs=removes_abs,
    )

def apply_program_template(input_grid: np.ndarray, prog: ProgramTemplate) -> np.ndarray:
    """
    Apply a learned program template to any input to synthesize the expected output.
    """
    g = np.asarray(input_grid, int)
    H, W = g.shape
    out = apply_rule_to_grid(g, prog.geom_rule).copy()

    def inb(r,c): return 0 <= r < H and 0 <= c < W

    # bulk recolor
    if prog.bulk_recolor:
        fr, to = prog.bulk_recolor
        out[out == fr] = to

    # relative adds
    for tpl in prog.adds_rel:
        anchors = positions_of_color(out, tpl.anchor_color)
        for (r0,c0) in anchors:
            for (dr,dc) in tpl.offsets:
                r, c = r0 + dr, c0 + dc
                if inb(r,c) and out[r,c] == 0:
                    out[r,c] = tpl.color

    # relative removes
    for tpl in prog.removes_rel:
        anchors = positions_of_color(out, tpl.anchor_color)
        for (r0,c0) in anchors:
            for (dr,dc) in tpl.offsets:
                r, c = r0 + dr, c0 + dc
                if inb(r,c):
                    out[r,c] = 0

    # absolute adds/removes
    for tpl in prog.adds_abs:
        for (r,c) in tpl.coords:
            if inb(r,c) and out[r,c] == 0:
                out[r,c] = tpl.color
    for tpl in prog.removes_abs:
        for (r,c) in tpl.coords:
            if inb(r,c):
                out[r,c] = 0

    return out

def program_template_to_dsl(p: ProgramTemplate) -> str:
    """Readable summary for logs/HUD."""
    parts = []
    if p.geom_rule and p.geom_rule != "identity":
        parts.append(p.geom_rule)
    if p.bulk_recolor:
        f,t = p.bulk_recolor
        parts.append(f"recolor_where(value=={f},to={t})")
    for a in p.adds_rel:
        parts.append(f"add_rel(anchor={a.anchor_color}, offsets={a.offsets}, color={a.color})")
    for r in p.removes_rel:
        parts.append(f"remove_rel(anchor={r.anchor_color}, offsets={r.offsets})")
    for a in p.adds_abs:
        parts.append(f"add_abs(coords={a.coords}, color={a.color})")
    for r in p.removes_abs:
        parts.append(f"remove_abs(coords={r.coords})")
    return "seq(" + "; ".join(parts) + ")"



def _maybe_stack_prefix_1(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if for all cells with a>0, b == int('1'+str(a)) and unchanged zeros."""
    if a.shape != b.shape: 
        return False
    for (x, y) in zip(a.flat, b.flat):
        x = int(x); y = int(y)
        if x == 0 and y == 0:
            continue
        if x == 0 and y != 0:
            return False
        # x > 0:
        try:
            if y != int("1" + str(x)):
                return False
        except Exception:
            return False
    return True

def _maybe_stack_tower_1_to_v(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if for all cells with a>0, b encodes [1..a] (top→down) and unchanged zeros."""
    if a.shape != b.shape:
        return False
    for (x, y) in zip(a.flat, b.flat):
        x = int(x); y = int(y)
        if x == 0 and y == 0:
            continue
        if x == 0 and y != 0:
            return False
        # x > 0:
        try:
            target = int("".join(str(k) for k in range(1, x + 1)))
            if y != target:
                return False
        except Exception:
            return False
    return True

def apply_stack_prefix_1(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, int).copy()
    nz = g > 0
    vec = np.vectorize(lambda v: int("1" + str(int(v))))
    g[nz] = vec(g[nz])
    return g

def apply_stack_tower_1_to_v(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, int).copy()
    H, W = g.shape
    out = np.zeros_like(g)
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v > 0:
                out[r, c] = int("".join(str(k) for k in range(1, v + 1)))
    return out






# ------------------------ quick self-test ------------------------
if __name__ == "__main__":
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

    # Deltas tests
    need, surplus = color_deltas(np.zeros((3,3), int),
                                 np.array([[0,6,0],[0,6,0],[0,0,0]]))
    assert need == {6: 2} and surplus == {}

    # Program (flat) tests
    p = detect_program(np.array([[1,0],[0,0]]), np.array([[6,0],[0,0]]))
    assert p.geom_rule == "identity" and p.bulk_recolor == (1,6)

    # Template learn/apply test (simple)
    t_in = np.array([[0,1,0],
                     [0,0,0],
                     [0,0,0]])
    t_out = np.array([[0,1,6],
                      [0,0,0],
                      [0,0,0]])  # "add 6 at (0,+1) relative to anchor 1"
    tpl = learn_program_template_from_pair(t_in, t_out)
    synth = apply_program_template(t_in, tpl)
    assert np.array_equal(synth, t_out)
    print("transformation.py OK")
