##### plotting.py
# compare_physical_symbolic.py
# Compare physical execution vs symbolic plan for a scene_memory run.
# Exposes export_run_pngs() for saving PNGs programmatically.
# This version:
# - Transparently reads .csv or .csv.gz in scene_memory/_raw/<run_tag>/
# - Fixes duplicate filename bug (separate filenames for 3D vs 2D)
# - Keeps headless-safe matplotlib backend

import os, glob, csv, json, argparse, time, gzip
from collections import defaultdict

# Use headless-safe backend
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------- utilities ----------

def list_runs(root="scene_memory"):
    runs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and os.path.basename(d) != "_raw"]
    runs.sort()
    return runs

def pick_run_interactively(root="scene_memory"):
    runs = list_runs(root)
    if not runs:
        raise FileNotFoundError(f"No runs found in '{root}/'.")
    print("Available scenes:")
    for i, r in enumerate(runs):
        print(f"[{i}] {os.path.basename(r)}")
    while True:
        try:
            idx = int(input("Select scene index: ").strip())
            if 0 <= idx < len(runs):
                return runs[idx]
        except Exception:
            pass
        print("Invalid selection. Try again.")

def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def read_csv_rows_gz(path_gz):
    with gzip.open(path_gz, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def col_as_float(rows, key):
    return np.array([safe_float(r.get(key, "nan")) for r in rows], dtype=float)


# ---------- data loading ----------

def _heavy_dir_of(run_dir, heavy_subdir="_raw"):
    root = os.path.dirname(run_dir)
    tag = os.path.basename(run_dir)
    return os.path.join(root, heavy_subdir, tag)

def _resolve_heavy_file(run_dir, filename):
    # Prefer heavy dir .csv, then .csv.gz
    heavy_dir = _heavy_dir_of(run_dir)
    p_csv = os.path.join(heavy_dir, filename)
    p_gz  = p_csv + ".gz"
    if os.path.exists(p_csv):
        return p_csv
    if os.path.exists(p_gz):
        return p_gz
    # fall back to run_dir (legacy plain placement)
    p_legacy = os.path.join(run_dir, filename)
    if os.path.exists(p_legacy):
        return p_legacy
    return None

def load_trajectory(run_dir):
    p = _resolve_heavy_file(run_dir, "trajectory.csv")
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"trajectory.csv not found for run '{run_dir}' (checked heavy dir and run dir).")
    rows = read_csv_rows_gz(p) if p.endswith(".gz") else read_csv_rows(p)
    t = col_as_float(rows, "t")
    x = col_as_float(rows, "x")
    y = col_as_float(rows, "y")
    z = col_as_float(rows, "z")
    phase = [r.get("phase", "") for r in rows]
    return dict(t=t, x=x, y=y, z=z, phase=phase)

def load_events(run_dir):
    p = os.path.join(run_dir, "events.csv")
    if not os.path.exists(p):
        return []
    rows = read_csv_rows(p)
    events = []
    for r in rows:
        events.append({
            "t": safe_float(r.get("t")),
            "kind": r.get("kind", ""),
            "label": r.get("label", ""),
            "x": safe_float(r.get("x")),
            "y": safe_float(r.get("y")),
            "z": safe_float(r.get("z")),
            "row": r.get("row"),
            "col": r.get("col"),
        })
    return events

def load_grids(run_dir):
    p = os.path.join(run_dir, "ARCtask.json")
    if not os.path.exists(p):
        # Some older versions stored metadata in Logistics.json only
        p2 = os.path.join(run_dir, "Logistics.json")
        if os.path.exists(p2):
            with open(p2, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- pairing logic (pick -> place) ----------

def pair_moves(events):
    moves = []
    pending = defaultdict(list)
    events_sorted = sorted(events, key=lambda e: (np.nan_to_num(e["t"]), e.get("kind", "")))
    for ev in events_sorted:
        if ev["kind"] == "pick":
            pending[ev["label"].strip()].append(ev)
        elif ev["kind"] == "place":
            lbl = ev["label"].strip()
            if pending[lbl]:
                pk = pending[lbl].pop(0)
                mv = {
                    "label": lbl,
                    "pick": pk,
                    "place": ev,
                    "dx": ev["x"] - pk["x"],
                    "dy": ev["y"] - pk["y"],
                    "dz": ev["z"] - pk["z"],
                    "cell_from": f'{pk.get("row")},{pk.get("col")}' if pk.get("row") and pk.get("col") else "",
                    "cell_to":   f'{ev.get("row")},{ev.get("col")}' if ev.get("row") and ev.get("col") else "",
                }
                moves.append(mv)
    return moves


# ---------- plotting helpers ----------

def set_equal_3d(ax, x, y, z):
    xs = x[np.isfinite(x)]; ys = y[np.isfinite(y)]; zs = z[np.isfinite(z)]
    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        return
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min())
    cx, cy, cz = xs.mean(), ys.mean(), zs.mean()
    r = max_range / 2 if max_range > 0 else 0.1
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_zlim(cz - r, cz + r)

def plot_3d_path_with_moves(run_dir, traj, moves):
    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(traj["x"], traj["y"], traj["z"], linewidth=2, label="EE path")
    if len(traj["x"]) > 0:
        ax.scatter(traj["x"][0], traj["y"][0], traj["z"][0], s=40, label="start")
        ax.scatter(traj["x"][-1], traj["y"][-1], traj["z"][-1], s=40, label="end")

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title(f"Physical execution — {os.path.basename(run_dir)}")
    ax.legend(loc="best")
    set_equal_3d(ax, traj["x"], traj["y"], traj["z"])
    plt.tight_layout()
    return fig

def plot_symbolic_grid(run_dir, grids, moves):
    if not grids:
        return None
    expected = np.array(grids.get("expected_output", []))
    if expected.size == 0:
        return None

    rows, cols = expected.shape
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for r in range(rows + 1):
        ax.plot([0, cols], [r, r], linewidth=0.5)
    for c in range(cols + 1):
        ax.plot([c, c], [0, rows], linewidth=0.5)

    ax.set_xlim(0, cols); ax.set_ylim(rows, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Symbolic plan: cell_from → cell_to")

    for mv in moves:
        if not mv["cell_from"] or not mv["cell_to"]:
            continue
        r0, c0 = [int(v) for v in mv["cell_from"].split(",")]
        r1, c1 = [int(v) for v in mv["cell_to"].split(",")]
        x0, y0 = c0 + 0.5, r0 + 0.5
        x1, y1 = c1 + 0.5, r1 + 0.5
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.8))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, mv["label"] or "", fontsize=8, ha="center", va="center")

    ax.set_xlabel("col"); ax.set_ylabel("row")
    plt.tight_layout()
    return fig


# ---------- metrics ----------

def path_length(traj):
    x, y, z = traj["x"], traj["y"], traj["z"]
    if len(x) < 2:
        return 0.0
    dp = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    return float(np.nansum(dp))

def total_pick_place_distance(moves):
    if not moves:
        return 0.0
    d = 0.0
    for mv in moves:
        d += float(np.sqrt(mv["dx"] ** 2 + mv["dy"] ** 2 + mv["dz"] ** 2))
    return d


# ---------- export API ----------

def export_run_pngs(run_dir, *, timestamp=True, show=False):
    traj = load_trajectory(run_dir)
    events = load_events(run_dir)
    grids = load_grids(run_dir)

    moves = pair_moves(events)
    fig3d = plot_3d_path_with_moves(run_dir, traj, moves)
    fig2d = plot_symbolic_grid(run_dir, grids, moves)

    tag = time.strftime("%Y%m%d-%H%M%S") if timestamp else "latest"
    out3d = os.path.join(run_dir, f"PhysicalPath_{tag}.png")
    fig3d.savefig(out3d, dpi=150)

    out2d = None
    if fig2d is not None:
        out2d = os.path.join(run_dir, f"SymbolicPlan_{tag}.png")
        fig2d.savefig(out2d, dpi=150)

    if show:
        plt.show()

    plt.close(fig3d)
    if fig2d is not None:
        plt.close(fig2d)

    return {"3d": out3d, "2d": out2d}


# ---------- CLI ----------

def main(run_dir=None):
    if not run_dir:
        run_dir = pick_run_interactively("scene_memory")

    traj = load_trajectory(run_dir)
    events = load_events(run_dir)
    grids = load_grids(run_dir)

    moves = pair_moves(events)
    fig3d = plot_3d_path_with_moves(run_dir, traj, moves)
    fig2d = plot_symbolic_grid(run_dir, grids, moves)

    ee_len = path_length(traj)
    mp_len = total_pick_place_distance(moves)
    overhead = (ee_len / mp_len - 1.0) * 100.0 if mp_len > 1e-9 else np.nan

    print(f"[metrics] EE path length:        {ee_len:.3f} m")
    print(f"[metrics] Sum pick→place dists:  {mp_len:.3f} m")
    if np.isfinite(overhead):
        print(f"[metrics] Overhead vs direct:   {overhead:+.1f}%")

    tag = time.strftime("%Y%m%d-%H%M%S")
    out3d = os.path.join(run_dir, f"PhysicalPath_{tag}.png")
    fig3d.savefig(out3d, dpi=150)

    if fig2d is not None:
        out2d = os.path.join(run_dir, f"SymbolicPlan_{tag}.png")
        fig2d.savefig(out2d, dpi=150)

    plt.close(fig3d)
    if fig2d is not None:
        plt.close(fig2d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="", help="scene_memory/<time_tag>")
    args = ap.parse_args()
    main(args.run_dir or None)
