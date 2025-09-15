##### dynamicworld.py

from __future__ import annotations
from typing import Tuple, Union, Optional, Sequence

def generate_world_xml(
    grid_shape: Union[Tuple[int, int], Tuple[int], int],
    write_path: str = "world.xml",
    *,
    grid_values: Optional[Sequence[Sequence[int]]] = None,   # 0 -> do not spawn
    include_rotation: bool = True,  # ignored; world is always rotatable-only
    cell: float = 0.10,
    plate_center: Tuple[float, float] = (0.4, 0.0),
    world_offset: Tuple[float, float, float] = (0.1, 0.0, 0.0),
    plate_rim: float = 0.02,
    frame_extra: float = 0.01,
    plate_thickness: float = 0.010,
    block_size: float = 0.03,
    handle_radius: float = 0.012,
    handle_height: float = 0.050,
    hub_radius: float = 0.030,
    hub_thickness: float = 0.003,
    cell_scale: float = 0.7,
    grid_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> None:
    # --- normalize (H, W) ---
    if isinstance(grid_shape, int):
        H = W = int(grid_shape)
    else:
        try:
            if len(grid_shape) == 1:
                H = W = int(grid_shape[0])
            else:
                H, W = map(int, grid_shape[:2])
        except Exception:
            raise ValueError(f"grid_shape must be int, (n,), or (H,W); got {grid_shape!r}")
    assert 1 <= H <= 6 and 1 <= W <= 6

    # Optional values (spawn only nonzeros)
    gv: Optional[list[list[int]]] = None
    if grid_values is not None:
        try:
            gv = [[int(grid_values[r][c]) for c in range(W)] for r in range(H)]
        except Exception as e:
            raise ValueError(f"`grid_values` must be indexable with shape ({H},{W}): {e}")

    # Nonzero palette (0 is intentionally absent)
    palette = {
        1: (1.0, 0.0, 0.0, 1.0),
        2: (0.0, 1.0, 0.0, 1.0),
        3: (0.0, 0.0, 1.0, 1.0),
        4: (1.0, 1.0, 0.0, 1.0),
        5: (1.0, 0.0, 1.0, 1.0),
        6: (0.0, 1.0, 1.0, 1.0),
        7: (1.0, 0.5, 0.0, 1.0),
        8: (0.5, 0.0, 1.0, 1.0),
        9: (1.0, 1.0, 1.0, 1.0),
    }

    # Geometry layout
    effective_cell = float(cell) * float(cell_scale)
    half_cell = 0.5 * effective_cell
    half_cells_x = 0.5 * max(W - 1, 0) * effective_cell
    half_cells_y = 0.5 * max(H - 1, 0) * effective_cell

    frame_half_w = (half_cells_x + half_cell) + float(frame_extra)
    frame_half_h = (half_cells_y + half_cell) + float(frame_extra)

    slab_thick = float(plate_thickness) * 0.5
    block_half = float(block_size) * 0.5

    ox, oy, oz = map(float, world_offset)
    cx, cy = plate_center
    cx = float(cx) + ox
    cy = float(cy) + oy

    plate_z_mid = slab_thick + oz
    plate_top_z = plate_z_mid + slab_thick

    half_diag = (frame_half_w**2 + frame_half_h**2) ** 0.5
    pad = max(0.005, 0.5 * float(plate_rim))
    plate_radius = half_diag + pad

    grid_eps = 0.0030
    grid_z = plate_top_z + grid_eps
    block_z_world = plate_top_z + block_half

    grid_x0 = cx - half_cells_x
    grid_y0 = cy - half_cells_y

    handle_rad = float(handle_radius)
    handle_half = float(handle_height) * 0.5
    hub_rad   = float(hub_radius)
    hub_half  = float(hub_thickness) * 0.5

    grid_extent = max(frame_half_w, frame_half_h)
    knob_r = min(plate_radius - 0.015, grid_extent + 0.015)

    # Memory sizing (count only spawned)
    if gv is None:
        N_blocks = H * W
    else:
        N_blocks = sum(1 for r in range(H) for c in range(W) if gv[r][c] != 0)

    N_table_lines = (H + 1) + (W + 1)
    N_geoms = 1 + 1 + 1 + 1 + N_table_lines + N_blocks
    nj = 1 + N_blocks
    ncon = max(512, 4 * N_blocks + 128)
    nstack = max(22000, 80 * (N_geoms + nj))

    xml = f"""<mujoco>
  <compiler angle="radian" meshdir="./mesh" texturedir="./texture"/>
  <visual>
    <quality shadowsize="4096" offsamples="4" numslices="32" numstacks="32"/>
    <headlight active="1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <option timestep="0.002" integrator="Euler" iterations="60" solver="Newton"
          impratio="5" noslip_iterations="6" tolerance="1e-8" cone="elliptic">
    <flag actuation="enable" gravity="enable" warmstart="enable"/>
  </option>

  <size nstack="{nstack}" njmax="{nj + 32}" nconmax="{ncon}"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="plane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2"
             width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="plane" reflectance="0.3" texture="plane" texrepeat="1 1" texuniform="true"/>

    <material name="knob_metal" rgba="0.35 0.35 0.38 1" specular="0.4" shininess="0.6"/>
    <material name="grid_line" rgba="{grid_rgba[0]} {grid_rgba[1]} {grid_rgba[2]} {grid_rgba[3]}"/>
    <material name="block_mat" rgba="0.9 0.4 0.2 1"/>
    <material name="plate_mat" rgba="0.12 0.12 0.12 1"/>
  </asset>

  <include file="panda.xml"/>

  <worldbody>
    <camera name="cam" mode="targetbody" pos="{1+ox:.3f} {1+oy:.3f} {1+oz:.3f}" target="panda_hand"/>
    <light directional="true" diffuse=".2 .2 .2" specular="0 0 0" pos="{0+ox:.3f} {1+oy:.3f} {5+oz:.3f}" dir="0 -1 -1" castshadow="false"/>
    <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="{0+ox:.3f} {-1+oy:.3f} {4+oz:.3f}" dir="0 0 -1"/>
    <light directional="true" diffuse="0 0 0" specular=".7 .7 .7" pos="{0+ox:.3f} {3+oy:.3f} {3+oz:.3f}" dir="0 -3 -3"/>

    <geom name="floor" type="plane" pos="0 0 0" group="1" size="0 0 .1" material="plane" condim="6"
          friction="1.0 0.1 0.01"/>

    <body name="spinning_plate" pos="{cx:.3f} {cy:.3f} {plate_z_mid:.3f}">
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.0012 0.0012 0.0012"/>
      <joint name="plate_rotation" type="hinge" axis="0 0 1" limited="false" damping="2.0" armature="0.03"/>

      <geom name="plate_geom" type="cylinder" size="{plate_radius:.3f} {slab_thick:.3f}" material="plate_mat"
            condim="6" friction="1.6 0.5 0.05" solref="0.0025 1" solimp="0.95 0.99 0.001"/>

      <geom name="plate_hub" type="cylinder" size="{hub_rad:.3f} {hub_half:.3f}" pos="0 0 {slab_thick + hub_half:.4f}"
            material="knob_metal" condim="6" friction="1.6 0.5 0.05"/>

      <geom name="knob_stem" type="cylinder"
            size="{handle_rad:.3f} {handle_half:.3f}"
            pos="{knob_r:.3f} 0 {slab_thick + handle_half:.4f}"
            material="knob_metal" contype="1" conaffinity="1"
            condim="6" friction="2.0 0.8 0.08"/>

      <!-- Table grid lines -->
"""
    line_thick = 0.004
    half_line_z = 0.5 * line_thick
    z_rel = grid_z - plate_z_mid
    half_width_lines  = half_cells_x + half_cell
    half_height_lines = half_cells_y + half_cell

    for i in range(H + 1):
        y = (-half_cells_y - half_cell) + i * effective_cell
        xml += f'''      <geom name="grid_h_{i}" type="box"
            size="{half_width_lines:.4f} {0.5*line_thick:.4f} {half_line_z:.4f}"
            pos="0 {y:.4f} {z_rel:.4f}"
            material="grid_line" contype="0" conaffinity="0"/>
'''
    for j in range(W + 1):
        x = (-half_cells_x - half_cell) + j * effective_cell
        xml += f'''      <geom name="grid_v_{j}" type="box"
            size="{0.5*line_thick:.4f} {half_height_lines:.4f} {half_line_z:.4f}"
            pos="{x:.4f} 0 {z_rel:.4f}"
            material="grid_line" contype="0" conaffinity="0"/>
'''
    xml += "    </body>\n"

    # Blocks: spawn ONLY nonzero cells if gv provided; otherwise spawn all.
    for r in range(H):
        for c_ in range(W):
            if gv is not None and int(gv[r][c_]) == 0:
                continue
            x = grid_x0 + c_ * effective_cell
            y = grid_y0 + r * effective_cell
            name = f"G{r+1}{c_+1}"

            rgba_attr = ""
            if gv is not None:
                val = int(gv[r][c_])
                if val != 0:
                    col = palette.get(val)
                    if col is not None:
                        rgba_attr = f'rgba="{col[0]:.3f} {col[1]:.3f} {col[2]:.3f} {col[3]:.3f}"'

            xml += f'''    <body name="{name}_b" pos="{x:.3f} {y:.3f} {block_z_world:.3f}">
      <joint type="free" damping="0.15"/>
      <geom name="{name}" type="box" size="{block_half:.3f} {block_half:.3f} {block_half:.3f}"
            {'material="block_mat"' if not rgba_attr else rgba_attr} density="2000"
            condim="6" friction="1.6 0.6 0.06" solref="0.0025 1" solimp="0.95 0.99 0.001"/>
    </body>
'''
    xml += """  </worldbody>
  <equality/>
</mujoco>
"""
    with open(write_path, "w", encoding="utf-8") as f:
        f.write(xml)
