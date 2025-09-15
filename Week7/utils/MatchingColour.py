import mujoco
import numpy as np

def find_colour_match(input_blocks, output_blocks, model, data):
    matching_pairs = []

    for out_name, out_pos in output_blocks.items():
        # Get the color of the current output block from the model (not data)
        out_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, out_name)
        out_color = model.geom_rgba[out_geom_id][:3]  # Corrected: Access from model, not data

        for in_name, in_pos in input_blocks.items():
            # Get the color of the current input block from the model
            in_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, in_name)
            in_color = model.geom_rgba[in_geom_id][:3]  # Corrected: Access from model, not data

            # Check if colors match
            if np.allclose(in_color, out_color):
                print(f"Match found: {in_name} (input) matches {out_name} (output) with color {in_color}")
                matching_pairs.append((in_name, out_name))

    return matching_pairs
