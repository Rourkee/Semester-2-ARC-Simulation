import random
import mujoco

def AssignColours(model, data, input_blocks, output_blocks):
    def random_colour():
        return [random.random(), random.random(), random.random(), 1.0]  # Random RGBA color

    # Generate random colors for all blocks
    input_colours = [random_colour() for _ in range(len(input_blocks))]
    output_colours = [random_colour() for _ in range(len(output_blocks))]

    # Generate the matching color and pick random input/output blocks
    match_colour = random_colour()
    input_match_index = random.randint(0, len(input_blocks) - 1)
    output_match_index = random.randint(0, len(output_blocks) - 1)

    input_colours[input_match_index] = match_colour
    output_colours[output_match_index] = match_colour

    # Get the matching block names
    matching_input_block = list(input_blocks.keys())[input_match_index]
    matching_output_block = list(output_blocks.keys())[output_match_index]

    print(f"Matching Color: {match_colour}")
    print(f"Matched Input Block: {matching_input_block}")
    print(f"Matched Output Block: {matching_output_block}")

    # Assign colors to the geoms
    for i, block in enumerate(input_blocks):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, block)
        model.geom_rgba[geom_id] = input_colours[i]

    for i, block in enumerate(output_blocks):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, block)
        model.geom_rgba[geom_id] = output_colours[i]
