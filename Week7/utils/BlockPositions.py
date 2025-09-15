import mujoco
import numpy as np

### Utility Functions to Get Block Information ###

def get_geom_position(data, geom_id):
    """ Returns the world position of a geom by its ID. """
    return data.geom_xpos[geom_id].copy()

def get_geom_rotation(data, geom_id):
    """ Returns the world rotation matrix of a geom by its ID. """
    return data.geom_xmat[geom_id].reshape(3, 3).copy()

def get_geom_size(model, geom_id):
    """ Returns the size of a geom by its ID from the model. """
    return model.geom_size[geom_id].copy()

def get_blocks_by_prefix(model, data, prefix):
    """ Reads if names start with 'I' or 'O' in world.xml (input or output).
        Returns dictionaries of block names mapped to their positions and sizes.
    """
    blocks = {}
    sizes = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)  # Gets the name of the geometry
        if name and name.startswith(prefix):        
            position = get_geom_position(data, geom_id)  # Get position of the block
            size = get_geom_size(model, geom_id)          # Get size of the block from model
            blocks[name] = position
            sizes[name] = size  # Save size with the block name
            print(f"Block: {name}, Position: {position}, Size: {size}")  # Debug print for verification
    return blocks, sizes

def get_input_output_blocks(model, data):
    """ Return two dictionaries: 
        - Input Blocks: {Block Name: Position}, {Block Name: Size}
        - Output Blocks: {Block Name: Position}, {Block Name: Size}
    """
    input_blocks, input_sizes = get_blocks_by_prefix(model, data, 'I')
    output_blocks, output_sizes = get_blocks_by_prefix(model, data, 'O')
    return (input_blocks, input_sizes), (output_blocks, output_sizes)
