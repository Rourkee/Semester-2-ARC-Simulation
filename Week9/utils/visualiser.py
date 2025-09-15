import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def show_block_views(input_blocks, output_blocks, available_blocks=None):
    fig = plt.figure(figsize=(9, 4.2))

    # Define 2-row, 3-column grid layout
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[0.13, 0.87],     # Title row smaller
        width_ratios=[1.4, 0.9, 0.9]    # Narrower 2D grid plots
    )

    if available_blocks:
        # Left: 3D Available Blocks
        ax3d = fig.add_subplot(gs[1, 0], projection='3d')
        ax3d.set_title("Available Blocks (3D)", pad=4)
        for name, pos in available_blocks.items():
            ax3d.scatter(pos[0], pos[1], pos[2], s=40)
            ax3d.text(pos[0], pos[1], pos[2], name, fontsize=7, ha='center', va='center')
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.view_init(elev=30, azim=45)

        pos = ax3d.get_position()
        new_pos = [pos.x0 - 10, pos.y0, pos.width, pos.height]    #small shift left
        ax3d.set_position(new_pos)

        # Centered title above 2D grids
        title_ax = fig.add_subplot(gs[0, 1:])
        title_ax.axis('off')
        title_ax.set_title("2D Input and Output", fontsize=12, fontweight='bold', pad=1)

        # Middle: 2D Input Grid
        ax_input = fig.add_subplot(gs[1, 1])
        ax_input.set_title("Before (Input)", pad=2)
        for name, pos in input_blocks.items():
            ax_input.plot(pos[0], pos[1], 's', markersize=16)
            ax_input.text(pos[0], pos[1], name, fontsize=7, ha='center', va='center')
        ax_input.set_xlabel("X")
        ax_input.set_ylabel("Y")
        ax_input.set_aspect('equal')

        # Right: 2D Output Grid
        ax_output = fig.add_subplot(gs[1, 2])
        ax_output.set_title("After (Output)", pad=2)
        for name, pos in output_blocks.items():
            ax_output.plot(pos[0], pos[1], 's', markersize=16)
            ax_output.text(pos[0], pos[1], name, fontsize=7, ha='center', va='center')
        ax_output.set_xlabel("X")
        ax_output.set_ylabel("")              # ← Remove Y-axis label
        ax_output.set_yticklabels([])         # ← Remove Y tick labels
        ax_output.set_aspect('equal')


    else:
        # Fallback layout if no 3D blocks are passed
        fig.suptitle("2D Input and Output Data", fontsize=12, fontweight='bold')
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("Before (Input)", pad=2)
        for name, pos in input_blocks.items():
            ax1.plot(pos[0], pos[1], 's', markersize=16)
            ax1.text(pos[0], pos[1], name, fontsize=7, ha='center', va='center')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_aspect('equal')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("After (Output)", pad=2)
        for name, pos in output_blocks.items():
            ax2.plot(pos[0], pos[1], 's', markersize=16)
            ax2.text(pos[0], pos[1], name, fontsize=7, ha='center', va='center')
        ax2.set_xlabel("X")
        ax2.set_aspect('equal')

    # Adjust layout: pull title closer to grids
    plt.subplots_adjust(top=0.14)
    plt.tight_layout()
    plt.show(block=False)

    # Bring the window to the front
    manager = plt.get_current_fig_manager()
    try:
        manager.window.attributes('-topmost', 1)
        manager.window.attributes('-topmost', 0)
    except Exception:
        try:
            manager.window.raise_()
        except:
            pass
