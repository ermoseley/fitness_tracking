#!/usr/bin/env python3

"""
Generate a simple cartoon-style logo for BodyMetrics.
Design: person flexing with a tape around waist, flat colors.
Outputs a 1024x1024 transparent PNG at assets/icon_1024.png
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle, FancyBboxPatch


def draw_logo(ax):
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Colors
    bg = "#0fb9b1"       # teal background circle
    skin = "#f5c6a5"     # skin tone
    torso = "#2d3436"    # dark gray torso
    tape = "#f6c945"     # measuring tape yellow
    tape_mark = "#2d3436" # tick marks

    # Background circle
    ax.add_patch(Circle((0.5, 0.5), 0.48, color=bg))

    # Torso (rounded box)
    ax.add_patch(FancyBboxPatch((0.32, 0.28), 0.36, 0.28, boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=0, facecolor=torso))

    # Head
    ax.add_patch(Circle((0.5, 0.70), 0.09, color=skin, linewidth=0))

    # Arms (flexed) - use wedges as simple biceps shapes
    # Left arm
    ax.add_patch(Wedge((0.34, 0.64), 0.17, 110, 200, width=0.10, facecolor=skin, linewidth=0))
    # Right arm
    ax.add_patch(Wedge((0.66, 0.64), 0.17, -20, 70, width=0.10, facecolor=skin, linewidth=0))

    # Measuring tape across waist
    tape_y = 0.40
    ax.add_patch(Rectangle((0.26, tape_y-0.03), 0.48, 0.06, facecolor=tape, edgecolor='none'))
    # Tape tick marks
    for i in range(0, 25):
        x = 0.28 + i * (0.44/24.0)
        h = 0.018 if i % 5 else 0.03
        ax.add_patch(Rectangle((x, tape_y-h/2), 0.003, h, facecolor=tape_mark, edgecolor='none'))

    # Simple waist tie end
    ax.add_patch(Rectangle((0.72, tape_y-0.02), 0.06, 0.04, facecolor=tape, edgecolor='none'))


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, 'icon_1024.png')
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_logo(ax)
    fig.savefig(out_png, dpi=100, transparent=True)
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()


