import sys
import cmath
import math
import os
import h5py
import matplotlib.pyplot as plt   # plots
from matplotlib.patches import FancyArrowPatch
import numpy as np
from scipy.interpolate import griddata

from liblibra_core import *
import util.libutil as comn
from libra_py import units
from libra_py import data_conv
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.data_savers as data_savers

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from kcrpmd_utils.kcrpmdtst import KcrpmdTst

def set_style():
    plt.rcParams.update({
        'figure.figsize': (4.0, 3.0),
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'lines.linewidth': 2,
        'axes.linewidth': 3,
        'axes.labelsize': 19,
        'axes.titlesize': 19,
        'xtick.direction': 'in',
        'xtick.top': True,
        'ytick.direction': 'in',
        'ytick.right': True,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.labelsize': 13.5,
        'ytick.labelsize': 13.5,
        'legend.fontsize': 11,
        'legend.frameon': False,
    })

def add_hbar(ax, pi, pf, label, tick_size=0.1, label_offset=0.1, linewidth=1.5, fontsize=10):
    """
    Adds a dimension bar with tick marks and a label.

    Parameters:
        ax           : matplotlib Axes object to draw on
        pi           : starting point of the bar
        pf           : ending point of the bar
        label        : string label to annotate the bar
        tick_size    : length of the perpendicular ticks
        label_offset : distance to offset the label perpendicular to the bar
        linewidth    : width of the lines
        fontsize     : size of the label text
    """
    # Line vector
    v = (pf[0] - pi[0], pf[1] - pi[1])
    length = (v[0]**2 + v[1]**2)**(0.5)
    if length == 0:
        raise ValueError("Dimension bar must have nonzero length.")
    # Normalize direction
    u = (v[0] / length, v[1] / length)
    # Perpendicular direction (to left of the vector)
    uperp = -u[1], u[0]
    # Determine tick start positions based on direction
    pi_ti = (pi[0] - 0.5 * uperp[0] * tick_size, pi[1] - 0.5 * uperp[1] * tick_size)
    pi_tf = (pi[0] + 0.5 * uperp[0] * tick_size, pi[1] + 0.5 * uperp[1] * tick_size)
    pf_ti = (pf[0] - 0.5 * uperp[0] * tick_size, pf[1] - 0.5 * uperp[1] * tick_size)
    pf_tf = (pf[0] + 0.5 * uperp[0] * tick_size, pf[1] + 0.5 * uperp[1] * tick_size)

    # Draw main bar
    ax.plot([pi[0], pf[0]], [pi[1], pf[1]], linewidth=linewidth, color='k')
    # Draw ticks
    ax.plot([pi_ti[0], pi_tf[0]], [pi_ti[1], pi_tf[1]], linewidth=linewidth, color='k')
    ax.plot([pf_ti[0], pf_tf[0]], [pf_ti[1], pf_tf[1]], linewidth=linewidth, color='k')
    # Label position (midpoint + offset)
    pm = ((pi[0] + pf[0]) / 2, (pi[1] + pf[1]) / 2)
    plabel = (pm[0] + uperp[0] * label_offset, pm[1] + uperp[1] * label_offset)
    ax.text(plabel[0], plabel[1], label, fontsize=fontsize, color='k', ha='center', va='center')

    return ax

def add_abar(ax, pi, pf, label, mutation_scale=10, label_offset=0.1, linewidth=1.5, fontsize=10):
    """
    Adds a double-headed arrow bar with a label.

    Parameters:
        ax             : matplotlib Axes object to draw on
        pi, pf         : start and end points of the arrow bar (tuples)
        label          : text label
        label_offset   : perpendicular offset of the label from the bar
        linewidth      : width of the arrow line
        fontsize       : size of label text
        mutation_scale : size of the arrowheads
    """

    # Compute vector and unit direction
    v = (pf[0] - pi[0], pf[1] - pi[1])
    length = (v[0]**2 + v[1]**2)**0.5
    if length == 0:
        raise ValueError("Arrow bar must have nonzero length.")
    u = (v[0] / length, v[1] / length)

    # Perpendicular direction for label offset
    uperp = (-u[1], u[0])

    # Create and add double-headed arrow
    arrow = FancyArrowPatch(
        posA=pi,
        posB=pf,
        shrinkA=0,
        shrinkB=0,
        arrowstyle='<->',
        linewidth=linewidth,
        color='k',
        mutation_scale=mutation_scale
    )
    ax.add_patch(arrow)

    # Label position (midpoint + offset)
    pm = ((pi[0] + pf[0]) / 2, (pi[1] + pf[1]) / 2)
    plabel = (pm[0] + uperp[0] * label_offset, pm[1] + uperp[1] * label_offset)
    ax.text(plabel[0], plabel[1], label, fontsize=fontsize, color='k', ha='center', va='center')

    return ax

