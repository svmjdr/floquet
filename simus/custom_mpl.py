"""
Functions to set custom :mod:`matplotlib` parameters.
"""
import shutil

import cycler
import matplotlib as mpl
import numpy as np

from matplotlib.colors import LinearSegmentedColormap


PALETTE = [
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a",
    "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"
]

POPULATIONS_CMAP = LinearSegmentedColormap('my_cm', {
    'red': (
        (0.0, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (1.0, 0.0, 0.0)
    ),
    'green': (
        (0.0, 1.0, 1.0),
        (0.1, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    ),
    'blue': (
        (0.0, 1.0, 1.0),
        (0.1, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    )
})


def custom_rc(rc=None):
    """
    Overload ``matplotlib.rcParams`` to enable advanced features if \
            available. In particular, use LaTeX if available.

    :param rc: An optional dict to overload some :mod:`matplotlib` rc params.
    :returns: A ``matplotlib.rc_context`` object to use in a ``with`` \
            statement.
    """
    custom_rc_ = {}
    # Add LaTeX in rc if available
    if(shutil.which("latex") is not None and
       shutil.which("gs") is not None and
       shutil.which("dvipng") is not None):
        # LateX dependencies are all available
        custom_rc_["text.usetex"] = True
        custom_rc_["text.latex.unicode"] = True
    # Use LaTeX default font family
    # See https://stackoverflow.com/questions/17958485/matplotlib-not-using-latex-font-while-text-usetex-true
    custom_rc_["font.family"] = "sans-serif"
    custom_rc_["font.serif"] = ["cm"] + mpl.rcParams["font.serif"]
    # Scale everything
    custom_rc_.update(_rc_scaling())
    # Set axes style
    custom_rc_.update(_rc_axes_style())
    # Overload if necessary
    if rc is not None:
        custom_rc_.update(rc)
    # Return a context object
    return custom_rc_


def _rc_scaling(font_scaling=3):
    """
    Scale the elements of the figure to get a better rendering.

    Settings borrowed from
    [Seaborn](https://github.com/mwaskom/seaborn/blob/master/seaborn/rcmod.py#L344).

    :returns: a :mod:`matplotlib` ``rcParams``-like dict.
    """
    rc_params = {
        "figure.figsize": np.array([8, 5.5]),
        "figure.dpi": 400,
        # Set misc font sizes
        "font.size": 12 * font_scaling,
        "axes.labelsize": 11 * font_scaling,
        "axes.titlesize": 14 * font_scaling,
        "xtick.labelsize": 10 * font_scaling,
        "ytick.labelsize": 10 * font_scaling,
        "legend.fontsize": 10 * font_scaling,
        # Set misc linewidth
        "grid.linewidth": 1,
        "lines.linewidth": 1.75,
        "patch.linewidth": .3,
        "lines.markersize": 7,
        "lines.markeredgewidth": 1.75,
        # Set ticks padding
        "xtick.major.pad": 7,
        "ytick.major.pad": 7,
    }
    return rc_params


def _rc_axes_style():
    """
    Set the style of the plot and the axes.

    Settings borrowed from
    [Seaborn](https://github.com/mwaskom/seaborn/blob/master/seaborn/rcmod.py#L344).

    :returns: a :mod:`matplotlib` ``rcParams``-like dict.
    """
    # Use dark gray instead of black for better readability on screen
    dark_gray = ".15"
    # Use ColorBrewer-Q10 palette as default one
    cycler_palette = cycler.cycler(
        "color", PALETTE)
    rc_params = {
        # Colors
        "figure.facecolor": "white",
        "text.color": dark_gray,
        "axes.prop_cycle": cycler_palette,
        # Legend
        "legend.frameon": False,  # No frame around legend
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": dark_gray,
        "ytick.color": dark_gray,
        "lines.solid_capstyle": "round",
        # Axes
        "axes.axisbelow": True,
        "axes.linewidth": 1,
        "axes.labelcolor": dark_gray,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.edgecolor": dark_gray,
        # Grid
        "grid.linestyle": "-",
        "grid.color": "EAEAF2",
        # Image
        "image.cmap": "Greys"
    }
    return rc_params
