# coding: utf-8
from __future__ import unicode_literals
import collections
import json
import os
import pickle

try:
    import matplotlib
    import matplotlib.pyplot as plt
except RuntimeError:
    # Ignore errors when running matplotlib in a venv and not calling with
    # custom backend before
    pass
import numpy as np
import qutip

from simus import custom_mpl


def save_bar_figure(data, save_path, xlabel, ylabel, title):
    """
    Plot a bar histogram figure, adapted from QuTip code.

    :param data: Data to plot.
    :param save_path: Path to the resulting dumped image file. If ``None``,
        figure is shown instead.
    :param xlabel: Label on the X axis.
    :param ylabel: Label on the Y axis.
    :param title: Title of the figure.
    """
    fig, ax = plt.subplots()
    ax.bar(
        np.arange(0, len(data)) - .4,
        [np.asscalar(x) for x in data],
        color="green", alpha=0.6, width=0.8
    )
    ax.set_xlim(-1, len(data))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)  # Avoid keeping two many figures open
    else:
        fig.show()


def save_pops_figure(data, save_path=None, xlabel='Basis state',
                     ylabel='Population', title='Populations histogram'):
    """
    Plot and save a populations distribution figure (for density matrices).

    :param data: Diagonal of the density matrix used to plot populations.
    :param save_path: Path to the resulting dumped image file. If ``None``,
        figure is shown instead.
    :param xlabel: Label on the X axis.
    :param ylabel: Label on the Y axis.
    :param title: Title of the figure.
    """
    save_bar_figure(data, save_path, xlabel, ylabel, title)


def save_probability_distribution_figure(
        data, save_path=None, xlabel='Basis state',
        ylabel='Probability', title='Probability distribution'
):
    """
    Plot and save a probability distribution figure (for quantum state).

    :param data: Quantum state used to plot probability distribution.
    :param save_path: Path to the resulting dumped image file. If ``None``,
        figure is shown instead.
    :param xlabel: Label on the X axis.
    :param ylabel: Label on the Y axis.
    :param title: Title of the figure.
    """
    save_bar_figure(
        compute_probabilities(data), save_path, xlabel, ylabel, title
    )


def load_params(params_file):
    """
    Load a JSON file of parameters into a parameter dict.

    .. note ::

        Also handle the correct multiplication by $2\pi$ etc.

    :param params_file: File to the JSON file to load params from.
    :returns: A parameters dict.
    """
    p = {}
    with open(params_file, 'r') as fh:
        params = json.load(fh)
        for k, v in params.items():
            if v['unit'] == 'GHz':
                # Parameters are in 10^9 rad/s
                p[k] = v['value'] * (2.0 * np.pi)
            else:
                p[k] = v['value']
    return p


def compute_probabilities(state):
    """
    Compute probability distribution from a given state.

    :param state: A quantum state vector.
    :returns: A vector of probability distribution.

    >>> compute_probabilities([1, 2, 3 + 1.0j])
    array([  1.,   4.,  10.])

    >>> compute_probabilities(qutip.Qobj(np.array([1, 2, 3 + 1.0j])))
    array([  1.,   4.,  10.])
    """
    if isinstance(state, qutip.Qobj):
        # Convert back to numpy array
        state = state.full().flatten()
    return np.real(np.power(np.abs(state), 2))


def tensor_pops_pad(data, orig_N_max_a, orig_N_max_b, new_N_max_a, new_N_max_b):
    """
    Pad populations from a density matrix with zeros, as if one of the
    dimensions was larger than actually.

    :param data: A diagonal density matrix or a vector of populations. Can be a
        ``qutip.Qobj`` object.
    :param orig_N_max_a: Original dimension on first space.
    :param orig_N_max_b: Original dimension on second space.
    :param new_N_max_a: New dimension on first space.
    :param new_N_max_b: New dimension on second space.
    :return: The padded populations vector (``numpy`` 1D array).

    .. note ::

        ``new_N_max_a`` and ``new_N_max_b`` should be larger or equal to the
        initial dimensions.

    >>> # Let's assume initially N_max_a = 2, N_max_b = 3
    >>> tensor_pops_pad([1, 1, 1, 1, 1, 1], 2, 3, 3, 3)  # Expand a
    array([1, 1, 1, 1, 1, 1, 0, 0, 0])

    >>> tensor_pops_pad([1, 1, 1, 1, 1, 1], 2, 3, 2, 4)  # Expand b
    array([1, 1, 1, 0, 1, 1, 1, 0])

    >>> tensor_pops_pad(np.array([1, 1, 1, 1, 1, 1]), 2, 3, 3, 4)
    array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])

    >>> tensor_pops_pad(np.diag([1, 1, 1, 1, 1, 1]), 2, 3, 3, 4)
    array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])

    >>> tensor_pops_pad(qutip.Qobj(np.diag([1, 1, 1, 1, 1, 1])), 2, 3, 3, 4)
    array([ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.])
    """
    # Ensure to get a 1D diagonal array
    if isinstance(data, qutip.Qobj):
        data = data.diag()
    else:
        data = np.array(data)
        if len(data.shape) > 1:
            data = np.diag(data)

    # Reshape it as a N_max_a * N_max_b array for easy padding
    pops_matrix = data.reshape(orig_N_max_a, orig_N_max_b)
    # Pad it on each axis (in each subspace) as expected
    padded_pops_matrix = np.lib.pad(
        pops_matrix,
        (
            (0, new_N_max_a - orig_N_max_a),
            (0, new_N_max_b - orig_N_max_b)
        ),
        'constant', constant_values=(0)
    )
    # Reshape it in a single 1D array before returning
    return padded_pops_matrix.reshape(new_N_max_a * new_N_max_b)


def load_data(dir):
    """
    Load all available dumps from the given directory.

    :param dir: A directory to load data from.
    :returns: A dictionary mapping n_bar values to the loaded data.
    """
    loaded_data = {}
    max_N_max_a, max_N_max_b = 0, 0
    omega_a, omega_p, kerr, f_0 = None, None, None, None
    for f in sorted(os.listdir(dir)):
        if not os.path.isdir(os.path.join(dir, f)):
            continue

        for g in sorted(os.listdir(os.path.join(dir, f))):
            if not g.endswith('.data'):
                continue
            with open(os.path.join(dir, f, g), 'rb') as fh:
                data = pickle.load(fh)

            # Ensure steadystate density matrix is of trace 1
            data['steadystate_pops'] = (
                data['steadystate_pops'] / sum(data['steadystate_pops'])
            )
            data['steadystate'] = qutip.Qobj(
                np.diag(data['steadystate_pops']),
                dims=[
                    [data['N_max_a'], data['N_max_b']],
                    [data['N_max_a'], data['N_max_b']]
                ]
            )

            n_bar = data['n_bar']
            N_max_a = data['N_max_a']
            N_max_b = data['N_max_b']
            if n_bar in loaded_data:
                # If some file was already loaded for this n_bar value, keep
                # the largest truncations
                previous_data = loaded_data[n_bar]
                if (
                    N_max_a < previous_data['N_max_a'] and
                    N_max_b < previous_data['N_max_b']
                ):
                    continue

            loaded_data[n_bar] = data
            omega_a = data['params']['omega_a']
            omega_p = data['params']['omega_p']
            if 'computed_H_params' in data and data['computed_H_params']:
                kerr, f_0 = data['computed_H_params']
            if N_max_a > max_N_max_a:
                max_N_max_a = N_max_a
            if N_max_b > max_N_max_b:
                max_N_max_b = N_max_b
    loaded_data = collections.OrderedDict(sorted(loaded_data.items(),
                                                 key=lambda t: t[0]))
    return (
        loaded_data,
        max_N_max_a, max_N_max_b,
        omega_a, omega_p,
        kerr, f_0
    )


def plot_article_pops(x, y, z, ax,
                      ylabel=None, eyeguide=None, eyeguide_color=None,
                      threshold=1e-2):
    """
    Plot populations (article figures style).

    :param x: The X axis data.
    :param y: The Y axis data.
    :param z: The color plot data (populations within [0, 1]).
    :param ax: An axis to plot on.
    :param ylabel: An optional label for the Y axis.
    :param eyeguide: Values to plot on top of the color plot to guide the eye.
    :param eyeguide_color: Color for the eyeguide. Optional.
    :param threshold:
    """
    # Ensure values are within expected range
    z = np.clip(z, 0, 1.0)
    # Mask values too small to be relevant
    max_y_index = None
    if threshold:
        z = np.ma.masked_less(z, threshold)
        max_y_index = (
            np.argmax([
                y for y, line in enumerate(z.mask) if not np.all(line)
            ]) + 2
        )

    cbar = ax.pcolor(
        x, y, z,
        cmap=custom_mpl.POPULATIONS_CMAP,
        vmin=0,
        vmax=1.0,
        shading='nearest'
    )
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=2))
    if eyeguide:
        ax.plot(
            x, eyeguide, color=eyeguide_color, linewidth=3.0
        )
    if max_y_index:
        max_y_index = np.ceil(max_y_index / 10) * 10
        ax.set_ylim(ymax=max_y_index)
    if ylabel:
        ax.set_ylabel(ylabel)

    return cbar


def find_frequencies_from_overlaps(n_bar, data, overlaps_threshold):
    """
    TODO
    """
    frequencies = []

    # Overlaps tensor
    X = data['X']
    # Floquet quasi-energies
    f_energies = data['f_energies']
    # Number of computed bands
    n_k_values = int((len(X[0, 0, :]) - 1) / 2)
    # Pump frequency
    omega_p = data['params']['omega_p']

    for i, f_mode_overlap in enumerate(data['steadystate'].diag()):
        # First, look at Floquet modes in the steadystate with enough
        # weight
        if f_mode_overlap < overlaps_threshold:
            continue

        # Then look at coupled elements through X tensor, in neighbor bands
        # We only consider the k = -2, -1, 0, 1, 2 bands
        k_range = list(range(n_k_values - 2, n_k_values + 3))
        for k in k_range:
            k_value = -n_k_values + k  # This is the real value, > 0 or < 0

            X_overlaps = abs(X[i, :, k])
            for j, X_overlap in enumerate(X_overlaps):
                # Look for a coupled enough Floquet mode in this band
                if X_overlap < overlaps_threshold:
                    continue

                # Compute frequency
                frequencies.append({
                    "n_bar": n_bar,
                    "frequency": -1.0 * (
                        f_energies[i] - f_energies[j] +
                        k_value * omega_p
                    ) / (2.0 * np.pi),
                    "area": np.pi * 7.0**2 * (
                        np.sqrt(f_mode_overlap) * X_overlap
                    )**2
                })
    return frequencies
