#!/usr/bin/env python
# coding: utf-8
"""
Plot Fig.4 from paper.
"""
from __future__ import print_function, unicode_literals

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import progressbar

sys.path.insert(
    0,
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..'
    ))
)
from simus import custom_mpl
from simus import floquet
from simus import operators
from simus import tools


def _bessel(n_bar, data):
    epsilon_p = operators.compute_epsilon_p(n_bar, data['params'])
    H_shunted, args, _, _ = operators.build_shunted_hamiltonian(
        data['N_max_a'], data['N_max_b'], epsilon_p, data['params']
    )
    # Compute the time-averaged hamiltonian
    H = (
        H_shunted[0] +
        scipy.special.jv(0, args['oscillating_prefactor']) * H_shunted[1][0]
    )
    # Kerr and g_2ph can be computed through the eigenstates
    eigval, eigvec = H.eigenstates()

    # Compute Kerr from time averaged model
    RWA_kerr = (
        -1.0 * (eigval[5] - 2 * eigval[2] + eigval[0]) / (2 * np.pi) * 1e6
    )  # In kHz

    # Compute g_2ph from time-averaged model
    ge_state = eigvec[1]  # One excitation in qubit
    fg_state = eigvec[5]  # Two excitations in cavity
    RWA_g2ph = (
        scipy.special.itj0y0(args['oscillating_prefactor'])[0] *
        np.abs(fg_state.overlap(H_shunted[2][0] * ge_state)) /
        np.sqrt(2) /
        (2 * np.pi) /
        1  # TODO: (r * s**2)
    )

    return RWA_kerr, RWA_g2ph


def main(out_directory, n_bar_max_zoom, overlaps_threshold):
    """
    Main entry point for this script.

    :param out_directory: Directory to load dumps from. Typically
        ``out/shunted_final_params``.
    :param n_bar_max_zoom: Maximum value of n_bar in the zoom plots.
    :param overlaps_threshold: Absolute threshold for selecting a frequency.
    """
    # Load data
    (
        loaded_data,
        max_N_max_a, max_N_max_b,
        omega_a, omega_p,
        kerr, f_0
    ) = tools.load_data(out_directory)

    # Data to plot
    kerr_nbar, kerr = [], []
    RWA_kerr, RWA_g2ph = [], []

    bar = progressbar.ProgressBar(max_value=len(loaded_data.keys()))
    for y, (n_bar, data) in bar(enumerate(loaded_data.items())):
        X = data['X']
        f_energies = data['f_energies']

        # Number of computed bands
        n_k_values = int((len(X[0, 0, :]) - 1) / 2)

        max_overlap_idx1 = np.argmax(data['steadystate'].diag())
        max_overlap_idx2, max_overlap_2 = None, None
        k_value_1 = None
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

                    if max_overlap_idx2 is None or X_overlap > max_overlap_2:
                        max_overlap_2 = X_overlap
                        max_overlap_idx2 = j
                        k_value_1 = k_value

        # Compute Kerr
        max_k, max_k_overlap = None, None
        for k in range(len(X[0, 0, :])):
            X2_overlaps = abs(X[max_overlap_idx2, :, k])
            max_overlap_tmp = np.max(X2_overlaps)
            if max_k is None or max_overlap_tmp > max_k_overlap:
                max_k = k
                max_k_overlap = max_overlap_tmp

        max_k_value = -n_k_values + max_k
        X2_overlaps = abs(X[max_overlap_idx2, :, max_k])
        kerr_nbar.append(n_bar)
        kerr.append(1e6 * (
            (-1.0 * (
                f_energies[max_overlap_idx1] - f_energies[max_overlap_idx2] +
                k_value_1 * omega_p
            ) / (2.0 * np.pi)) -
            (-1.0 * (
                f_energies[max_overlap_idx2] -
                f_energies[np.argmax(X2_overlaps)] +
                max_k_value * omega_p
            ) / (2.0 * np.pi))
        ))  # In kHz

        bessel_approx = _bessel(n_bar, data)
        RWA_kerr.append(bessel_approx[0])
        RWA_g2ph.append(bessel_approx[1])

    # Plot everything
    with plt.rc_context(rc=custom_mpl.custom_rc()):
        fig = plt.figure(figsize=(15, 15))
        # Define grid
        gs = gridspec.GridSpec(2, 1)
        kerr_axis = fig.add_subplot(gs[0, 0])
        g_2ph_axis = fig.add_subplot(gs[-1, 0], sharex=kerr_axis)
        # Disable ticks on shared axes
        plt.setp(kerr_axis.get_xticklabels(), visible=False)

        # Plot Kerr values
        kerr_axis.plot(
            loaded_data.keys(),
            RWA_kerr,
            color=custom_mpl.PALETTE[3],
            linewidth=3.0,
            zorder=1
        )
        kerr_axis.scatter(
            kerr_nbar, kerr,
            s=(np.pi * 3.0**2), color=custom_mpl.PALETTE[0],
            zorder=2
        )
        kerr_axis.set_ylabel('Kerr (in kHz)')

        # Plot g2ph
        g_2ph_axis.plot(
            loaded_data.keys(),
            RWA_g2ph,
            color=custom_mpl.PALETTE[3],
            zorder=1
        )
        g_2ph_axis.scatter(
            [150, 600, 3000], [1.3, 1.2, 0.39],
            s=(np.pi * 3.0**2),
            color=custom_mpl.PALETTE[0],
            zorder=2
        )
        g_2ph_axis.set_ylabel(
            r'$g_{\mathrm{2ph}} / \left[\varphi_a^0 \left(\varphi_b^0\right)^2\right]$'
        )

        # Set X axis label
        g_2ph_axis.set_xlabel(
            r'$\left|a_p\right|^2 / '
            r'\left[4 \left|\omega_p - \omega_a\right|^2\right] '
            r'\approx \bar{n}$'
        )

        for ax in [kerr_axis, g_2ph_axis]:
            ax.margins(x=0.01)
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=2))
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=2))

        # Save figure
        fig.tight_layout()
        fig.savefig(os.path.join(out_directory, 'fig4.pdf'))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            'Usage: %s OUT_DIRECTORY [N_BAR_MAX_ZOOM] [OVERLAPS_THRESHOLD]' %
            sys.argv[0]
        )
    out_directory = sys.argv[1]

    n_bar_max_zoom = 200
    if len(sys.argv) > 2:
        n_bar_max_zoom = float(sys.argv[2])

    overlaps_threshold = 0.1
    if len(sys.argv) > 3:
        overlaps_threshold = float(sys.argv[3])

    main(out_directory, n_bar_max_zoom, overlaps_threshold)
