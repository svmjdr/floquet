#!/usr/bin/env python
# coding: utf-8
"""
Plot Fig.3 from paper.
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


def RWA_frequency(n_bar, data):
    """
    Compute the ac Stark shifted frequency using the time-averaged model.
    """
    epsilon_p = operators.compute_epsilon_p(n_bar, data['params'])
    H_shunted, args, _, _ = operators.build_shunted_hamiltonian(
        data['N_max_a'], data['N_max_b'], epsilon_p, data['params']
    )
    # Compute the time-averaged hamiltonian
    H = (
        H_shunted[0] +
        scipy.special.jv(0, args['oscillating_prefactor']) * H_shunted[1][0]
    )
    # ac Stark shifted frequency can be computed through the difference of
    # eigenenergies
    eigv = H.eigenenergies()
    return (eigv[2] - eigv[0]) / (2 * np.pi)


def main(out_directory, overlaps_threshold):
    """
    Main entry point for this script.

    :param out_directory: Directory to load dumps from. Typically
        ``out/shunted_final_params``.
    :param overlaps_threshold: Absolute threshold for selecting a frequency.
    """
    # Load data
    (
        loaded_data,
        max_N_max_a, max_N_max_b,
        omega_a, omega_p,
        kerr, f_0
    ) = tools.load_data(out_directory)

    # Compute omega_bar_a from any n_bar value
    any_n_bar = next(iter(loaded_data))
    omega_bar_a = operators.compute_shunted_parameters(
        operators.compute_epsilon_p(
            any_n_bar,
            loaded_data[any_n_bar]['params']
        ),
        loaded_data[any_n_bar]['params']
    )[0]

    # Data to plot
    frequencies = []  # This is a list of dicts, see below
    RWA_frequencies = []
    transmon_populations = np.zeros((max_N_max_b, len(loaded_data.keys())))
    mean_transmon_excitations, purities = [], []

    bar = progressbar.ProgressBar(max_value=len(loaded_data.keys()))
    for y, (n_bar, data) in bar(enumerate(loaded_data.items())):
        frequencies.extend(
            tools.find_frequencies_from_overlaps(
                n_bar, data, overlaps_threshold
            )
        )
        RWA_frequencies.append(RWA_frequency(n_bar, data))

        # Write steadystate in real tensor space
        real_ss = floquet.floquet_basis_transform(
            data['f_modes_0'], data['steadystate']
        )
        # Compute transmon populations
        transmon_populations[:data['N_max_b'], y] = np.abs(
            real_ss.ptrace(1).diag()
        )
        mean_transmon_excitations.append(np.sum([
            k * pop
            for k, pop in enumerate(transmon_populations[:data['N_max_b'], y])
        ]))
        # Compute purities
        purities.append(np.sum(data['steadystate'].diag()**2))

    # Plot everything
    with plt.rc_context(rc=custom_mpl.custom_rc()):
        fig = plt.figure(figsize=(15, 15))
        # Define grid
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.05, 1, 1])
        colorbar_axis = fig.add_subplot(gs[0, :])
        transmon_pops_axis = fig.add_subplot(gs[1, 0])
        frequencies_axis = fig.add_subplot(gs[-1, 0],
                                           sharex=transmon_pops_axis)
        # Disable ticks on shared axes
        plt.setp(transmon_pops_axis.get_xticklabels(), visible=False)

        # Plot transmon populations
        colorbar = tools.plot_article_pops(
            loaded_data.keys(), range(max_N_max_b), transmon_populations,
            transmon_pops_axis,
            ylabel="Transmon eigenstates",
            eyeguide=mean_transmon_excitations,
            eyeguide_color=custom_mpl.PALETTE[7],
            threshold=None
        )
        fig.colorbar(colorbar, cax=colorbar_axis, orientation="horizontal")

        # Plot purities on the right axis
        purities_axis = transmon_pops_axis.twinx()
        purities_axis.plot(
            loaded_data.keys(),
            purities,
            color=custom_mpl.PALETTE[0],
            linewidth=3.0
        )
        purities_axis.set_ylim((0, 1))
        purities_axis.set_ylabel(
            r"$\mathrm{Tr}\left[\rho_{\mathrm{ss}}(0)^2\right]$"
        )

        # Plot found frequencies
        frequencies_axis.plot(
            loaded_data.keys(),
            [omega_bar_a / (2 * np.pi) for x in loaded_data.keys()],
            color=custom_mpl.PALETTE[1],
            zorder=1
        )
        frequencies_axis.plot(
            loaded_data.keys(),
            RWA_frequencies,
            color=custom_mpl.PALETTE[3],
            linewidth=3.0,
            zorder=2
        )
        for point in frequencies:
            frequencies_axis.scatter(
                point['n_bar'],
                point['frequency'],
                s=point['area'],
                color=custom_mpl.PALETTE[0],
                zorder=3
            )
        frequencies_axis.set_ylim((np.min(RWA_frequencies) - 5e-3, f_0 + 5e-3))
        frequencies_axis.set_ylabel(r'$\omega_{\mathrm{ac}} / 2 \pi$ (in GHz)')

        # Set X axis label
        frequencies_axis.set_xlabel(
            r'$\left|a_p\right|^2 / '
            r'\left[4 \left|\omega_p - \omega_a\right|^2\right] '
            r'\approx \bar{n}$'
        )

        for ax in [transmon_pops_axis, purities_axis, frequencies_axis]:
            ax.margins(x=0.01)
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=2))
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=2))

        # Save figure
        fig.tight_layout()
        fig.savefig(os.path.join(out_directory, 'fig3.pdf'))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            'Usage: %s OUT_DIRECTORY [OVERLAPS_THRESHOLD]' %
            sys.argv[0]
        )
    out_directory = sys.argv[1]

    overlaps_threshold = 0.1
    if len(sys.argv) > 2:
        overlaps_threshold = float(sys.argv[2])

    main(out_directory, overlaps_threshold)
