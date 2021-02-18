#!/usr/bin/env python
# coding: utf-8
"""
Plot Fig.2 from paper.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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
from simus import tools


def main(out_directory, overlaps_threshold):
    """
    Main entry point for this script.

    :param out_directory: Directory to load dumps from. Typically
        ``out/unshunted_final_params``.
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
    frequencies = []  # This is a list of dicts, see below
    transmon_populations = np.zeros((max_N_max_b, len(loaded_data.keys())))
    mean_transmon_excitations, purities = [], []

    bar = progressbar.ProgressBar(max_value=len(loaded_data.keys()))
    for y, (n_bar, data) in bar(enumerate(loaded_data.items())):
        frequencies.extend(
            tools.find_frequencies_from_overlaps(
                n_bar, data, overlaps_threshold
            )
        )

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
        # Compute purity of the steadystate
        purities.append(np.sum(data['steadystate'].diag()**2))

    # Plot everything
    with plt.rc_context(rc=custom_mpl.custom_rc()):
        fig, transmon_pops_axis = plt.subplots(figsize=(15, 7))     

        # Plot transmon populations
        colorbar = tools.plot_article_pops(
            loaded_data.keys(), range(max_N_max_b), transmon_populations,
            transmon_pops_axis,
            ylabel="Transmon eigenstates",
            eyeguide=mean_transmon_excitations,
            eyeguide_color="r"
        )
        transmon_pops_axis.set_xlabel(r'$\bar{n}$')
        fig.colorbar(colorbar)
        # Save figure
        fig.tight_layout()
        fig.savefig(os.path.join(out_directory, 'fig2.pdf'))


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
