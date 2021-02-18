# coding: utf-8
"""
Compute and plot Floquet quasi-energies.
"""
from __future__ import absolute_import, print_function, unicode_literals

import logging
import matplotlib.pyplot as plt
import multiprocessing
import os
import pickle

import joblib
import numpy as np
import qutip

from simus import floquet
from simus import operators


def plot_f_energies(n_bar_list, N_max_a, N_max_b, params):
    """
    Plot Floquet quasi-energies for a range of pump powers.

    :param n_bar_list: List of mean number of photons at $\omega_p$ in the
        pump.
    :param N_max_a: Truncation of the cavity Fock space.
    :param N_max_b: Truncation of the transmon eigenstates space.
    :param params: Dictionary of parameters to use.
    :return: A list of Floquet quasi-energies, for each pump power.

    .. note ::

        For other arguments, see ``compute_single_f_energies`` function.
    """
    f_energies_table = compute_f_energies(n_bar_list, N_max_a, N_max_b, params)
    fig, ax = plt.subplots()
    for i, f_energies in enumerate(f_energies_table):
        ax.plot(
            [n_bar_list[i] for _ in f_energies],
            f_energies,
            'x'
        )
    ax.set_xlabel(r'$\bar{n}$')
    ax.set_ylabel(r'$\varepsilon_{\alpha}$')

    OUT_DIRECTORY = os.path.join('out', '%d_%d' % (N_max_a, N_max_b))
    fig.savefig(
        os.path.join(OUT_DIRECTORY, 'f_energies.png')
    )


def compute_f_energies(n_bar_list, *args, **kwargs):
    """
    Compute Floquet quasi-energies for a range of pump powers.

    :param n_bar_list: List of mean number of photons at $\omega_p$ in the
        pump.
    :return: A list of Floquet quasi-energies, for each pump power.

    .. note ::

        For other arguments, see ``compute_single_f_energies`` function.
    """
    run_matrix = []
    for n_bar in n_bar_list:
        run_matrix.append(
            ((n_bar,) + args, kwargs)
        )

    return joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
        joblib.delayed(compute_single_f_energies)(*args_kwargs[0],
                                                  **args_kwargs[1])
        for args_kwargs in run_matrix
    )


def compute_single_f_energies(n_bar, N_max_a, N_max_b, p,
                              out_directory=None, prefix=''):
    """
    Compute Floquet quasi-energies for a given pump power.

    :param n_bar: Mean number of photons at $\omega_p$ in the pump.
    :param N_max_a: Truncation of the cavity Fock space.
    :param N_max_b: Truncation of the transmon eigenstates space.
    :param p: Dictionary of parameters to use.
    :param out_directory: Directory where to output files.
    :param prefix: A prefix to prepend to output files.
    :return: Floquet quasi-energies.
    """
    # Use custom logging format, to show truncations
    LOGGER = logging.getLogger('(%f, %d, %d)' % (n_bar, N_max_a, N_max_b))
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.debug('Parameters are: %s', p)

    # Ensure output directory exists
    if out_directory is None:
        out_directory = os.path.join('out', '%d_%d' % (N_max_a, N_max_b))
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)

    # Do not recompute if dump file already is exists
    DUMP_PATH = os.path.join(
        out_directory,
        "%squasi_energies_nbar_%f.data" % (prefix, n_bar)
    )
    if os.path.isfile(DUMP_PATH):
        LOGGER.warn('Dump file already exist, SKIPPING computation!')
        with open(DUMP_PATH, 'rb') as fh:
            return pickle.load(fh)

    LOGGER.info('Building hamiltonian…')
    T = 2 * np.pi / p['omega_p']
    N_max_charge = p.get('N_max_charge', 100)
    H, args, c_ops = operators.build_hamiltonian(N_max_a, N_max_b, n_bar, p,
                                                 N_max_charge=N_max_charge)

    LOGGER.info('Computing Floquet quasienergies…')
    # Manually compute the propagator at t=T.
    # This is required to ensure convergence.
    propagator_steps = np.linspace(
        0,
        T,
        p.get('propagator_steps', 10)
    )
    U = qutip.propagator(H, propagator_steps, [], args)[-1]
    # t=0 Floquet modes
    _, f_energies = floquet.floquet_modes(H, T, args, sort=True, U=U)

    with open(DUMP_PATH, 'wb') as fh:
        pickle.dump(f_energies, fh)

    return f_energies
