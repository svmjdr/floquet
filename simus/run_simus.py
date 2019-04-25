# coding: utf-8
"""
Main simulations procedures, to run a single simulation or a batch of
simulations using multiple processes.
"""
from __future__ import absolute_import, print_function, unicode_literals

import logging
import multiprocessing
import os
import pickle
import traceback

import joblib
import numpy as np
import qutip

from simus import floquet
from simus import operators
from simus import tools


def run_simulations(n_bar_list, *args, **kwargs):
    """
    Run a simulation batch.

    :param n_bar_list: List of mean number of photons to run simulations for.

    .. note ::

        For other arguments, see ``run_single_simulation`` function.
    """
    run_matrix = []
    for i, n_bar in enumerate(n_bar_list):
        kwargs['compute_kerr'] = (i == 0)  # Only compute Kerr once.
        run_matrix.append(
            ((n_bar,) + args, dict(kwargs))
        )

    n_jobs = multiprocessing.cpu_count()
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(wrap_run_single_simulation)(*args_kwargs[0],
                                                   **args_kwargs[1])
        for args_kwargs in run_matrix
    )


def wrap_run_single_simulation(*args, **kwargs):
    """
    Wrap the ``run_single_simulation`` function to catch exceptions.
    """
    try:
        return run_single_simulation(*args, **kwargs)
    except Exception:
        logging.error("Unexpected exception: %s", traceback.format_exc())


def run_single_simulation(n_bar, N_max_a, N_max_b, p,
                          compute_kerr=False,
                          out_directory=None, prefix=''):
    """
    Run a single simulation.

    :param n_bar: Mean number of photons at $\omega_p$ in the pump.
    :param N_max_a: Truncature of the cavity Fock space.
    :param N_max_b: Truncature of the transmon eigenstates space.
    :param p: Dictionary of parameters to use.
    :param compute_kerr: Whether Kerr should be computed from hamiltonian or
        not.
    :param out_directory: Directory where to output files.
    :param prefix: A prefix to prepend to output files.
    :return: Dumped data.
    """
    # Use custom logging format, to show truncatures
    LOGGER = logging.getLogger('(%f, %d, %d)' % (n_bar, N_max_a, N_max_b))
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.debug('Parameters are: %s', p)

    # Ensure output directory exists
    if out_directory is None:
        out_directory = os.path.join('out', '%d_%d' % (N_max_a, N_max_b))
    if not os.path.isdir(out_directory):
        try:
            os.makedirs(out_directory)
        except FileExistsError:
            pass

    # Do not recompute if dump file already is exists
    DUMP_PATH = os.path.join(
        out_directory,
        "%sdump_nbar_%f.data" % (prefix, n_bar)
    )
    if os.path.isfile(DUMP_PATH):
        LOGGER.warn('Dump file already exist, SKIPPING computation!')
        with open(DUMP_PATH, 'rb') as fh:
            return pickle.load(fh)

    T = 2 * np.pi / p['omega_p']
    N_max_charge = p.get('N_max_charge', 100)
    H, args, c_ops, computed_H_params = operators.build_hamiltonian(
        N_max_a, N_max_b, n_bar, p,
        LOGGER, out_directory,
        N_max_charge=N_max_charge,
        compute_kerr=compute_kerr
    )

    LOGGER.info('Computing Floquet modes at t=0…')
    # Manually compute the propagator at t=T.
    # This is required to ensure convergence.
    propagator_steps = np.linspace(
        0,
        T,
        p.get('propagator_steps', 10)
    )
    U = qutip.propagator(H, propagator_steps, [], args)[-1]
    # t=0 Floquet modes
    f_modes_0, f_energies = floquet.floquet_modes(H, T, args, sort=True, U=U)

    LOGGER.info('Computing decomposition of initial state on Floquet modes…')
    psi_0 = (H[0] + H[1][0]).groundstate()[1]  # Real ground state
    # Find its decomposition on Floquet modes at t = 0
    f_coeff = floquet.floquet_state_decomposition(f_modes_0, f_energies, psi_0)
    # Plot probabilities on each Floquet mode
    tools.save_probability_distribution_figure(
        f_coeff,
        save_path=os.path.join(
            out_directory,
            '%sinitial_state_floquet_modes_nbar_%f.png' % (prefix, n_bar)
        ),
        xlabel='Floquet mode index',
        ylabel='Probability',
        title='Decomposition of initial state on Floquet modes',
    )
    # Plot the probability distribution of the Floquet mode with the most
    # significant overlap with psi_0
    floquet_overlaps = tools.compute_probabilities(f_coeff)
    most_overlapping_floquet_index = np.argmax(floquet_overlaps)
    tools.save_probability_distribution_figure(
        f_modes_0[most_overlapping_floquet_index],
        save_path=os.path.join(
            out_directory,
            '%smost_overlapping_floquet_mode_initial_nbar_%f.png' % (prefix,
                                                                     n_bar)
        ),
        xlabel='Index in the Fock tensor transmon eigenstates space.',
        ylabel='Probability',
        title=(
            r'Most overlapping Floquet mode (%d) at $t=0$ with $\left|\Psi(0)\right>$' %
            most_overlapping_floquet_index
        )
    )

    LOGGER.info('Computing rate matrices…')
    _, X, _, Amat = floquet.floquet_master_equation_rates(
        f_modes_0, f_energies, c_ops[0], H, T, args,
        J_cb=lambda w: 0.5 * p['gamma'] / T,
        w_th=0, kmax=5, f_modes_table_t=None, nT=100)

    LOGGER.info('Computing steadystate…')
    steadystate, steadystate_eigvals = (
        # Safety is false as we handle the checks manually below
        floquet.floquet_master_equation_steadystate(Amat, safety=False)
    )
    LOGGER.info('Steadystate eigenvalue norm is %g.',
                 np.abs(steadystate_eigvals[0]))
    is_checked = True
    try:
        # Ensure that there is a clear separation in the spectrum, which means
        # that the nullspace is well delimited and of size 1.
        assert (
            len(
                steadystate_eigvals[
                    np.isclose(
                        steadystate_eigvals,
                        np.min(np.abs(steadystate_eigvals))
                    )
                ]
            ) == 1
        )
        # Check convergence of the nullspace computation
        assert np.isclose(np.abs(steadystate_eigvals[0]), 0, atol=1e-12)
        # Ensure the computed steadystate matrix is indeed a density matrix
        # diagonal (positivity and trace 1)
        # Numerical uncertainties make values below 0. Ensure they are simply
        # numerical uncertainties, close to 0.
        assert np.isclose(np.min(steadystate.diag()), 0, atol=1e-7)
        assert all(steadystate.diag() < 1)
        assert np.isclose(steadystate.tr(), 1.0)
    except AssertionError:
        is_checked = False
        LOGGER.warning('Steadystates checks failed.')

    # Plot steadystate in Floquet basis
    tools.save_pops_figure(
        steadystate.diag(),
        save_path=os.path.join(
            out_directory,
            '%ssteadystate_on_floquet_modes_nbar_%f.png' % (prefix, n_bar)
        ),
        xlabel='Floquet mode index',
        ylabel='Probability',
        title='Decomposition of steadystate on Floquet modes'
    )
    # Plot most overlapping Floquet mode in the steadystate
    most_overlapping_floquet_index = np.argmax(steadystate.diag())
    tools.save_probability_distribution_figure(
        f_modes_0[most_overlapping_floquet_index],
        save_path=os.path.join(
            out_directory,
            ('%smost_overlapping_floquet_mode_steadystate_nbar_%f.png' %
             (prefix, n_bar))
        ),
        xlabel='Index in the Fock tensor transmon eigenstates space.',
        ylabel='Probability',
        title=(
            r'Most overlapping Floquet mode (%d) from the steadystate' %
            most_overlapping_floquet_index
        )
    )
    # Plot steadystate in tensor basis
    real_density_matrix = floquet.floquet_basis_transform(
        f_modes_0, steadystate
    )
    tools.save_pops_figure(
        real_density_matrix.diag(),
        save_path=os.path.join(
            out_directory,
            '%ssteadystate_on_tensor_modes_nbar_%f.png' % (prefix, n_bar)
        ),
        xlabel='Index in the Fock tensor transmon eigenstates space.',
        ylabel='Probability',
        title='Decomposition of steadystate on tensor modes'
    )

    LOGGER.info('Dumping data…')
    to_save = {
        "params": p,
        "N_max_a": N_max_a,
        "N_max_b": N_max_b,
        "n_bar": n_bar,
        "f_coeff": f_coeff,
        "psi_0": psi_0,
        "f_modes_0": f_modes_0,
        "f_energies": f_energies,
        "steadystate_pops": steadystate.diag(),
        "steadystate_eigenvalue": steadystate_eigvals,
        "Amat": Amat,
        "X": X,
        "is_checked": is_checked,
        "computed_H_params": computed_H_params
    }
    with open(DUMP_PATH, 'wb') as fh:
        pickle.dump(to_save, fh)

    LOGGER.info('All done!')
    return to_save
