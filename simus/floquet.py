# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
from scipy import angle, pi, exp
from qutip.qobj import Qobj
from qutip.mesolve import mesolve
from qutip.solver import Options
from qutip.propagator import propagator
from qutip.utilities import n_thermal

__all__ = ['floquet_modes', 'floquet_modes_table', 'floquet_modes_t_lookup',
           'floquet_state_decomposition',
           'floquet_master_equation_rates',
           'floquet_master_equation_steadystate', 'floquet_basis_transform']


def floquet_modes(H, T, args=None, sort=False, U=None):
    """
    Calculate the initial Floquet modes Phi_alpha(0) for a driven system with
    period T.

    Returns a list of :class:`qutip.qobj` instances representing the Floquet
    modes and a list of corresponding quasienergies, sorted by increasing
    quasienergy in the interval [-pi/T, pi/T]. The optional parameter `sort`
    decides if the output is to be sorted in increasing quasienergies or not.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian, time-dependent with period `T`

    args : dictionary
        dictionary with variables required to evaluate H

    T : float
        The period of the time-dependence of the hamiltonian. The default value
        'None' indicates that the 'tlist' spans a single period of the driving.

    U : :class:`qutip.qobj`
        The propagator for the time-dependent Hamiltonian with period `T`.
        If U is `None` (default), it will be calculated from the Hamiltonian
        `H` using :func:`qutip.propagator.propagator`.

    Returns
    -------

    output : list of kets, list of quasi energies

        Two lists: the Floquet modes as kets and the quasi energies.

    """

    if U is None:
        # get the unitary propagator
        U = propagator(H, T, [], args)

    # find the eigenstates for the propagator
    evals, evecs = U.eigenstates()

    eargs = angle(evals)
    # note: angle is in the interval [-pi, pi], so that the quasi energy is in
    # the interval [-pi/T, pi/T] where T is the period of the driving.
    e_quasi = -eargs / T

    # sort by the quasi energy
    if sort:
        order = np.argsort(-e_quasi)
    else:
        order = list(range(len(evals)))

    return [evecs[o] for o in order], e_quasi[order]


def floquet_modes_table(f_modes_0, f_energies, tlist, H, T, c_op, args=None):
    """
    Pre-calculate the Floquet modes for a range of times spanning the floquet
    period. Can later be used as a table to look up the floquet modes for
    any time.

    Parameters
    ----------

    f_modes_0 : list of :class:`qutip.qobj` (kets)
        Floquet modes at :math:`t`

    f_energies : list
        Floquet energies.

    tlist : array
        The list of times at which to evaluate the floquet modes.

    H : :class:`qutip.qobj`
        system Hamiltonian, time-dependent with period `T`

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        dictionary with variables required to evaluate H

    Returns
    -------

    output : nested list

        A nested list of Floquet modes as kets for each time in `tlist`

    """

    # truncate tlist to the driving period
    tlist_period = tlist[np.where(tlist <= T)]

    f_modes_table_t = [[] for t in tlist_period]

    opt = Options()
    opt.rhs_reuse = True

    for n, f_mode in enumerate(f_modes_0):
        output = mesolve(H, f_mode, tlist_period, [c_op], [], args, opt)
        for t_idx, f_state_t in enumerate(output.states):
            f_modes_table_t[t_idx].append(
                f_state_t * exp(1j * f_energies[n] * tlist_period[t_idx]))

    return f_modes_table_t


def floquet_modes_t_lookup(f_modes_table_t, t, T):
    """
    Lookup the floquet mode at time t in the pre-calculated table of floquet
    modes in the first period of the time-dependence.

    Parameters
    ----------

    f_modes_table_t : nested list of :class:`qutip.qobj` (kets)
        A lookup-table of Floquet modes at times precalculated by
        :func:`qutip.floquet.floquet_modes_table`.

    t : float
        The time for which to evaluate the Floquet modes.

    T : float
        The period of the time-dependence of the hamiltonian.

    Returns
    -------

    output : nested list

        A list of Floquet modes as kets for the time that most closely matching
        the time `t` in the supplied table of Floquet modes.
    """

    # find t_wrap in [0,T] such that t = t_wrap + n * T for integer n
    t_wrap = t - int(t / T) * T

    # find the index in the table that corresponds to t_wrap (= tlist[t_idx])
    t_idx = int(t_wrap / T * len(f_modes_table_t))

    # XXX: might want to give a warning if the cast of t_idx to int discard
    # a significant fraction in t_idx, which would happen if the list of time
    # values isn't perfect matching the driving period
    # if debug: print "t = %f -> t_wrap = %f @ %d of %d" % (t, t_wrap, t_idx,
    # N)

    return f_modes_table_t[t_idx]


def floquet_state_decomposition(f_states, f_energies, psi):
    """
    Decompose the wavefunction `psi` (typically an initial state) in terms of
    the Floquet states, :math:`\psi = \sum_\\alpha c_\\alpha \psi_\\alpha(0)`.

    Parameters
    ----------

    f_states : list of :class:`qutip.qobj` (kets)
        A list of Floquet modes.

    f_energies : array
        The Floquet energies.

    psi : :class:`qutip.qobj`
        The wavefunction to decompose in the Floquet state basis.

    Returns
    -------

    output : array

        The coefficients :math:`c_\\alpha` in the Floquet state decomposition.

    """
    return [f_states[i].overlap(psi)
            for i in np.arange(len(f_energies))]


def floquet_master_equation_rates(f_modes_0, f_energies, c_op, H, T,
                                  args, J_cb, w_th, kmax=5,
                                  f_modes_table_t=None, nT=100):
    """
    Calculate the rates and matrix elements for the Floquet-Markov master
    equation.

    Parameters
    ----------

    f_modes_0 : list of :class:`qutip.qobj` (kets)
        A list of initial Floquet modes.

    f_energies : array
        The Floquet energies.

    c_op : :class:`qutip.qobj`
        The collapse operators describing the dissipation.

    H : :class:`qutip.qobj`
        System Hamiltonian, time-dependent with period `T`.

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        Dictionary with variables required to evaluate H.

    J_cb : callback functions
        A callback function that computes the noise power spectrum, as
        a function of frequency, associated with the collapse operator `c_op`.

    w_th : float
        The temperature in units of frequency.

    k_max : int
        The truncation of the number of sidebands (default 5).

    f_modes_table_t : nested list of :class:`qutip.qobj` (kets)
        A lookup-table of Floquet modes at times precalculated by
        :func:`qutip.floquet.floquet_modes_table` (optional).

    nT : int
        Number of steps to take in the numerical integration.

    Returns
    -------

    output : list

        A list (Delta, X, Gamma, A) containing the matrices Delta, X, Gamma
        and A used in the construction of the Floquet-Markov master equation.

    """

    N = len(f_energies)
    M = 2 * kmax + 1

    omega = (2 * pi) / T

    Delta = np.zeros((N, N, M))
    X = np.zeros((N, N, M), dtype=complex)
    Gamma = np.zeros((N, N, M))
    A = np.zeros((N, N))

    dT = T / nT
    tlist = np.arange(dT, T + dT / 2, dT)

    if f_modes_table_t is None:
        f_modes_table_t = floquet_modes_table(f_modes_0, f_energies,
                                              np.linspace(0, T, nT + 1), H, T, c_op,
                                              args)

    c_op = c_op.full()
    for t in tlist:
        # Use numpy representation to compute overlaps, which is more
        # efficient.
        f_modes_t = [
            f.full() for f in floquet_modes_t_lookup(f_modes_table_t, t, T)
        ]
        for a in range(N):
            bra_a = np.dot(np.conj(f_modes_t[a].T), c_op)
            for b in range(N):
                scalar_product = np.asscalar(np.dot(
                    bra_a,
                    f_modes_t[b]
                ))
                k_idx = 0
                for k in range(-kmax, kmax + 1, 1):
                    X[a, b, k_idx] += (dT / T) * exp(-1j * k * omega * t) * \
                        scalar_product
                    k_idx += 1

    Heaviside = lambda x: ((np.sign(x) + 1) / 2.0)
    for a in range(N):
        for b in range(N):
            k_idx = 0
            for k in range(-kmax, kmax + 1, 1):
                Delta[a, b, k_idx] = f_energies[a] - f_energies[b] + k * omega
                Gamma[a, b, k_idx] = 2 * pi * Heaviside(Delta[a, b, k_idx]) * \
                    J_cb(Delta[a, b, k_idx]) * abs(X[a, b, k_idx]) ** 2
                k_idx += 1

    for a in range(N):
        for b in range(N):
            for k in range(-kmax, kmax + 1, 1):
                k1_idx = k + kmax
                k2_idx = -k + kmax
                A[a, b] += Gamma[a, b, k1_idx] + \
                    n_thermal(abs(Delta[a, b, k1_idx]), w_th) * \
                    (Gamma[a, b, k1_idx] + Gamma[b, a, k2_idx])

    return Delta, X, Gamma, A


def floquet_master_equation_steadystate(A, safety=True):
    """
    Returns the steadystate density matrix (in the floquet basis!) for the
    Floquet-Markov master equation.

    .. note ::

        This function uses the fact that this can be simplified to an
        eigenvalue problem.


    Parameters
    ----------

    A : matrix
        A matrix used to build the master equation.
    safety : bool
        Whether to check that the nullspace is well separated from the other
        eigenspaces or not (defaults to True).
    """
    # Compute auxiliary B matrix
    B = np.zeros_like(A)
    N = A.shape[0]
    for a in range(N):
        for nu in range(N):
            if a != nu:
                B[nu, a] = A[a, nu]
            else:
                B[nu, a] = -1.0 * np.sum([
                    A[a, b] for b in range(N) if b != a]
                )

    # Look for kernel of B
    eigvals, eigvecs = Qobj(B).eigenstates()
    kernel_index = np.argmin(np.abs(eigvals))
    if safety:
        # Ensure that there is a clear separation in the spectrum, which means
        # that the nullspace is well delimited and of size 1.
        assert len(eigvals[np.isclose(eigvals, eigvals[kernel_index])]) == 1

    # Qobj().eigenstates normalizes in L2 norm, we want a L1 normalization
    ss_pops = np.real(eigvecs[kernel_index].full())
    ss_pops = ss_pops / np.sum(ss_pops)
    steadystate = Qobj(np.diag(ss_pops.flatten()))

    return steadystate, np.sort(eigvals[np.isclose(eigvals,
                                                   eigvals[kernel_index])])


def floquet_basis_transform(f_modes, rho0):
    """
    Make a basis transform that takes rho0 from the floquet basis to the
    computational basis.
    """
    return rho0.transform(f_modes, True)
