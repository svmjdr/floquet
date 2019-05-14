# coding: utf-8
# Do not import unicode_literals or hamiltonian spec breaks in Python 2.
import collections
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import qutip


def compute_epsilon_p(n_bar, p):
    """
    Compute the pump amplitude as a function of a rough approximation of
    ``n_bar``.
    """
    return np.sqrt(n_bar) * np.sqrt(8) * (p['omega_p'] - p['omega_a'])


def change_basis(operator, change_basis_matrix):
    """
    Write a given operator in another basis.

    :param operator: The operator to write in another basis.
    :param change_basis_matrix: The matrix to pass from the old basis to the
        new basis (each column contains the coordinates of a new basis vector
        in the old basis).
    :return: The operator in the new basis.

    >>> # Write sigmax operator in its eigenbasis
    >>> change_basis( \
            qutip.sigmax(), \
            qutip.Qobj(np.column_stack( \
                x.full() for x in qutip.sigmax().eigenstates()[1]) \
            ) \
        ).full()
    array([[-1.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j]])
    """
    return change_basis_matrix.dag() * operator * change_basis_matrix


def compute_eigvals_kerr_f_0(H, H_cavity, p, out_directory):
    """
    Compute eigenenergies of the time-independent part of the hamiltonian. Plot
    them. Also compute the Kerr value from these eigenenergies.

    :param H: The time-independent part of the hamiltonian.
    :param H_cavity: Cavity part of the time-independent hamiltonian, to
        compute mean cavity occupation.
    :param p: A dict of parameters.
    :param out_directory: Output directory in which images should be stored.
    :return: The kerr value and frequency for nbar = 0, both in GHz.
    """
    eigvals, eigvecs = H.eigenstates()

    x, y = [], []  # Scatter plot data
    # Group eigvals by cavity occupation
    eigvals_by_n_a = collections.defaultdict(list)
    for eigval, eigvec in zip(eigvals, eigvecs):
        # Compute cavity occupation
        cavity_occupation = np.real(
            qutip.expect(H_cavity / p['omega_a'], eigvec)
        )
        # Append to scatter plot
        x.append(cavity_occupation)
        y.append(eigval)
        # Take the round to group by integer cavity occupation
        eigvals_by_n_a[round(cavity_occupation)].append(eigval)

    # Dump eigvals plot
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_ylabel(r'Energies (in $10^9$ rad/s)')
    ax.set_xlabel('Cavity mean occupation')
    ax.set_title('Time-independent hamiltonian eigenvalues')
    fig.savefig(os.path.join(out_directory, 'hamiltonian_eigvals.png'))

    # Look for minimum for each cavity occupation number
    min_eigval_by_n_a = {
        k: min(v) for k, v in eigvals_by_n_a.items()
    }
    sorted_keys = sorted(min_eigval_by_n_a.keys())
    # Compute kerr
    kerr = np.real(
        2 * min_eigval_by_n_a[sorted_keys[1]] -
        min_eigval_by_n_a[sorted_keys[2]] -
        min_eigval_by_n_a[sorted_keys[0]]
    ) / (2 * np.pi)
    f_0 = (min_eigval_by_n_a[sorted_keys[1]] -
           min_eigval_by_n_a[sorted_keys[0]]) / (2 * np.pi)
    return kerr, f_0


def build_unshunted_hamiltonian(N_max_a, N_max_b, N_max_charge, epsilon_p, p):
    """
    Build operators for simulation of a transmon in a cavity in the unshunted
    case.

    :param N_max_a: Truncation on the cavity Fock space.
    :param N_max_b: Truncation on the transmon eigenstates space.
    :param N_max_charge: To compute the transmon eigenvectors, we use the
        representation in the charge states basis. This is the truncation in
        this space and should be really large (indices range from -N_max_charge
        to +N_max_charge).
    :param epsilon_p: Pump amplitude.
    :param p: Dict of parameters values.
    :returns: A tuple of H, args, H_0_cavity and c_ops.
    """
    # Preconditions
    assert N_max_b < (2 * N_max_charge + 1)

    # Operators on the cavity
    a = qutip.destroy(N_max_a)
    X_a = (a + a.dag()) / np.sqrt(2)
    Id_a = qutip.qeye(N_max_a)

    # Operators on the transmon, in charge state representation
    N_ch = qutip.Qobj(np.diag(range(-N_max_charge, N_max_charge + 1)))
    cos_phi = qutip.Qobj(
        np.diag(0.5 * np.ones(2 * N_max_charge), k=1) +
        np.diag(0.5 * np.ones(2 * N_max_charge), k=-1)
    )
    sin_phi = qutip.Qobj(
        np.diag(-0.5j * np.ones(2 * N_max_charge), k=1) +
        np.diag(0.5j * np.ones(2 * N_max_charge), k=-1)
    )
    Id_ch = qutip.qeye(2 * N_max_charge + 1)

    # Transmon hamiltonian to find transmon eigenstates
    H_tr = qutip.Qobj(
        4.0 * p['epsilon_c'] * (N_ch - p['N_g'] * Id_ch)**2 +
        -1.0 * p['epsilon_j'] * cos_phi
    )
    _, eigenvec_tr = H_tr.eigenstates()
    # Matrix to pass from charge states basis to transmon basis. Each column is
    # a transmon eigenstate expressed in the charge states basis.
    change_basis_matrix = qutip.Qobj(
        np.column_stack(x.full() for x in eigenvec_tr)
    )
    # Then, rewrite transmon operators in this basis…
    N_ch_eigv_basis = change_basis(N_ch, change_basis_matrix)
    cos_phi_eigv_basis = change_basis(cos_phi, change_basis_matrix)
    sin_phi_eigv_basis = change_basis(sin_phi, change_basis_matrix)
    # and truncate them to keep only N_max_b components.
    N_ch_eigv_basis = qutip.Qobj(N_ch_eigv_basis[:N_max_b, :N_max_b])
    cos_phi_eigv_basis = qutip.Qobj(cos_phi_eigv_basis[:N_max_b, :N_max_b])
    sin_phi_eigv_basis = qutip.Qobj(sin_phi_eigv_basis[:N_max_b, :N_max_b])
    Id_ch_eigv_basis = qutip.qeye(N_max_b)

    # Cavity + transmon tensor operators in Fock tensor eigenstates of transmon
    # space
    N_a_tensor = qutip.tensor(a.dag() * a, Id_ch_eigv_basis)
    X_a_tensor = qutip.tensor(X_a, Id_ch_eigv_basis)
    N_ch_tensor = qutip.tensor(Id_a, N_ch_eigv_basis)
    cos_phi_tensor = qutip.tensor(Id_a, cos_phi_eigv_basis)
    sin_phi_tensor = qutip.tensor(Id_a, sin_phi_eigv_basis)
    Id_tensor = qutip.tensor(Id_a, Id_ch_eigv_basis)

    # Time-independent part of hamiltonian
    H_0_cavity = p['omega_a'] * N_a_tensor
    H_0_coupling = (
        p['epsilon_g'] * (N_ch_tensor - p['N_g'] * Id_tensor) * X_a_tensor
    )
    H_0_qubit = 4.0 * p['epsilon_c'] * (N_ch_tensor - p['N_g'] * Id_tensor)**2
    H_0 = (
        H_0_cavity +
        H_0_coupling +
        H_0_qubit
    )

    # Time-dependent part
    oscillating_prefactor = (
        -1.0 * p['epsilon_g'] * epsilon_p *
        p['omega_a'] / p['omega_p'] / (p['omega_p']**2 - p['omega_a']**2)
    )
    H_1 = -1.0 * p['epsilon_j'] * cos_phi_tensor
    H_1_coef = 'cos(oscillating_prefactor * sin(omega_p * t))'
    H_2 = p['epsilon_j'] * sin_phi_tensor
    H_2_coef = 'sin(oscillating_prefactor * sin(omega_p * t))'
    # Complete QuTiP hamiltonian object
    args = {
        'oscillating_prefactor': oscillating_prefactor,
        'omega_p': p['omega_p']
    }
    H = [H_0, [H_1, H_1_coef], [H_2, H_2_coef]]

    # Dissipation operators
    c_ops = [X_a_tensor]

    return H, args, H_0_cavity, c_ops


def compute_shunted_parameters(epsilon_p, p):
    """
    TODO
    """
    theta = -0.5 * np.arctan(
        2 * p['epsilon_g'] * np.sqrt(p['epsilon_l'] * p['omega_a']) /
        (p['omega_a']**2 - 8 * p['epsilon_c'] * p['epsilon_l'])
    )

    omega_tilde_a = (
        p['omega_a'] * np.cos(theta)**2 +
        (
            8 * p['epsilon_c'] * p['epsilon_l'] / p['omega_a'] *
            np.sin(theta)**2
        ) +
        (
            -1.0 * p['epsilon_g'] *
            np.sqrt(p['epsilon_l'] / p['omega_a']) * np.sin(2 * theta)
        )
    )
    omega_tilde_q = (
        p['omega_a'] * np.sin(theta)**2 +
        (
            8 * p['epsilon_c'] * p['epsilon_l'] / p['omega_a'] *
            np.cos(theta)**2
        ) +
        (
            p['epsilon_g'] * np.sqrt(p['epsilon_l'] / p['omega_a']) *
            np.sin(2 * theta)
        )
    )

    r = (
        np.cos(theta) * np.sqrt(p['omega_a'] / p['epsilon_l']) *
        np.power(omega_tilde_q / p['omega_a'], 0.25)
    )
    s = (
        np.sin(theta) * np.sqrt(p['omega_a'] / p['epsilon_l']) *
        np.power(omega_tilde_a / p['omega_a'], 0.25)
    )

    omega_bar_a = np.sqrt(omega_tilde_a * p['omega_a'])
    omega_bar_q = np.sqrt(omega_tilde_q * p['omega_a'])

    oscillating_prefactor = (
        0.5 * np.sqrt(p['omega_a'] / p['epsilon_l']) *
        epsilon_p * p['omega_p'] * np.sin(2 * theta) * (
            1.0 / (p['omega_p']**2 - p['omega_a'] * omega_tilde_q) -
            1.0 / (p['omega_p']**2 - p['omega_a'] * omega_tilde_a)
        )
    )

    c_op_r = np.cos(theta) * np.power(p['omega_a'] / omega_tilde_a, 0.25)
    c_op_s = np.sin(theta) * np.power(p['omega_a'] / omega_tilde_q, 0.25)

    return (
        omega_bar_a, omega_bar_q,
        r, s,
        oscillating_prefactor,
        c_op_r, c_op_s
    )


def build_shunted_hamiltonian(N_max_a, N_max_b, epsilon_p, p):
    """
    Build operators for simulation of a transmon in a cavity in the unshunted
    case.

    :param N_max_a: Truncation on the cavity Fock space.
    :param N_max_b: Truncation on the transmon eigenstates space.
    :param epsilon_p: Pump amplitude.
    :param p: Dict of parameters values.
    :returns: A tuple of H, args, H_0_cavity and c_ops.
    """
    (
        omega_bar_a, omega_bar_q,
        r, s,
        oscillating_prefactor,
        c_op_r, c_op_s
    ) = compute_shunted_parameters(epsilon_p, p)

    # Operators on the cavity
    a = qutip.destroy(N_max_a)
    X_a = (a + a.dag()) / np.sqrt(2)
    P_a = (a - a.dag()) / 1.0j / np.sqrt(2)
    Id_a = qutip.qeye(N_max_a)
    # Operators on the transmon
    b = qutip.destroy(N_max_b)
    X_b = (b + b.dag()) / np.sqrt(2)
    P_b = (b - b.dag()) / 1.0j / np.sqrt(2)
    Id_b = qutip.qeye(N_max_b)
    # Compute tensor operators
    Id_tensor = qutip.tensor(Id_a, Id_b)
    X_a_tensor = qutip.tensor(X_a, Id_b)
    P_a_tensor = qutip.tensor(P_a, Id_b)
    X_b_tensor = qutip.tensor(Id_a, X_b)
    P_b_tensor = qutip.tensor(Id_a, P_b)

    # Time independent part of hamiltonian in Fock basis
    H_0_cavity = 0.5 * omega_bar_a * (
        X_a_tensor**2 + P_a_tensor**2
    )
    H_0_qubit = 0.5 * omega_bar_q * (
        X_b_tensor**2 + P_b_tensor**2
    )
    H_0 = H_0_cavity + H_0_qubit

    # Time dependent part in Fock basis
    phi_X_tensor = (r * X_b_tensor - s * X_a_tensor)
    exp_j_phi_X_tensor = (1.0j * phi_X_tensor).expm()
    H_1 = -0.5 * p['epsilon_j'] * (
        exp_j_phi_X_tensor + exp_j_phi_X_tensor.dag()
    )
    H_1_coef = 'cos(oscillating_prefactor * sin(omega_p * t))'
    H_2 = -0.5j * p['epsilon_j'] * (
        exp_j_phi_X_tensor - exp_j_phi_X_tensor.dag()
    )
    H_2_coef = 'sin(oscillating_prefactor * sin(omega_p * t))'

    # Rewrite everything in transmon eigenbasis
    _, eigenvec = (H_0 + H_1).eigenstates()
    H_0_cavity_eigv_basis = (
        H_0_cavity.transform(eigenvec) -
        0.5 * omega_bar_a * Id_tensor
    )
    H_0_eigv_basis = H_0.transform(eigenvec)
    H_1_eigv_basis = H_1.transform(eigenvec)
    H_2_eigv_basis = H_2.transform(eigenvec)

    # Complete QuTiP hamiltonian object
    args = {
        'oscillating_prefactor': oscillating_prefactor,
        'omega_p': p['omega_p']
    }
    H = [
        H_0_eigv_basis,
        [H_1_eigv_basis, H_1_coef],
        [H_2_eigv_basis, H_2_coef]
    ]

    # Dissipation operators
    c_ops = [
        (c_op_r * X_a_tensor + c_op_s * X_b_tensor).transform(eigenvec)
    ]

    return H, args, H_0_cavity_eigv_basis, c_ops


def build_hamiltonian(N_max_a, N_max_b, n_bar, p,
                      logger, out_directory, compute_kerr=False,
                      N_max_charge=100):
    """
    Build operators for simulation of a transmon in a cavity.

    :param N_max_a: Truncation on the cavity Fock space.
    :param N_max_b: Truncation on the transmon eigenstates space.
    :param n_bar: Mean number of photons in the pump, at $\omega_p$.
    :param p: Dict of parameters values.
    :param logger: A logger object.
    :param out_directory: Where to output images.
    :param compute_kerr: Whether Kerr should be computed or not.
    :param N_max_charge: To compute the transmon eigenvectors, we use the
        representation in the charge states space. This is the truncation in
        this space and should be really large (indices range from -N_max_charge
        to +N_max_charge).
    :returns: A tuple of H, args, c_ops and computed params.
    """
    # Compute corresponding pump energy
    epsilon_p = compute_epsilon_p(n_bar, p)

    # Note: This is the place where we define the hamiltonian and the collapse
    # operators. Currently, only shunted transmon and unshunted (regular)
    # transmon are supported. You can easily extend this to match your specific
    # hamiltonian.
    if 'epsilon_l' in p and p['epsilon_l'] > 0:
        logger.info('Building shunted hamiltonian...')
        H, args, H_0_cavity, c_ops = build_shunted_hamiltonian(
            N_max_a, N_max_b, epsilon_p, p
        )
    else:
        logger.info('Building unshunted hamiltonian...')
        H, args, H_0_cavity, c_ops = build_unshunted_hamiltonian(
            N_max_a, N_max_b, N_max_charge, epsilon_p, p
        )

    # Compute kerr and plot eigvals
    computed_params = []
    if compute_kerr:
        computed_params = compute_eigvals_kerr_f_0(
            H[0] + H[1][0],
            H_0_cavity,
            p,
            out_directory
        )
        logger.info('Kerr value is %g MHz.', computed_params[0] * 1e3)
        logger.info('Frequency at nbar=0 is %g GHz.', computed_params[1])

    return H, args, c_ops, computed_params
