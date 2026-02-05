"""
WoMa mixed HHe with heavy elements equations of state
"""

import numpy as np
from numba import njit
import h5py

from woma.misc import glob_vars as gv
from woma.misc import utils as ut
from woma.misc import io


def load_table_mixed(filename):
    """Load the table data from hdf5. See Material_Mixed_HHe_Heavy.gen_write_table()."""
    with h5py.File(filename, "r") as f:
        # Header attributes
        name = f["Header"].attrs[io.Di_hdf5_eos_label["name"]]
        version_date = f["Header"].attrs[io.Di_hdf5_eos_label["version_date"]]
        Y_X = f["Header"].attrs[io.Di_hdf5_eos_label["Y/X"]]
        num_mix = f["Header"].attrs[io.Di_hdf5_eos_label["num_mix"]]
        num_rho = f["Header"].attrs[io.Di_hdf5_eos_label["num_rho"]]
        num_T = f["Header"].attrs[io.Di_hdf5_eos_label["num_T"]]

        # Table data
        A1_mix = f["table/" + io.Di_hdf5_eos_label["A1_mix"]][()]
        A1_rho = f["table/" + io.Di_hdf5_eos_label["A1_rho"]][()]
        A1_T = f["table/" + io.Di_hdf5_eos_label["A1_T"]][()]
        A3_u = f["table/" + io.Di_hdf5_eos_label["A3_u"]][()]
        A3_P = f["table/" + io.Di_hdf5_eos_label["A3_P"]][()]
        A3_c = f["table/" + io.Di_hdf5_eos_label["A3_c"]][()]
        A3_s = f["table/" + io.Di_hdf5_eos_label["A3_s"]][()]

    # Checks
    assert num_mix == len(A1_mix)
    assert num_rho == len(A1_rho)
    assert num_T == len(A1_T)
    assert A3_u.shape == (num_mix, num_rho, num_T)
    assert A3_P.shape == (num_mix, num_rho, num_T)
    assert A3_c.shape == (num_mix, num_rho, num_T)
    assert A3_s.shape == (num_mix, num_rho, num_T)

    return A1_mix, np.log(A1_rho), np.log(A1_T), A3_u, A3_P, A3_c, A3_s


# ========
# Awkwardly initialise tables as global variables needed for numba
# ========
A1_mixed_mat_id = np.array(
    [gv.id_mixed_HHe_rock, gv.id_mixed_HHe_water, gv.id_mixed_HHe_iron]
)
(
    A1_mix_mixed_HHe_rock,
    A1_log_rho_mixed_HHe_rock,
    A1_log_T_mixed_HHe_rock,
    A3_u_mixed_HHe_rock,
    A3_P_mixed_HHe_rock,
    A3_c_mixed_HHe_rock,
    A3_s_mixed_HHe_rock,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2)),
)
(
    A1_mix_mixed_HHe_water,
    A1_log_rho_mixed_HHe_water,
    A1_log_T_mixed_HHe_water,
    A3_u_mixed_HHe_water,
    A3_P_mixed_HHe_water,
    A3_c_mixed_HHe_water,
    A3_s_mixed_HHe_water,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2)),
)
(
    A1_mix_mixed_HHe_iron,
    A1_log_rho_mixed_HHe_iron,
    A1_log_T_mixed_HHe_iron,
    A3_u_mixed_HHe_iron,
    A3_P_mixed_HHe_iron,
    A3_c_mixed_HHe_iron,
    A3_s_mixed_HHe_iron,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2)),
)


# ========
# Generic
# ========
@njit
def _Z_rho_T_single(rho, T, mat_id, mix, Z_choice):
    """Compute an equation of state parameter from the density and temperature.

    For a single heavy mixed component.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    mix : float
        Mixing mass fraction.

    Z_choice : str
        The parameter to calculate, choose from:
            P       Pressure (Pa).
            u       Specific internal energy (J kg^-1).
            c       Sound speed (m s^-1).
            s       Specific entropy (J K^-1 kg^-1).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    # Unpack the arrays of Z, mix, log(rho), and log(T)
    A3_Z = np.zeros((2, 2, 2), dtype=np.float32)
    if mat_id == gv.id_mixed_HHe_rock:
        A1_mix, A1_log_rho, A1_log_T = (
            A1_mix_mixed_HHe_rock,
            A1_log_rho_mixed_HHe_rock,
            A1_log_T_mixed_HHe_rock,
        )
        if Z_choice == "P":
            A3_Z = A3_P_mixed_HHe_rock
        elif Z_choice == "u":
            A3_Z = A3_u_mixed_HHe_rock
        elif Z_choice == "c":
            A3_Z = A3_c_mixed_HHe_rock
        elif Z_choice == "s":
            A3_Z = A3_s_mixed_HHe_rock
    elif mat_id == gv.id_mixed_HHe_water:
        A1_mix, A1_log_rho, A1_log_T = (
            A1_mix_mixed_HHe_water,
            A1_log_rho_mixed_HHe_water,
            A1_log_T_mixed_HHe_water,
        )
        if Z_choice == "P":
            A3_Z = A3_P_mixed_HHe_water
        elif Z_choice == "u":
            A3_Z = A3_u_mixed_HHe_water
        elif Z_choice == "c":
            A3_Z = A3_c_mixed_HHe_water
        elif Z_choice == "s":
            A3_Z = A3_s_mixed_HHe_water
    elif mat_id == gv.id_mixed_HHe_iron:
        A1_mix, A1_log_rho, A1_log_T = (
            A1_mix_mixed_HHe_iron,
            A1_log_rho_mixed_HHe_iron,
            A1_log_T_mixed_HHe_iron,
        )
        if Z_choice == "P":
            A3_Z = A3_P_mixed_HHe_iron
        elif Z_choice == "u":
            A3_Z = A3_u_mixed_HHe_iron
        elif Z_choice == "c":
            A3_Z = A3_c_mixed_HHe_iron
        elif Z_choice == "s":
            A3_Z = A3_s_mixed_HHe_iron
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1 or A3_Z.shape == (2, 2, 2):
        raise ValueError(
            "Please load the corresponding EoS table. See woma.load_eos_tables()."
        )

    # Convert to log
    log_rho = np.log(rho)
    log_T = np.log(T * 1)  # why is numba so weird?

    # 3D interpolation (linear with mix, log(rho), log(T)) to find Z(mix, rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Mix
    idx_mix_intp_mix = ut.find_index_and_interp(mix, A1_mix)
    idx_mix = int(idx_mix_intp_mix[0])
    intp_mix = idx_mix_intp_mix[1]

    # Density
    idx_rho_intp_rho = ut.find_index_and_interp(log_rho, A1_log_rho)
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Temperature
    idx_T_intp_T = ut.find_index_and_interp(log_T, A1_log_T)
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    # Subarrays to interpolate
    if mix == 0:
        A1_A2_Z_mix = [A3_Z[0]]
    elif mix == 1:
        A1_A2_Z_mix = [A3_Z[-1]]
    else:
        A1_A2_Z_mix = [A3_Z[idx_mix], A3_Z[idx_mix + 1]]

    A1_Z_mix = []
    for A2_Z in A1_A2_Z_mix:
        # Table values, interpolate with log values
        Z_1 = np.log(A2_Z[idx_rho, idx_T])
        Z_2 = np.log(A2_Z[idx_rho, idx_T + 1])
        Z_3 = np.log(A2_Z[idx_rho + 1, idx_T])
        Z_4 = np.log(A2_Z[idx_rho + 1, idx_T + 1])

        # Z(rho, T)
        Z = (1 - intp_rho) * ((1 - intp_T) * Z_1 + intp_T * Z_2) + intp_rho * (
            (1 - intp_T) * Z_3 + intp_T * Z_4
        )

        # Record Z for each mix
        A1_Z_mix.append(Z)

    # Extract or interpolate
    if mix in [0, 1]:
        Z = A1_Z_mix[0]
    else:
        Z = (1 - intp_mix) * A1_Z_mix[0] + intp_mix * A1_Z_mix[1]

    # Convert back from log
    return np.exp(Z)


@njit
def Z_rho_T(rho, T, A1_mix, Z_choice):
    """Compute an equation of state parameter from the density and temperature.

    For (one or) multiple heavy mixed components.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    Z_choice : str
        The parameter to calculate, choose from:
            P       Pressure (Pa).
            u       Specific internal energy (J kg^-1).
            c       Sound speed (m s^-1).
            s       Specific entropy (J K^-1 kg^-1).

    A1_mix : [float]
        Mixing mass fraction of each heavy component, currently [rock, water, iron].

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """

    # No heavy-element fraction
    mix_tot = sum(A1_mix)
    if mix_tot == 0:
        return _Z_rho_T_single(rho, T, A1_mixed_mat_id[0], 0, Z_choice)

    # Accumulate contribution from each non-zero mix
    Z = 0
    for mix, mat_id in zip(A1_mix, A1_mixed_mat_id):
        if mix > 0:
            # Evaluate for this single heavy mix
            Z_mat = _Z_rho_T_single(rho, T, mat_id, mix, Z_choice)

            Z += Z_mat * mix / mix_tot

    return Z


@njit
def A1_Z_rho_T(A1_rho, A1_T, A1_mix, Z_choice):
    """Compute equation of state parameters from arrays of density and
    temperature, for mixed EoS with (one or) multiple heavy mixed components.

    Parameters
    ----------
    A1_rho : [float]
        Densities (kg m^-3).

    A1_T : [float]
        Temperatures (K).

    A1_mix : [float]
        Constant mixing mass fraction of each heavy component: [rock, water, iron].

    Z_choice : str
        The parameter to calculate, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            c       Sound speed.

    Returns
    -------
    A1_Z : float
        The chosen parameter values (SI).
    """

    assert A1_rho.ndim == 1
    assert A1_T.ndim == 1
    assert A1_mix.ndim == 1
    assert A1_rho.shape[0] == A1_T.shape[0]
    assert A1_mix.shape[0] == A1_mixed_mat_id.shape[0]

    A1_Z = np.zeros_like(A1_rho)

    for i, rho in enumerate(A1_rho):
        A1_Z[i] = Z_rho_T(rho, A1_T[i], A1_mix, Z_choice)

    return A1_Z


@njit
def _Z_rho_Y_single(rho, Y, mat_id, mix, Z_choice, Y_choice):
    """Compute an equation of state parameter from the density and another
    parameter, for mixed EoS with a single mixed component.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    Y : float
        The chosen input parameter (SI).

    mat_id : int
        Material id.

    mix : float
        Mixing mass fraction.

    Z_choice, Y_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure (Pa).
            u       Specific internal energy (J kg^-1).
            c       Sound speed (m s^-1).
            s       Specific entropy (J K^-1 kg^-1).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    if Y_choice == "T":
        return _Z_rho_T_single(rho, Y, mat_id, mix, Z_choice)

    # Unpack the arrays of Z, mix, log(rho), and log(T)
    A3_Z = np.zeros((2, 2, 2), dtype=np.float32)
    if mat_id == gv.id_mixed_HHe_rock:
        A1_mix, A1_log_rho, A1_log_T = (
            A1_mix_mixed_HHe_rock,
            A1_log_rho_mixed_HHe_rock,
            A1_log_T_mixed_HHe_rock,
        )
        if Z_choice == "P":
            A3_Z = A3_P_mixed_HHe_rock
        elif Z_choice == "u":
            A3_Z = A3_u_mixed_HHe_rock
        elif Z_choice == "c":
            A3_Z = A3_c_mixed_HHe_rock
        elif Z_choice == "s":
            A3_Z = A3_s_mixed_HHe_rock
        if Y_choice == "P":
            A3_Y = A3_P_mixed_HHe_rock
        elif Y_choice == "u":
            A3_Y = A3_u_mixed_HHe_rock
        elif Y_choice == "c":
            A3_Y = A3_c_mixed_HHe_rock
        elif Y_choice == "s":
            A3_Y = A3_s_mixed_HHe_rock
    elif mat_id == gv.id_mixed_HHe_water:
        A1_mix, A1_log_rho, A1_log_T = (
            A1_mix_mixed_HHe_water,
            A1_log_rho_mixed_HHe_water,
            A1_log_T_mixed_HHe_water,
        )
        if Z_choice == "P":
            A3_Z = A3_P_mixed_HHe_water
        elif Z_choice == "u":
            A3_Z = A3_u_mixed_HHe_water
        elif Z_choice == "c":
            A3_Z = A3_c_mixed_HHe_water
        elif Z_choice == "s":
            A3_Z = A3_s_mixed_HHe_water
        if Y_choice == "P":
            A3_Y = A3_P_mixed_HHe_water
        elif Y_choice == "u":
            A3_Y = A3_u_mixed_HHe_water
        elif Y_choice == "c":
            A3_Y = A3_c_mixed_HHe_water
        elif Y_choice == "s":
            A3_Y = A3_s_mixed_HHe_water
    elif mat_id == gv.id_mixed_HHe_iron:
        A1_mix, A1_log_rho, A1_log_T = (
            A1_mix_mixed_HHe_iron,
            A1_log_rho_mixed_HHe_iron,
            A1_log_T_mixed_HHe_iron,
        )
        if Z_choice == "P":
            A3_Z = A3_P_mixed_HHe_iron
        elif Z_choice == "u":
            A3_Z = A3_u_mixed_HHe_iron
        elif Z_choice == "c":
            A3_Z = A3_c_mixed_HHe_iron
        elif Z_choice == "s":
            A3_Z = A3_s_mixed_HHe_iron
        if Y_choice == "P":
            A3_Y = A3_P_mixed_HHe_iron
        elif Y_choice == "u":
            A3_Y = A3_u_mixed_HHe_iron
        elif Y_choice == "c":
            A3_Y = A3_c_mixed_HHe_iron
        elif Y_choice == "s":
            A3_Y = A3_s_mixed_HHe_iron
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1 or A3_Z.shape == (2, 2, 2):
        raise ValueError(
            "Please load the corresponding EoS table. See woma.load_eos_tables()."
        )

    # Convert to log
    log_rho = np.log(rho)
    log_Y = np.log(Y)

    # 3D interpolation (linear with mix, log(rho), log(Y)) to find Z(mix, rho, Y).
    # If rho and/or Y are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Mix
    idx_mix_intp_mix = ut.find_index_and_interp(mix, A1_mix)
    idx_mix = int(idx_mix_intp_mix[0])
    intp_mix = idx_mix_intp_mix[1]

    # Subarrays to interpolate
    if mix == 0:
        A1_A2_Z_mix = [A3_Z[0]]
        A1_A2_Y_mix = [A3_Y[0]]
    elif mix == 1:
        A1_A2_Z_mix = [A3_Z[-1]]
        A1_A2_Y_mix = [A3_Y[-1]]
    else:
        A1_A2_Z_mix = [A3_Z[idx_mix], A3_Z[idx_mix + 1]]
        A1_A2_Y_mix = [A3_Y[idx_mix], A3_Y[idx_mix + 1]]

    # Density
    idx_rho_intp_rho = ut.find_index_and_interp(log_rho, A1_log_rho)
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Y in this and the next density slice, for each mix-selected 2D Y array
    A1_idx_Y_1_mix = []
    A1_intp_Y_1_mix = []
    A1_idx_Y_2_mix = []
    A1_intp_Y_2_mix = []
    for A2_Y in A1_A2_Y_mix:
        A2_log_Y = np.log(A2_Y)
        idx_Y_1_intp_Y_1 = ut.find_index_and_interp(log_Y, A2_log_Y[idx_rho])
        idx_Y_1 = int(idx_Y_1_intp_Y_1[0])
        intp_Y_1 = idx_Y_1_intp_Y_1[1]
        idx_Y_2_intp_Y_2 = ut.find_index_and_interp(log_Y, A2_log_Y[idx_rho + 1])
        idx_Y_2 = int(idx_Y_2_intp_Y_2[0])
        intp_Y_2 = idx_Y_2_intp_Y_2[1]

        # Record for each mix
        A1_idx_Y_1_mix.append(idx_Y_1)
        A1_intp_Y_1_mix.append(intp_Y_1)
        A1_idx_Y_2_mix.append(idx_Y_2)
        A1_intp_Y_2_mix.append(intp_Y_2)

    # Interpolate for each mix table
    A1_Z_mix = []
    for A2_Z, idx_Y_1, intp_Y_1, idx_Y_2, intp_Y_2 in zip(
        A1_A2_Z_mix, A1_idx_Y_1_mix, A1_intp_Y_1_mix, A1_idx_Y_2_mix, A1_intp_Y_2_mix
    ):
        # Table values, interpolate with log values
        Z_1 = np.log(A2_Z[idx_rho, idx_Y_1])
        Z_2 = np.log(A2_Z[idx_rho, idx_Y_1 + 1])
        Z_3 = np.log(A2_Z[idx_rho + 1, idx_Y_2])
        Z_4 = np.log(A2_Z[idx_rho + 1, idx_Y_2 + 1])

        # Z(rho, Y)
        Z = (1 - intp_rho) * ((1 - intp_Y_1) * Z_1 + intp_Y_1 * Z_2) + intp_rho * (
            (1 - intp_Y_2) * Z_3 + intp_Y_2 * Z_4
        )

        # Record Z for each mix
        A1_Z_mix.append(Z)

    # Extract or interpolate
    if mix in [0, 1]:
        Z = A1_Z_mix[0]
    else:
        Z = (1 - intp_mix) * A1_Z_mix[0] + intp_mix * A1_Z_mix[1]

    # Convert back from log
    return np.exp(Z)


@njit
def Z_rho_Y(rho, Y, A1_mix, Z_choice, Y_choice):
    """Compute an equation of state parameter from the density and another
    parameter, for mixed EoS with (one or) multiple heavy mixed components.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    Y : float
        The chosen input parameter (SI).

    A1_mix : [float]
        Mixing mass fraction of each heavy component, currently [rock, water, iron].

    Z_choice, Y_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag (Z_choice only).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    # No heavy-element fraction
    mix_tot = sum(A1_mix)
    if mix_tot == 0:
        return _Z_rho_Y_single(rho, Y, A1_mixed_mat_id[0], 0, Z_choice, Y_choice)

    # Accumulate contribution from each non-zero mix
    Z = 0
    for mix, mat_id in zip(A1_mix, A1_mixed_mat_id):
        if mix > 0:
            # Evaluate for this single heavy mix
            Z_mat = _Z_rho_Y_single(rho, Y, mat_id, mix, Z_choice, Y_choice)

            Z += Z_mat * mix / mix_tot

    return Z


@njit
def A1_Z_rho_Y_mix(A1_rho, A1_Y, A1_A1_mix, Z_choice, Y_choice):
    """Compute equation of state parameters from arrays of density and
    temperature, for mixed EoS with (one or) multiple heavy mixed components.

    Parameters
    ----------
    A1_rho : [float]
        Densities (kg m^-3).

    A1_Y : [float]
        The chosen input parameter (SI).

    A1_A1_mix : [[float]]
        Mixing mass fractions of each heavy component, currently [rock, water, iron].

    Z_choice, Y_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            c       Sound speed.

    Returns
    -------
    A1_Z : float
        The chosen parameter values (SI).
    """

    assert A1_rho.ndim == 1
    assert A1_Y.ndim == 1
    assert A1_A1_mix.ndim == 2
    assert A1_rho.shape[0] == A1_Y.shape[0]
    assert A1_rho.shape[0] == A1_A1_mix.shape[0]

    A1_Z = np.zeros_like(A1_rho)

    for i, rho in enumerate(A1_rho):
        A1_Z[i] = Z_rho_Y(rho, A1_Y[i], A1_A1_mix[i], Z_choice, Y_choice)

    return A1_Z


@njit
def _Z_T_Y_single(T, Y, mat_id, mix, Z_choice, Y_choice):
    """Compute an equation of state parameter from the temperature and another
    parameter, for mixed EoS with a single mixed component.

    Parameters
    ----------
    T : float
        Temperature (K).

    Y : float
        The chosen input parameter (SI).

    mat_id : int
        Material id.

    mix : float
        Mixing mass fraction.

    Z_choice, Y_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure (Pa).
            u       Specific internal energy (J kg^-1).
            c       Sound speed (m s^-1).
            s       Specific entropy (J K^-1 kg^-1).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    if Y_choice == "rho":
        return _Z_rho_T_single(Y, T, mat_id, mix, Z_choice)

    # Unpack the arrays of Z, mix, log(rho), and log(T)
    A3_Z = np.zeros((2, 2, 2), dtype=np.float32)
    if mat_id == gv.id_mixed_HHe_rock:
        A1_mix, A1_log_rho, A1_log_T = (
            A1_mix_mixed_HHe_rock,
            A1_log_rho_mixed_HHe_rock,
            A1_log_T_mixed_HHe_rock,
        )
        if Z_choice == "P":
            A3_Z = A3_P_mixed_HHe_rock
        elif Z_choice == "u":
            A3_Z = A3_u_mixed_HHe_rock
        elif Z_choice == "c":
            A3_Z = A3_c_mixed_HHe_rock
        elif Z_choice == "s":
            A3_Z = A3_s_mixed_HHe_rock
        if Y_choice == "P":
            A3_Y = A3_P_mixed_HHe_rock
        elif Y_choice == "u":
            A3_Y = A3_u_mixed_HHe_rock
        elif Y_choice == "c":
            A3_Y = A3_c_mixed_HHe_rock
        elif Y_choice == "s":
            A3_Y = A3_s_mixed_HHe_rock
    elif mat_id == gv.id_mixed_HHe_water:
        A1_mix, A1_log_rho, A1_log_T = (
            A1_mix_mixed_HHe_water,
            A1_log_rho_mixed_HHe_water,
            A1_log_T_mixed_HHe_water,
        )
        if Z_choice == "P":
            A3_Z = A3_P_mixed_HHe_water
        elif Z_choice == "u":
            A3_Z = A3_u_mixed_HHe_water
        elif Z_choice == "c":
            A3_Z = A3_c_mixed_HHe_water
        elif Z_choice == "s":
            A3_Z = A3_s_mixed_HHe_water
        if Y_choice == "P":
            A3_Y = A3_P_mixed_HHe_water
        elif Y_choice == "u":
            A3_Y = A3_u_mixed_HHe_water
        elif Y_choice == "c":
            A3_Y = A3_c_mixed_HHe_water
        elif Y_choice == "s":
            A3_Y = A3_s_mixed_HHe_water
    elif mat_id == gv.id_mixed_HHe_iron:
        A1_mix, A1_log_rho, A1_log_T = (
            A1_mix_mixed_HHe_iron,
            A1_log_rho_mixed_HHe_iron,
            A1_log_T_mixed_HHe_iron,
        )
        if Z_choice == "P":
            A3_Z = A3_P_mixed_HHe_iron
        elif Z_choice == "u":
            A3_Z = A3_u_mixed_HHe_iron
        elif Z_choice == "c":
            A3_Z = A3_c_mixed_HHe_iron
        elif Z_choice == "s":
            A3_Z = A3_s_mixed_HHe_iron
        if Y_choice == "P":
            A3_Y = A3_P_mixed_HHe_iron
        elif Y_choice == "u":
            A3_Y = A3_u_mixed_HHe_iron
        elif Y_choice == "c":
            A3_Y = A3_c_mixed_HHe_iron
        elif Y_choice == "s":
            A3_Y = A3_s_mixed_HHe_iron
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1 or A3_Z.shape == (2, 2, 2):
        raise ValueError(
            "Please load the corresponding EoS table. See woma.load_eos_tables()."
        )

    # Convert to log
    log_T = np.log(T)
    log_Y = np.log(Y)

    # 3D interpolation (linear with mix, log(T), log(Y)) to find Z(mix, T, Y).
    # If T and/or Y are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Mix
    idx_mix_intp_mix = ut.find_index_and_interp(mix, A1_mix)
    idx_mix = int(idx_mix_intp_mix[0])
    intp_mix = idx_mix_intp_mix[1]

    # Subarrays to interpolate
    if mix == 0:
        A1_A2_Z_mix = [A3_Z[0]]
        A1_A2_Y_mix = [A3_Y[0]]
    elif mix == 1:
        A1_A2_Z_mix = [A3_Z[-1]]
        A1_A2_Y_mix = [A3_Y[-1]]
    else:
        A1_A2_Z_mix = [A3_Z[idx_mix], A3_Z[idx_mix + 1]]
        A1_A2_Y_mix = [A3_Y[idx_mix], A3_Y[idx_mix + 1]]

    # Density
    idx_T_intp_T = ut.find_index_and_interp(log_T, A1_log_T)
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    # Y in this and the next temperature slice, for each mix-selected 2D Y array
    A1_idx_Y_1_mix = []
    A1_intp_Y_1_mix = []
    A1_idx_Y_2_mix = []
    A1_intp_Y_2_mix = []
    for A2_Y in A1_A2_Y_mix:
        A2_log_Y = np.log(A2_Y)
        idx_Y_1_intp_Y_1 = ut.find_index_and_interp(log_Y, A2_log_Y[:, idx_T])
        idx_Y_1 = int(idx_Y_1_intp_Y_1[0])
        intp_Y_1 = idx_Y_1_intp_Y_1[1]
        idx_Y_2_intp_Y_2 = ut.find_index_and_interp(log_Y, A2_log_Y[:, idx_T + 1])
        idx_Y_2 = int(idx_Y_2_intp_Y_2[0])
        intp_Y_2 = idx_Y_2_intp_Y_2[1]

        # Record for each mix
        A1_idx_Y_1_mix.append(idx_Y_1)
        A1_intp_Y_1_mix.append(intp_Y_1)
        A1_idx_Y_2_mix.append(idx_Y_2)
        A1_intp_Y_2_mix.append(intp_Y_2)

    # Interpolate for each mix table
    A1_Z_mix = []
    for A2_Z, idx_Y_1, intp_Y_1, idx_Y_2, intp_Y_2 in zip(
        A1_A2_Z_mix, A1_idx_Y_1_mix, A1_intp_Y_1_mix, A1_idx_Y_2_mix, A1_intp_Y_2_mix
    ):
        # Table values, interpolate with log values
        Z_1 = np.log(A2_Z[idx_T, idx_Y_1])
        Z_2 = np.log(A2_Z[idx_T, idx_Y_1 + 1])
        Z_3 = np.log(A2_Z[idx_T + 1, idx_Y_2])
        Z_4 = np.log(A2_Z[idx_T + 1, idx_Y_2 + 1])

        # Z(T, Y)
        Z = (1 - intp_T) * ((1 - intp_Y_1) * Z_1 + intp_Y_1 * Z_2) + intp_T * (
            (1 - intp_Y_2) * Z_3 + intp_Y_2 * Z_4
        )

        # Record Z for each mix
        A1_Z_mix.append(Z)

    # Extract or interpolate
    if mix in [0, 1]:
        Z = A1_Z_mix[0]
    else:
        Z = (1 - intp_mix) * A1_Z_mix[0] + intp_mix * A1_Z_mix[1]

    # Convert back from log
    return np.exp(Z)


@njit
def Z_T_Y(T, Y, A1_mix, Z_choice, Y_choice):
    """Compute an equation of state parameter from the temperature and another
    parameter, for mixed EoS with (one or) multiple heavy mixed components.

    Parameters
    ----------
    T : float
        Temperature (K).

    Y : float
        The chosen input parameter (SI).

    A1_mix : [float]
        Mixing mass fraction of each heavy component, currently [rock, water, iron].

    Z_choice, Y_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag (Z_choice only).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    # No heavy-element fraction
    mix_tot = sum(A1_mix)
    if mix_tot == 0:
        return _Z_T_Y_single(T, Y, A1_mixed_mat_id[0], 0, Z_choice, Y_choice)

    # Accumulate contribution from each non-zero mix
    Z = 0
    for mix, mat_id in zip(A1_mix, A1_mixed_mat_id):
        if mix > 0:
            # Evaluate for this single heavy mix
            Z_mat = _Z_T_Y_single(T, Y, mat_id, mix, Z_choice, Y_choice)

            Z += Z_mat * mix / mix_tot

    return Z


@njit
def A1_Z_T_Y_mix(A1_T, A1_Y, A1_A1_mix, Z_choice, Y_choice):
    """Compute equation of state parameters from arrays of density and
    temperature, for mixed EoS with (one or) multiple heavy mixed components.

    Parameters
    ----------
    A1_T : [float]
        Temperatures (K).

    A1_Y : [float]
        The chosen input parameter (SI).

    A1_A1_mix : [[float]]
        Mixing mass fractions of each heavy component, currently [rock, water, iron].

    Z_choice, Y_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            c       Sound speed.

    Returns
    -------
    A1_Z : float
        The chosen parameter values (SI).
    """

    assert A1_T.ndim == 1
    assert A1_Y.ndim == 1
    assert A1_A1_mix.ndim == 2
    assert A1_T.shape[0] == A1_Y.shape[0]
    assert A1_T.shape[0] == A1_A1_mix.shape[0]

    A1_Z = np.zeros_like(A1_T)

    for i, T in enumerate(A1_T):
        A1_Z[i] = Z_T_Y(T, A1_Y[i], A1_A1_mix[i], Z_choice, Y_choice)

    return A1_Z
