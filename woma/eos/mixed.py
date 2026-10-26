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
        A1_mix = f["Table/" + io.Di_hdf5_eos_label["mix"]][()]
        A1_rho = f["Table/" + io.Di_hdf5_eos_label["rho"]][()]
        A1_T = f["Table/" + io.Di_hdf5_eos_label["T"]][()]
        A3_u = f["Table/" + io.Di_hdf5_eos_label["u"]][()]
        A3_P = f["Table/" + io.Di_hdf5_eos_label["P"]][()]
        A3_c = f["Table/" + io.Di_hdf5_eos_label["c"]][()]
        A3_s = f["Table/" + io.Di_hdf5_eos_label["s"]][()]

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
    if A1_mix is None:
        raise ValueError("A1_mix cannot be None for mixed_HHe_heavy")

    mix_tot = sum(A1_mix)
    num_mat = len(A1_mix)

    # Y arrays for each heavy material's mixes
    if Z_choice == "P":
        A3_Z_rock = A3_P_mixed_HHe_rock
    elif Z_choice == "u":
        A3_Z_rock = A3_u_mixed_HHe_rock
    elif Z_choice == "c":
        A3_Z_rock = A3_c_mixed_HHe_rock
    elif Z_choice == "s":
        A3_Z_rock = A3_s_mixed_HHe_rock
    else:
        raise ValueError("Invalid Z_choice")

    if Z_choice == "P":
        A3_Z_water = A3_P_mixed_HHe_water
    elif Z_choice == "u":
        A3_Z_water = A3_u_mixed_HHe_water
    elif Z_choice == "c":
        A3_Z_water = A3_c_mixed_HHe_water
    elif Z_choice == "s":
        A3_Z_water = A3_s_mixed_HHe_water
    else:
        raise ValueError("Invalid Z_choice")

    if Z_choice == "P":
        A3_Z_iron = A3_P_mixed_HHe_iron
    elif Z_choice == "u":
        A3_Z_iron = A3_u_mixed_HHe_iron
    elif Z_choice == "c":
        A3_Z_iron = A3_c_mixed_HHe_iron
    elif Z_choice == "s":
        A3_Z_iron = A3_s_mixed_HHe_iron
    else:
        raise ValueError("Invalid Z_choice")

    # Check necessary data loaded
    if (
        A3_Z_rock.shape == (2, 2, 2)
        or A3_Z_water.shape == (2, 2, 2)
        or A3_Z_iron.shape == (2, 2, 2)
    ):
        raise ValueError(
            "Please load the corresponding EoS table. See woma.load_eos_tables()."
        )

    # Check same table structure
    # assert np.array_equal(A1_mix_mixed_HHe_rock, A1_mix_mixed_HHe_water)
    # assert np.array_equal(A1_mix_mixed_HHe_rock, A1_mix_mixed_HHe_iron)
    # assert np.array_equal(A1_log_rho_mixed_HHe_rock, A1_log_rho_mixed_HHe_water)
    # assert np.array_equal(A1_log_rho_mixed_HHe_rock, A1_log_rho_mixed_HHe_iron)
    # assert np.array_equal(A1_log_T_mixed_HHe_rock, A1_log_T_mixed_HHe_water)
    # assert np.array_equal(A1_log_T_mixed_HHe_rock, A1_log_T_mixed_HHe_iron)

    # Shared table axis arrays
    A1_mix_table = A1_mix_mixed_HHe_rock
    A1_log_rho = A1_log_rho_mixed_HHe_rock
    A1_log_T = A1_log_T_mixed_HHe_rock

    # Convert to log
    log_rho = np.log(rho)
    log_T = np.log(T)

    # 3D interpolation (linear with mix, log(rho), log(T)) to find Z(mix, rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Mix
    idx_mix_intp_mix = ut.find_index_and_interp(mix_tot, A1_mix_table)
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

    # Table values, interpolate with log values
    log_Z_1 = 0
    log_Z_2 = 0
    log_Z_3 = 0
    log_Z_4 = 0
    for i_mix, A3_Z in enumerate([A3_Z_rock, A3_Z_water, A3_Z_iron]):
        if mix_tot == 0:
            f_mix = 1 / num_mat
        else:
            f_mix = A1_mix[i_mix] / mix_tot
        log_Z_1 += f_mix * (
            (1 - intp_mix) * np.log(A3_Z[idx_mix, idx_rho, idx_T])
            + intp_mix * np.log(A3_Z[idx_mix + 1, idx_rho, idx_T])
        )
        log_Z_2 += f_mix * (
            (1 - intp_mix) * np.log(A3_Z[idx_mix, idx_rho, idx_T + 1])
            + intp_mix * np.log(A3_Z[idx_mix + 1, idx_rho, idx_T + 1])
        )
        log_Z_3 += f_mix * (
            (1 - intp_mix) * np.log(A3_Z[idx_mix, idx_rho + 1, idx_T])
            + intp_mix * np.log(A3_Z[idx_mix + 1, idx_rho + 1, idx_T])
        )
        log_Z_4 += f_mix * (
            (1 - intp_mix) * np.log(A3_Z[idx_mix, idx_rho + 1, idx_T + 1])
            + intp_mix * np.log(A3_Z[idx_mix + 1, idx_rho + 1, idx_T + 1])
        )

    # Z(rho, T)
    log_Z = (1 - intp_rho) * ((1 - intp_T) * log_Z_1 + intp_T * log_Z_2) + intp_rho * (
        (1 - intp_T) * log_Z_3 + intp_T * log_Z_4
    )

    return np.exp(log_Z)


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
def T_rho_Y(rho, Y, A1_mix, Y_choice):
    """Compute the temperature from the density and another equation of state
    parameter, for mixed EoS with (one or) multiple heavy mixed components.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    Y : float
        The chosen input parameter (SI).

    A1_mix : [float]
        Mixing mass fraction of each heavy component, currently [rock, water, iron].

    Y_choice : str
        The other input parameter, choose from:
            P       Pressure (Pa).
            u       Specific internal energy (J kg^-1).
            c       Sound speed (m s^-1).
            s       Specific entropy (J K^-1 kg^-1).

    Returns
    -------
    T : float
        The temperature (K).
    """
    if A1_mix is None:
        raise ValueError("A1_mix cannot be None for mixed_HHe_heavy")

    num_mat = len(A1_mix)

    # Y arrays for each heavy material's mixes
    if Y_choice == "P":
        A3_Y_rock = A3_P_mixed_HHe_rock
    elif Y_choice == "u":
        A3_Y_rock = A3_u_mixed_HHe_rock
    elif Y_choice == "c":
        A3_Y_rock = A3_c_mixed_HHe_rock
    elif Y_choice == "s":
        A3_Y_rock = A3_s_mixed_HHe_rock
    else:
        raise ValueError("Invalid Y_choice")

    if Y_choice == "P":
        A3_Y_water = A3_P_mixed_HHe_water
    elif Y_choice == "u":
        A3_Y_water = A3_u_mixed_HHe_water
    elif Y_choice == "c":
        A3_Y_water = A3_c_mixed_HHe_water
    elif Y_choice == "s":
        A3_Y_water = A3_s_mixed_HHe_water
    else:
        raise ValueError("Invalid Y_choice")

    if Y_choice == "P":
        A3_Y_iron = A3_P_mixed_HHe_iron
    elif Y_choice == "u":
        A3_Y_iron = A3_u_mixed_HHe_iron
    elif Y_choice == "c":
        A3_Y_iron = A3_c_mixed_HHe_iron
    elif Y_choice == "s":
        A3_Y_iron = A3_s_mixed_HHe_iron
    else:
        raise ValueError("Invalid Y_choice")

    # Check necessary data loaded
    if (
        A3_Y_rock.shape == (2, 2, 2)
        or A3_Y_water.shape == (2, 2, 2)
        or A3_Y_iron.shape == (2, 2, 2)
    ):
        raise ValueError(
            "Please load the corresponding EoS table. See woma.load_eos_tables()."
        )

    # Shared table axis arrays
    A1_mix_table = A1_mix_mixed_HHe_rock
    A1_log_rho = A1_log_rho_mixed_HHe_rock
    A1_log_T = A1_log_T_mixed_HHe_rock

    # Convert to log
    log_rho = np.log(rho)
    log_Y = np.log(Y)

    # Mix fraction with respect to HHe rather than between heavy components
    mix_tot = np.sum(A1_mix)

    # Find the relevant mixing fractions for interpolation
    idx_mix_intp_mix = ut.find_index_and_interp(mix_tot, A1_mix_table)
    idx_mix = int(idx_mix_intp_mix[0])
    intp_mix = idx_mix_intp_mix[1]

    # Density
    idx_rho_intp_rho = ut.find_index_and_interp(log_rho, A1_log_rho)
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Do the mix interpolation and rho extraction to derive a single Y(rho, T) array
    A2_log_Y = np.zeros((2, len(A1_log_T)))
    for i_rho in [0, 1]:
        for i_mix, A3_Y in enumerate([A3_Y_rock, A3_Y_water, A3_Y_iron]):
            if mix_tot == 0:
                A2_log_Y[i_rho] += np.log(A3_Y[0, idx_rho + i_rho]) / num_mat
            elif mix_tot == 1:
                A2_log_Y[i_rho] += (
                    A1_mix[i_mix] / mix_tot * np.log(A3_Y[-1, idx_rho + i_rho])
                )
            else:
                A2_log_Y[i_rho] += (
                    A1_mix[i_mix]
                    / mix_tot
                    * (
                        (1 - intp_mix) * np.log(A3_Y[idx_mix, idx_rho + i_rho])
                        + intp_mix * np.log(A3_Y[idx_mix + 1, idx_rho + i_rho])
                    )
                )

    # Y (in each density slice of the 2D u array)
    idx_Y_1_intp_Y_1 = ut.find_index_and_interp(log_Y, A2_log_Y[0])
    idx_Y_1 = int(idx_Y_1_intp_Y_1[0])
    intp_Y_1 = idx_Y_1_intp_Y_1[1]
    idx_Y_2_intp_Y_2 = ut.find_index_and_interp(log_Y, A2_log_Y[1])
    idx_Y_2 = int(idx_Y_2_intp_Y_2[0])
    intp_Y_2 = idx_Y_2_intp_Y_2[1]

    # Normal interpolation
    log_T = (1 - intp_rho) * (
        (1 - intp_Y_1) * A1_log_T[idx_Y_1] + intp_Y_1 * A1_log_T[idx_Y_1 + 1]
    ) + intp_rho * (
        (1 - intp_Y_2) * A1_log_T[idx_Y_2] + intp_Y_2 * A1_log_T[idx_Y_2 + 1]
    )

    # Convert back from log
    return np.exp(log_T)


@njit
def rho_X_T(X, T, A1_mix, X_choice):
    """Compute the density from the temperature and another equation of state
    parameter, for mixed EoS with (one or) multiple heavy mixed components.

    ## Not yet fully tested

    Parameters
    ----------
    X : float
        The chosen input parameter (SI).

    T : float
        Temperature (K).

    A1_mix : [float]
        Mixing mass fraction of each heavy component, currently [rock, water, iron].

    X_choice : str
        The other input parameter, choose from:
            P       Pressure (Pa).
            u       Specific internal energy (J kg^-1).
            c       Sound speed (m s^-1).
            s       Specific entropy (J K^-1 kg^-1).

    Returns
    -------
    rho : float
        The Density (kg m^-3).
    """
    if A1_mix is None:
        raise ValueError("A1_mix cannot be None for mixed_HHe_heavy")

    num_mat = len(A1_mix)

    # X arrays for each heavy material's mixes
    if X_choice == "P":
        A3_X_rock = A3_P_mixed_HHe_rock
    elif X_choice == "u":
        A3_X_rock = A3_u_mixed_HHe_rock
    elif X_choice == "c":
        A3_X_rock = A3_c_mixed_HHe_rock
    elif X_choice == "s":
        A3_X_rock = A3_s_mixed_HHe_rock
    else:
        raise ValueError("Invalid X_choice")

    if X_choice == "P":
        A3_X_water = A3_P_mixed_HHe_water
    elif X_choice == "u":
        A3_X_water = A3_u_mixed_HHe_water
    elif X_choice == "c":
        A3_X_water = A3_c_mixed_HHe_water
    elif X_choice == "s":
        A3_X_water = A3_s_mixed_HHe_water
    else:
        raise ValueError("Invalid X_choice")

    if X_choice == "P":
        A3_X_iron = A3_P_mixed_HHe_iron
    elif X_choice == "u":
        A3_X_iron = A3_u_mixed_HHe_iron
    elif X_choice == "c":
        A3_X_iron = A3_c_mixed_HHe_iron
    elif X_choice == "s":
        A3_X_iron = A3_s_mixed_HHe_iron
    else:
        raise ValueError("Invalid X_choice")

    # Check necessary data loaded
    if (
        A3_X_rock.shape == (2, 2, 2)
        or A3_X_water.shape == (2, 2, 2)
        or A3_X_iron.shape == (2, 2, 2)
    ):
        raise ValueError(
            "Please load the corresponding EoS table. See woma.load_eos_tables()."
        )

    # Shared table axis arrays
    A1_mix_table = A1_mix_mixed_HHe_rock
    A1_log_T = A1_log_T_mixed_HHe_rock
    A1_log_rho = A1_log_rho_mixed_HHe_rock

    # Convert to log
    log_T = np.log(T)
    log_X = np.log(X)

    # Mix fraction with respect to HHe rather than between heavy components
    mix_tot = np.sum(A1_mix)

    # Find the relevant mixing fractions for interpolation
    idx_mix_intp_mix = ut.find_index_and_interp(mix_tot, A1_mix_table)
    idx_mix = int(idx_mix_intp_mix[0])
    intp_mix = idx_mix_intp_mix[1]

    # Temperature
    idx_T_intp_T = ut.find_index_and_interp(log_T, A1_log_T)
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    # Do the mix interpolation and T extraction to derive a single X(rho, T) array
    A2_log_X = np.zeros((2, len(A1_log_rho)))
    for i_T in [0, 1]:
        for i_mix, A3_Y in enumerate([A3_X_rock, A3_X_water, A3_X_iron]):
            if mix_tot == 0:
                A2_log_X[i_T] += np.log(A3_Y[0, :, idx_T + i_T]) / num_mat
            elif mix_tot == 1:
                A2_log_X[i_T] += (
                    A1_mix[i_mix] / mix_tot * np.log(A3_Y[-1, :, idx_T + i_T])
                )
            else:
                A2_log_X[i_T] += (
                    A1_mix[i_mix]
                    / mix_tot
                    * (
                        (1 - intp_mix) * np.log(A3_Y[idx_mix, :, idx_T + i_T])
                        + intp_mix * np.log(A3_Y[idx_mix + 1, :, idx_T + i_T])
                    )
                )

    # X (in each temperature slice of the 2D u array)
    idx_X_1_intp_X_1 = ut.find_index_and_interp(log_X, A2_log_X[0])
    idx_X_1 = int(idx_X_1_intp_X_1[0])
    intp_X_1 = idx_X_1_intp_X_1[1]
    idx_X_2_intp_X_2 = ut.find_index_and_interp(log_X, A2_log_X[1])
    idx_X_2 = int(idx_X_2_intp_X_2[0])
    intp_X_2 = idx_X_2_intp_X_2[1]

    # Normal interpolation
    log_rho = (1 - intp_T) * (
        (1 - intp_X_1) * A1_log_rho[idx_X_1] + intp_X_1 * A1_log_rho[idx_X_1 + 1]
    ) + intp_T * (
        (1 - intp_X_2) * A1_log_rho[idx_X_2] + intp_X_2 * A1_log_rho[idx_X_2 + 1]
    )

    # Convert back from log
    return np.exp(log_rho)


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
            T       Temperature.
            phase   Phase KPA flag (Z_choice only).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """

    if A1_mix is None:
        raise ValueError("A1_mix cannot be None for mixed_HHe_heavy")

    if Y_choice == "T":
        return Z_rho_T(rho, Y, A1_mix, Z_choice)
    elif Z_choice == "T":
        return T_rho_Y(rho, Y, A1_mix, Y_choice)

    # Could instead duplicate most of T_rho_Y then use the idx and intp directly for Z
    return Z_rho_T(rho, T_rho_Y(rho, Y, A1_mix, Y_choice), A1_mix, Z_choice)


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
def Z_X_T(X, T, A1_mix, Z_choice, X_choice):
    """Compute an equation of state parameter from the temperature and another
    parameter, for mixed EoS with (one or) multiple heavy mixed components.

    Parameters
    ----------
    T : float
        Temperature (K).

    X : float
        The chosen input parameter (SI).

    A1_mix : [float]
        Mixing mass fraction of each heavy component, currently [rock, water, iron].

    Z_choice, X_choice : str
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
    if A1_mix is None:
        raise ValueError("A1_mix cannot be None for mixed_HHe_heavy")

    return Z_rho_T(rho_X_T(X, T, A1_mix, X_choice), T, A1_mix, Z_choice)


@njit
def A1_Z_X_T_mix(A1_X, A1_T, A1_A1_mix, Z_choice, X_choice):
    """Compute equation of state parameters from arrays of density and
    temperature, for mixed EoS with (one or) multiple heavy mixed components.

    Parameters
    ----------
    A1_T : [float]
        Temperatures (K).

    A1_X : [float]
        The chosen input parameter (SI).

    A1_A1_mix : [[float]]
        Mixing mass fractions of each heavy component, currently [rock, water, iron].

    Z_choice, X_choice : str
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
    assert A1_X.ndim == 1
    assert A1_A1_mix.ndim == 2
    assert A1_T.shape[0] == A1_X.shape[0]
    assert A1_T.shape[0] == A1_A1_mix.shape[0]

    A1_Z = np.zeros_like(A1_T)

    for i, T in enumerate(A1_T):
        A1_Z[i] = Z_X_T(A1_X[i], T, A1_A1_mix[i], Z_choice, X_choice)

    return A1_Z
