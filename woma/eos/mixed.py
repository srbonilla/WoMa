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
# Minimum positive float32, to avoid log(<=0) pressures, as in SWIFT
FLT_MIN = np.finfo(np.float32).tiny


@njit
def _A3_Z_mixed(Z_choice):
    """Return each heavy component's table array of the chosen parameter."""
    if Z_choice == "P":
        A3_Z_rock, A3_Z_water, A3_Z_iron = (
            A3_P_mixed_HHe_rock,
            A3_P_mixed_HHe_water,
            A3_P_mixed_HHe_iron,
        )
    elif Z_choice == "u":
        A3_Z_rock, A3_Z_water, A3_Z_iron = (
            A3_u_mixed_HHe_rock,
            A3_u_mixed_HHe_water,
            A3_u_mixed_HHe_iron,
        )
    elif Z_choice == "c":
        A3_Z_rock, A3_Z_water, A3_Z_iron = (
            A3_c_mixed_HHe_rock,
            A3_c_mixed_HHe_water,
            A3_c_mixed_HHe_iron,
        )
    elif Z_choice == "s":
        A3_Z_rock, A3_Z_water, A3_Z_iron = (
            A3_s_mixed_HHe_rock,
            A3_s_mixed_HHe_water,
            A3_s_mixed_HHe_iron,
        )
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

    return A3_Z_rock, A3_Z_water, A3_Z_iron


@njit
def _find_index_and_interp(x, A1_x):
    """Index and interpolation factor of a value in a monotonically increasing
    array, extrapolating from the edge and edge-but-one values if outside.

    Same as find_value_in_monot_incr_array() and its edge handling in SWIFT,
    with intp = 0 for duplicate elements.
    """
    idx = np.searchsorted(A1_x, x, side="right") - 1
    if idx <= -1:
        idx = 0
    elif idx >= len(A1_x) - 1:
        idx = len(A1_x) - 2

    delta = A1_x[idx + 1] - A1_x[idx]
    if delta != 0:
        intp = (x - A1_x[idx]) / delta
    else:
        intp = 0.0

    return idx, intp


@njit
def _mix_weights(A1_mix):
    """The mix-table position and each heavy component's weight.

    Parameters
    ----------
    A1_mix : [float]
        Mixing mass fraction of each heavy component, currently [rock, water, iron].

    Returns
    -------
    idx_mix : int
        The index in the mix table, clamped to the table.

    intp_mix : float
        The interpolation factor between idx_mix and idx_mix + 1.

    A1_f_mat : [float]
        The relative weight of each heavy component (equal if no heavy elements).
    """
    num_mat = len(A1_mix)
    A1_mix_table = A1_mix_mixed_HHe_rock

    # Mix fraction with respect to HHe rather than between each heavy component
    mix_tot = 0.0
    for i_mat in range(num_mat):
        mix_tot += A1_mix[i_mat]

    # Relative weight of each heavy component
    A1_f_mat = np.empty(num_mat)
    for i_mat in range(num_mat):
        if mix_tot > 0:
            A1_f_mat[i_mat] = A1_mix[i_mat] / mix_tot
        else:
            A1_f_mat[i_mat] = 1 / num_mat

    # Mix, clamped to the table
    if mix_tot >= 0.9999 or mix_tot >= A1_mix_table[-1]:
        idx_mix = len(A1_mix_table) - 2
        intp_mix = 1.0
    elif mix_tot <= A1_mix_table[0]:
        idx_mix = 0
        intp_mix = 0.0
    else:
        idx_mix, intp_mix = _find_index_and_interp(mix_tot, A1_mix_table)

    return idx_mix, intp_mix, A1_f_mat


@njit
def _log_Z_mixed(A3_Z, Z_choice, idx_mix, intp_mix, A1_f_mat, i_rho, i_T):
    """log(Z) at one table density index i_rho and/or temperature index i_T,
    interpolated in mix and combined over the heavy components.

    Pass i_rho or i_T as -1 to return the full array along that axis.
    """
    if i_rho == -1:
        log_Z = np.zeros(A3_Z[0].shape[1])
    elif i_T == -1:
        log_Z = np.zeros(A3_Z[0].shape[2])
    else:
        log_Z = np.zeros(1)

    # Add contribution from each mixing component
    for i_mat in range(len(A1_f_mat)):
        if A1_f_mat[i_mat] == 0:
            continue

        if i_rho == -1:
            Z_1 = A3_Z[i_mat][idx_mix, :, i_T]
            Z_2 = A3_Z[i_mat][idx_mix + 1, :, i_T]
        elif i_T == -1:
            Z_1 = A3_Z[i_mat][idx_mix, i_rho, :]
            Z_2 = A3_Z[i_mat][idx_mix + 1, i_rho, :]
        else:
            Z_1 = A3_Z[i_mat][idx_mix, i_rho, i_T : i_T + 1]
            Z_2 = A3_Z[i_mat][idx_mix + 1, i_rho, i_T : i_T + 1]

        if Z_choice == "P":
            # Avoid log(-ve) -- though that shouldn't occur in these tables
            log_Z_1 = np.log(np.maximum(Z_1, FLT_MIN))
            log_Z_2 = np.log(np.maximum(Z_2, FLT_MIN))
        else:
            log_Z_1 = np.log(Z_1)
            log_Z_2 = np.log(Z_2)

        log_Z += A1_f_mat[i_mat] * ((1 - intp_mix) * log_Z_1 + intp_mix * log_Z_2)

    return log_Z


@njit
def _bilinear_log_Z(
    Z_choice, idx_mix, intp_mix, A1_f_mat, idx_rho, intp_rho, idx_T, intp_T
):
    """log(Z) interpolated linearly in log(rho), log(T)."""
    A3_Z = _A3_Z_mixed(Z_choice)

    log_Z_1 = _log_Z_mixed(A3_Z, Z_choice, idx_mix, intp_mix, A1_f_mat, idx_rho, idx_T)[
        0
    ]
    log_Z_2 = _log_Z_mixed(
        A3_Z, Z_choice, idx_mix, intp_mix, A1_f_mat, idx_rho, idx_T + 1
    )[0]
    log_Z_3 = _log_Z_mixed(
        A3_Z, Z_choice, idx_mix, intp_mix, A1_f_mat, idx_rho + 1, idx_T
    )[0]
    log_Z_4 = _log_Z_mixed(
        A3_Z, Z_choice, idx_mix, intp_mix, A1_f_mat, idx_rho + 1, idx_T + 1
    )[0]

    return (1 - intp_rho) * ((1 - intp_T) * log_Z_1 + intp_T * log_Z_2) + intp_rho * (
        (1 - intp_T) * log_Z_3 + intp_T * log_Z_4
    )


@njit
def _idx_intp_T_rho_Y(rho, Y, idx_mix, intp_mix, A1_f_mat, Y_choice):
    """The table density and temperature indices and interpolation factors
    for the density and another parameter.

    Interpolates each temperature column to this density before inverting, so
    that the inversion and any later evaluation use the same (rho, T).
    """
    A3_Y = _A3_Z_mixed(Y_choice)

    # Density
    idx_rho, intp_rho = _find_index_and_interp(np.log(rho), A1_log_rho_mixed_HHe_rock)

    # Y(T) at this density
    A1_log_Y_T = (1 - intp_rho) * _log_Z_mixed(
        A3_Y, Y_choice, idx_mix, intp_mix, A1_f_mat, idx_rho, -1
    ) + intp_rho * _log_Z_mixed(
        A3_Y, Y_choice, idx_mix, intp_mix, A1_f_mat, idx_rho + 1, -1
    )

    # Temperature
    idx_T, intp_T = _find_index_and_interp(np.log(Y), A1_log_Y_T)

    # If below the minimum Y at this rho then just use the lowest table values
    if idx_rho > 0 and intp_T < 0:
        intp_T = 0.0

    return idx_rho, intp_rho, idx_T, intp_T


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

    idx_mix, intp_mix, A1_f_mat = _mix_weights(A1_mix)

    # 3D interpolation (linear with mix, log(rho), log(T)) to find Z(mix, rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.
    idx_rho, intp_rho = _find_index_and_interp(np.log(rho), A1_log_rho_mixed_HHe_rock)
    idx_T, intp_T = _find_index_and_interp(np.log(T), A1_log_T_mixed_HHe_rock)

    return np.exp(
        _bilinear_log_Z(
            Z_choice, idx_mix, intp_mix, A1_f_mat, idx_rho, intp_rho, idx_T, intp_T
        )
    )


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

    idx_mix, intp_mix, A1_f_mat = _mix_weights(A1_mix)
    idx_rho, intp_rho, idx_T, intp_T = _idx_intp_T_rho_Y(
        rho, Y, idx_mix, intp_mix, A1_f_mat, Y_choice
    )

    A1_log_T = A1_log_T_mixed_HHe_rock
    return np.exp((1 - intp_T) * A1_log_T[idx_T] + intp_T * A1_log_T[idx_T + 1])


@njit
def rho_X_T(X, T, A1_mix, X_choice):
    """Compute the density from the temperature and another equation of state
    parameter, for mixed EoS with (one or) multiple heavy mixed components.

    ## Not yet fully tested. Requires X to increase with density at fixed T.

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

    idx_mix, intp_mix, A1_f_mat = _mix_weights(A1_mix)
    A3_X = _A3_Z_mixed(X_choice)

    # Temperature
    idx_T, intp_T = _find_index_and_interp(np.log(T), A1_log_T_mixed_HHe_rock)

    # X(rho) at this temperature
    A1_log_X_rho = (1 - intp_T) * _log_Z_mixed(
        A3_X, X_choice, idx_mix, intp_mix, A1_f_mat, -1, idx_T
    ) + intp_T * _log_Z_mixed(
        A3_X, X_choice, idx_mix, intp_mix, A1_f_mat, -1, idx_T + 1
    )

    # Density
    idx_rho, intp_rho = _find_index_and_interp(np.log(X), A1_log_X_rho)

    A1_log_rho = A1_log_rho_mixed_HHe_rock
    return np.exp(
        (1 - intp_rho) * A1_log_rho[idx_rho] + intp_rho * A1_log_rho[idx_rho + 1]
    )


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
            c       Sound speed.
            T       Temperature.

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

    idx_mix, intp_mix, A1_f_mat = _mix_weights(A1_mix)
    idx_rho, intp_rho, idx_T, intp_T = _idx_intp_T_rho_Y(
        rho, Y, idx_mix, intp_mix, A1_f_mat, Y_choice
    )

    # Interpolate Z at the same (rho, T)
    return np.exp(
        _bilinear_log_Z(
            Z_choice, idx_mix, intp_mix, A1_f_mat, idx_rho, intp_rho, idx_T, intp_T
        )
    )


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
