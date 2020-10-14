""" 
WoMa SESAME and ANEOS (in SESAME-style tables) equations of state
"""

import numpy as np
from numba import njit

from woma.misc import glob_vars as gv
from woma.misc import utils as ut


@njit
def find_index_and_interp(x, A1_x):
    """Return the index and interpolation factor of a value in an array.

    Allows x outside A1_x. If so then intp will be < 0 or > 1.

    Parameters
    ----------
    x : float
        The value to find.

    A1_x : [float]
        The array to search.

    Returns
    -------
    idx : int
        The index of the last array element smaller than the value.

        0               If x is below A1_x.
        len(A1_x) - 2   If x is above A1_x.

    intp : float
        The interpolation factor for how far the values is from the
        indexed array value to the next.

        < 0     If x is below A1_x.
        > 1     If x is above A1_x.
    """
    idx = np.searchsorted(A1_x, x) - 1
    # Return error values if outside the array
    if idx == -1:
        idx = 0
    elif idx >= len(A1_x) - 1:
        idx = len(A1_x) - 2

    # Check for duplicate elements
    if A1_x[idx + 1] != A1_x[idx]:
        intp = (x - A1_x[idx]) / (A1_x[idx + 1] - A1_x[idx])
    else:
        intp = 1.0

    return np.array([idx, intp])


def load_table_SESAME(Fp_table):
    """Load and return the table file data.

    # header (six lines)
    date
    num_rho  num_T
    A1_rho
    A1_T
    A2_u[0, 0]              A2_P[0, 0]      A2_c[0, 0]      A2_s[0, 0]
    A2_u[1, 0]              ...
    ...                     ...
    A2_u[num_rho, 0]        ...
    A2_u[0, 1]              ...
    ...                     ...
    A2_u[num_rho, 1]        ...
    ...                     ...
    A2_u[num_rho, num_u]    ...

    Parameters
    ----------
    Fp_table (str)
        The table file path.

    Returns
    -------
    A2_u, A2_P, A2_s : [[float]]
        2D table arrays of sp. int. energy (J kg^-1), pressure (Pa), and sp.
        entropy (J kg^-1 K^-1).

    A1_log_rho, A1_log_T : [float]
        1D arrays of natural logs of density (kg m^-3) and temperature (K).

    A2_log_u : [[float]]
        2D table array of natural logs of sp. int. energy (J kg^-1).
    """
    # Load
    Fp_table = ut.check_end(Fp_table, ".txt")
    with open(Fp_table) as f:
        # Skip the header
        for i in range(7):
            f.readline()

        num_rho, num_T = np.array(f.readline().split(), dtype=int)
        A2_u = np.empty((num_rho, num_T))
        A2_P = np.empty((num_rho, num_T))
        A2_c = np.empty((num_rho, num_T))
        A2_s = np.empty((num_rho, num_T))

        A1_rho = np.array(f.readline().split(), dtype=float)
        A1_T = np.array(f.readline().split(), dtype=float)

        for i_T in range(num_T):
            for i_rho in range(num_rho):
                (
                    A2_u[i_rho, i_T],
                    A2_P[i_rho, i_T],
                    A2_c[i_rho, i_T],
                    A2_s[i_rho, i_T],
                ) = np.array(f.readline().split(), dtype=float)

    # change values equal to 0
    small = 1e-10
    A1_rho[A1_rho <= 0] = small
    A1_T[A1_T <= 0] = small
    A2_u[A2_u <= 0] = small

    return A2_u, A2_P, A2_s, np.log(A1_rho), np.log(A1_T), np.log(A2_u)


# Load SESAME tables as global variables for numba
(
    A2_u_SESAME_iron,
    A2_P_SESAME_iron,
    A2_s_SESAME_iron,
    A1_log_rho_SESAME_iron,
    A1_log_T_SESAME_iron,
    A2_log_u_SESAME_iron,
) = load_table_SESAME(gv.Fp_SESAME_iron)
(
    A2_u_SESAME_basalt,
    A2_P_SESAME_basalt,
    A2_s_SESAME_basalt,
    A1_log_rho_SESAME_basalt,
    A1_log_T_SESAME_basalt,
    A2_log_u_SESAME_basalt,
) = load_table_SESAME(gv.Fp_SESAME_basalt)
(
    A2_u_SESAME_water,
    A2_P_SESAME_water,
    A2_s_SESAME_water,
    A1_log_rho_SESAME_water,
    A1_log_T_SESAME_water,
    A2_log_u_SESAME_water,
) = load_table_SESAME(gv.Fp_SESAME_water)
(
    A2_u_SS08_water,
    A2_P_SS08_water,
    A2_s_SS08_water,
    A1_log_rho_SS08_water,
    A1_log_T_SS08_water,
    A2_log_u_SS08_water,
) = load_table_SESAME(gv.Fp_SS08_water)

# Load ANEOS as SESAME-style tables
(
    A2_u_ANEOS_forsterite,
    A2_P_ANEOS_forsterite,
    A2_s_ANEOS_forsterite,
    A1_log_rho_ANEOS_forsterite,
    A1_log_T_ANEOS_forsterite,
    A2_log_u_ANEOS_forsterite,
) = load_table_SESAME(gv.Fp_ANEOS_forsterite)

(
    A2_u_ANEOS_iron,
    A2_P_ANEOS_iron,
    A2_s_ANEOS_iron,
    A1_log_rho_ANEOS_iron,
    A1_log_T_ANEOS_iron,
    A2_log_u_ANEOS_iron,
) = load_table_SESAME(gv.Fp_ANEOS_iron)

(
    A2_u_ANEOS_Fe85Si15,
    A2_P_ANEOS_Fe85Si15,
    A2_s_ANEOS_Fe85Si15,
    A1_log_rho_ANEOS_Fe85Si15,
    A1_log_T_ANEOS_Fe85Si15,
    A2_log_u_ANEOS_Fe85Si15,
) = load_table_SESAME(gv.Fp_ANEOS_Fe85Si15)

# Load AQUA as SESAME-style tables
(
    A2_u_AQUA,
    A2_P_AQUA,
    A2_s_AQUA,
    A1_log_rho_AQUA,
    A1_log_T_AQUA,
    A2_log_u_AQUA,
) = load_table_SESAME(gv.Fp_AQUA)

# Load CMS19 as SESAME-style tables
(
    A2_u_CMS19_H,
    A2_P_CMS19_H,
    A2_s_CMS19_H,
    A1_log_rho_CMS19_H,
    A1_log_T_CMS19_H,
    A2_log_u_CMS19_H,
) = load_table_SESAME(gv.Fp_CMS19_H)

(
    A2_u_CMS19_He,
    A2_P_CMS19_He,
    A2_s_CMS19_He,
    A1_log_rho_CMS19_He,
    A1_log_T_CMS19_He,
    A2_log_u_CMS19_He,
) = load_table_SESAME(gv.Fp_CMS19_He)

(
    A2_u_CMS19_HHe,
    A2_P_CMS19_HHe,
    A2_s_CMS19_HHe,
    A1_log_rho_CMS19_HHe,
    A1_log_T_CMS19_HHe,
    A2_log_u_CMS19_HHe,
) = load_table_SESAME(gv.Fp_CMS19_HHe)



@njit
def P_u_rho(u, rho, mat_id):
    """Compute the pressure from the internal energy and density.

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    P : float
        Pressure (Pa).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A2_log_u_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A2_log_u_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_water,
            A1_log_rho_SESAME_water,
            A2_log_u_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SS08_water,
            A1_log_rho_SS08_water,
            A2_log_u_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A2_log_u_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A2_log_u_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A2_log_u_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_AQUA,
            A1_log_rho_AQUA,
            A2_log_u_AQUA,
        )
    elif mat_id == gv.id_CMS19_H:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_CMS19_H,
            A1_log_rho_CMS19_H,
            A2_log_u_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_CMS19_He,
            A1_log_rho_CMS19_He,
            A2_log_u_CMS19_He,
        )
    elif mat_id == gv.id_CMS19_HHe:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_CMS19_HHe,
            A1_log_rho_CMS19_HHe,
            A2_log_u_CMS19_HHe,
        )
    else:
        raise ValueError("Invalid material ID")

    # Ignore the first elements of rho = 0, T = 0
    A2_P = A2_P[1:, 1:]
    A2_log_u = A2_log_u[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_u = np.log(u)

    # 2D interpolation (bilinear with log(rho), log(u)) to find P(rho, u).
    # If rho and/or u are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # u (in this and the next density slice of the 2D u array)
    idx_u_1_intp_u_1 = find_index_and_interp(log_u, A2_log_u[idx_rho])
    idx_u_1 = int(idx_u_1_intp_u_1[0])
    intp_u_1 = idx_u_1_intp_u_1[1]
    idx_u_2_intp_u_2 = find_index_and_interp(log_u, A2_log_u[idx_rho + 1])
    idx_u_2 = int(idx_u_2_intp_u_2[0])
    intp_u_2 = idx_u_2_intp_u_2[1]

    P_1 = A2_P[idx_rho, idx_u_1]
    P_2 = A2_P[idx_rho, idx_u_1 + 1]
    P_3 = A2_P[idx_rho + 1, idx_u_2]
    P_4 = A2_P[idx_rho + 1, idx_u_2 + 1]

    # If below the minimum u at this rho then just use the lowest table values
    if idx_rho >= 0 and (intp_u_1 < 0 or intp_u_2 < 0 or P_1 > P_2 or P_3 > P_4):
        intp_u_1 = 0
        intp_u_2 = 0

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([P_1, P_2, P_3, P_4]) < 0)
    if num_non_pos > 2:
        return 0.0

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_u_1 < 0 or intp_u_2 < 0:
            return 0.0
        else:
            # P_tiny  = np.amin(A2_P[A2_P > 0]) * 1e-3
            P_tiny = np.amin(np.abs(A2_P)) * 1e-3
            if P_1 <= 0:
                P_1 = P_tiny
            if P_2 <= 0:
                P_2 = P_tiny
            if P_3 <= 0:
                P_3 = P_tiny
            if P_4 <= 0:
                P_4 = P_tiny

    # Interpolate with the log values
    P_1 = np.log(P_1)
    P_2 = np.log(P_2)
    P_3 = np.log(P_3)
    P_4 = np.log(P_4)

    # P(rho, u)
    P = (1 - intp_rho) * ((1 - intp_u_1) * P_1 + intp_u_1 * P_2) + intp_rho * (
        (1 - intp_u_2) * P_3 + intp_u_2 * P_4
    )

    # Convert back from log
    return np.exp(P)


@njit
def s_u_rho(u, rho, mat_id):
    """Compute the specific entropy from the internal energy and density.

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    s : float
        Specific entropy (J K^-1 kg^-1).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A2_log_u_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A2_log_u_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_SESAME_water,
            A1_log_rho_SESAME_water,
            A2_log_u_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_SS08_water,
            A1_log_rho_SS08_water,
            A2_log_u_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A2_log_u_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A2_log_u_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A2_log_u_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_AQUA,
            A1_log_rho_AQUA,
            A2_log_u_AQUA,
        )
    elif mat_id == gv.id_CMS19_H:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_CMS19_H,
            A1_log_rho_CMS19_H,
            A2_log_u_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_CMS19_He,
            A1_log_rho_CMS19_He,
            A2_log_u_CMS19_He,
        )
    elif mat_id == gv.id_CMS19_HHe:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_CMS19_HHe,
            A1_log_rho_CMS19_HHe,
            A2_log_u_CMS19_HHe,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check this material has entropy values
    if (A2_s == 0).all():
        raise ValueError("No entropy values for this material")

    # Ignore the first elements of rho = 0, T = 0
    A2_s = A2_s[1:, 1:]
    A2_log_u = A2_log_u[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_u = np.log(u)

    # 2D interpolation (bilinear with log(rho), log(u)) to find s(rho, u).
    # If rho and/or u are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # u (in this and the next density slice of the 2D u array)
    idx_u_1_intp_u_1 = find_index_and_interp(log_u, A2_log_u[idx_rho])
    idx_u_1 = int(idx_u_1_intp_u_1[0])
    intp_u_1 = idx_u_1_intp_u_1[1]
    idx_u_2_intp_u_2 = find_index_and_interp(log_u, A2_log_u[idx_rho + 1])
    idx_u_2 = int(idx_u_2_intp_u_2[0])
    intp_u_2 = idx_u_2_intp_u_2[1]

    s_1 = A2_s[idx_rho, idx_u_1]
    s_2 = A2_s[idx_rho, idx_u_1 + 1]
    s_3 = A2_s[idx_rho + 1, idx_u_2]
    s_4 = A2_s[idx_rho + 1, idx_u_2 + 1]

    # If below the minimum u at this rho then just use the lowest table values
    if idx_rho >= 0 and (intp_u_1 < 0 or intp_u_2 < 0 or s_1 > s_2 or s_3 > s_4):
        intp_u_1 = 0
        intp_u_2 = 0

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([s_1, s_2, s_3, s_4]) < 0)
    if num_non_pos > 2:
        return 0.0

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_u_1 < 0 or intp_u_2 < 0:
            return 0.0
        else:
            s_tiny = np.amin(np.abs(A2_s)) * 1e-3
            if s_1 <= 0:
                s_1 = s_tiny
            if s_2 <= 0:
                s_2 = s_tiny
            if s_3 <= 0:
                s_3 = s_tiny
            if s_4 <= 0:
                s_4 = s_tiny

    # Interpolate with the log values
    s_1 = np.log(s_1)
    s_2 = np.log(s_2)
    s_3 = np.log(s_3)
    s_4 = np.log(s_4)

    # s(rho, u)
    s = (1 - intp_rho) * ((1 - intp_u_1) * s_1 + intp_u_1 * s_2) + intp_rho * (
        (1 - intp_u_2) * s_3 + intp_u_2 * s_4
    )

    # Convert back from log
    return np.exp(s)


@njit
def u_rho_T(rho, T, mat_id):
    """Compute the internal energy from the density and temperature.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Returns
    -------
    u : float
        Specific internal energy (J kg^-1).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A1_log_T_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A1_log_T_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_water,
            A1_log_rho_SESAME_water,
            A1_log_T_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SS08_water,
            A1_log_rho_SS08_water,
            A1_log_T_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A1_log_T_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A1_log_T_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A1_log_T_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_AQUA,
            A1_log_rho_AQUA,
            A1_log_T_AQUA,
        )
    elif mat_id == gv.id_CMS19_H:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_CMS19_H,
            A1_log_rho_CMS19_H,
            A1_log_T_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_CMS19_He,
            A1_log_rho_CMS19_He,
            A1_log_T_CMS19_He,
        )
    elif mat_id == gv.id_CMS19_HHe:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_CMS19_HHe,
            A1_log_rho_CMS19_HHe,
            A1_log_T_CMS19_HHe,
        )
    else:
        raise ValueError("Invalid material ID")

    # Ignore the first elements of rho = 0, T = 0
    A2_u = A2_u[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_T = np.log(T * 1)  # why is numba so weird?

    # 2D interpolation (bilinear with log(rho), log(T)) to find u(rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Temperature
    idx_T_intp_T = find_index_and_interp(log_T, A1_log_T[1:])
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    u_1 = A2_u[idx_rho, idx_T]
    u_2 = A2_u[idx_rho, idx_T + 1]
    u_3 = A2_u[idx_rho + 1, idx_T]
    u_4 = A2_u[idx_rho + 1, idx_T + 1]

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([u_1, u_2, u_3, u_4]) < 0)
    if num_non_pos > 2:
        return 0.0

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_T < 0:
            return 0.0
        else:
            u_tiny = np.amin(np.abs(A2_u)) * 1e-3
            if u_1 <= 0:
                u_1 = u_tiny
            if u_2 <= 0:
                u_2 = u_tiny
            if u_3 <= 0:
                u_3 = u_tiny
            if u_4 <= 0:
                u_4 = u_tiny

    # Interpolate with the log values
    u_1 = np.log(u_1)
    u_2 = np.log(u_2)
    u_3 = np.log(u_3)
    u_4 = np.log(u_4)

    # u(rho, T)
    u = (1 - intp_rho) * ((1 - intp_T) * u_1 + intp_T * u_2) + intp_rho * (
        (1 - intp_T) * u_3 + intp_T * u_4
    )

    # Convert back from log
    return np.exp(u)


@njit
def s_rho_T(rho, T, mat_id):
    """Compute the specific entropy from the density and temperature.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Returns
    -------
    s : float
        Specific entropy (J kg^-1 K^-1).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A1_log_T_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A1_log_T_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_water,
            A1_log_rho_SESAME_water,
            A1_log_T_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SS08_water,
            A1_log_rho_SS08_water,
            A1_log_T_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A1_log_T_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A1_log_T_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A1_log_T_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_AQUA,
            A1_log_rho_AQUA,
            A1_log_T_AQUA,
        )
    elif mat_id == gv.id_CMS19_H:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_CMS19_H,
            A1_log_rho_CMS19_H,
            A1_log_T_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_CMS19_He,
            A1_log_rho_CMS19_He,
            A1_log_T_CMS19_He,
        )
    elif mat_id == gv.id_CMS19_HHe:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_CMS19_HHe,
            A1_log_rho_CMS19_HHe,
            A1_log_T_CMS19_HHe,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check this material has entropy values
    if (A2_s == 0).all():
        raise ValueError("No entropy values for this material")

    # Ignore the first elements of rho = 0, T = 0
    A2_s = A2_s[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_T = np.log(T)

    # 2D interpolation (bilinear with log(rho), log(T)) to find s(rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Temperature
    idx_T_intp_T = find_index_and_interp(log_T, A1_log_T[1:])
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    s_1 = A2_s[idx_rho, idx_T]
    s_2 = A2_s[idx_rho, idx_T + 1]
    s_3 = A2_s[idx_rho + 1, idx_T]
    s_4 = A2_s[idx_rho + 1, idx_T + 1]

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([s_1, s_2, s_3, s_4]) < 0)
    if num_non_pos > 2:
        return 0.0

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_T < 0:
            return 0.0
        else:
            s_tiny = np.amin(np.abs(A2_s)) * 1e-3
            if s_1 <= 0:
                s_1 = s_tiny
            if s_2 <= 0:
                s_2 = s_tiny
            if s_3 <= 0:
                s_3 = s_tiny
            if s_4 <= 0:
                s_4 = s_tiny

    # s(rho, T)
    s = (1 - intp_rho) * ((1 - intp_T) * s_1 + intp_T * s_2) + intp_rho * (
        (1 - intp_T) * s_3 + intp_T * s_4
    )

    return s


@njit
def T_rho_s(rho, s, mat_id):
    """Compute the temperature from the density and specific entropy.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    s : float
        Specific entropy (J kg^-1 K^-1).

    mat_id : int
        Material id.

    Returns
    -------
    T : float
        Temperature (K).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A2_s_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A2_s_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_SESAME_water,
            A1_log_rho_SESAME_water,
            A2_s_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_SS08_water,
            A1_log_rho_SS08_water,
            A2_s_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A2_s_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A2_s_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A2_s_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_AQUA,
            A1_log_rho_AQUA,
            A2_s_AQUA,
        )
    elif mat_id == gv.id_CMS19_H:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_CMS19_H,
            A1_log_rho_CMS19_H,
            A2_s_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_CMS19_He,
            A1_log_rho_CMS19_He,
            A2_s_CMS19_He,
        )
    elif mat_id == gv.id_CMS19_HHe:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_CMS19_HHe,
            A1_log_rho_CMS19_HHe,
            A2_s_CMS19_HHe,
        )
    else:
        raise ValueError("Invalid material ID")

    # Convert to log
    log_rho = np.log(rho)

    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho)
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # s (in this and the next density slice of the 2D s array)
    idx_s_1_intp_s_1 = find_index_and_interp(s, A2_s[idx_rho])
    idx_s_1 = int(idx_s_1_intp_s_1[0])
    intp_s_1 = idx_s_1_intp_s_1[1]
    idx_s_2_intp_s_2 = find_index_and_interp(s, A2_s[idx_rho + 1])
    idx_s_2 = int(idx_s_2_intp_s_2[0])
    intp_s_2 = idx_s_2_intp_s_2[1]

    # Normal interpolation
    log_T = (1 - intp_rho) * (
        (1 - intp_s_1) * A1_log_T[idx_s_1] + intp_s_1 * A1_log_T[idx_s_1 + 1]
    ) + intp_rho * (
        (1 - intp_s_2) * A1_log_T[idx_s_2] + intp_s_2 * A1_log_T[idx_s_2 + 1]
    )

    # Convert back from log
    T = np.exp(log_T)
    if T < 0:
        T = 0

    return T
