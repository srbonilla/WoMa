""" 
WoMa Hubbard & MacFarlane (1980) equations of state
"""

import numpy as np
from numba import njit
import os

from woma.misc import glob_vars as gv


def load_u_cold_array(mat_id):
    """Load precomputed values of cold internal energy.

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    u_cold_array : [float]
        Precomputed values of cold internal energy from function
        _create_u_cold_array() (J kg^-1).
    """
    if mat_id == gv.id_HM80_ice:
        u_cold_array = np.load(gv.Fp_u_cold_HM80_ice)
    elif mat_id == gv.id_HM80_rock:
        u_cold_array = np.load(gv.Fp_u_cold_HM80_rock)
    else:
        raise ValueError("Invalid material ID")

    return u_cold_array


# Set None values for cold internal energy arrays
A1_u_cold_HM80_ice = np.zeros(1)
A1_u_cold_HM80_rock = np.zeros(1)


def load_table_HM80(Fp_table):
    """Load and return the table file data.

    # header (four lines)
    date
    log_rho_min  log_rho_max  num_rho  log_u_min  log_u_max  num_u
    P_0_0   P_0_1   ...     P_0_num_u           # Array of pressures
    P_1_0   ...     ...     P_1_num_u
    ...     ...     ...     ...
    P_num_rho_0     ...     P_num_rho_num_u
    T_0_0   T_0_1   ...     T_0_num_u           # Array of temperatures
    T_1_0   ...     ...     T_1_num_u
    ...     ...     ...     ...
    T_num_rho_0     ...     T_num_rho_num_u

    Parameters
    ----------
    Fp_table : str
        The table file path.

    Returns
    -------
    log_rho_min : float
        Natural log of the minimum density (kg m^-3).

    log_rho_max : float
        Natural log of the maximum density (kg m^-3).

    num_rho : int
        Number of different density values tabulated.

    log_rho_step : float
        Step between consecutive tabulated density values.

    log_u_min : float
        Natural log of the minimum specific internal energy (J kg^-1).

    log_u_max : float
        Natural log of the maximum specific internal energy (J kg^-1).

    num_u : int
        Number of different specific internal energy values tabulated.

    log_u_step : float
        Step between consecutive tabulated specific internal energy values.

    A2_log_P, A2_log_T : [float]
        2D arrays of natural logs of pressure (Pa) and temperature (K).
    """
    # Parameters
    log_rho_min, log_rho_max, num_rho, log_u_min, log_u_max, num_u = np.genfromtxt(
        Fp_table, skip_header=6, max_rows=1
    )

    # Tables
    A2_data = np.loadtxt(Fp_table, skiprows=7)

    num_rho = int(num_rho)
    num_u = int(num_u)
    A2_P = A2_data[:num_rho]
    A2_T = A2_data[num_rho:]

    assert A2_P.shape == (num_rho, num_u)
    assert A2_T.shape == (num_rho, num_u)

    log_rho_step = (log_rho_max - log_rho_min) / (num_rho - 1)
    log_u_step = (log_u_max - log_u_min) / (num_u - 1)

    # change values equal to 0
    small = 1e-10
    A2_P[A2_P == 0] = small
    A2_T[A2_T == 0] = small

    return (
        log_rho_min,
        log_rho_max,
        num_rho,
        log_rho_step,
        log_u_min,
        log_u_max,
        num_u,
        log_u_step,
        np.log(A2_P),
        np.log(A2_T),
    )


# Assume H2-He mass fraction x = 0.75 = 2*n_H2 / (2*n_H2 + 4*n_He) --> ratio:
n_H2_n_He = 2 / (1 / 0.75 - 1)
m_mol_HHe = (2 * n_H2_n_He + 4) / (n_H2_n_He + 1)

# Set default table values
(
    log_rho_min_HM80_HHe,
    log_rho_max_HM80_HHe,
    num_rho_HM80_HHe,
    log_rho_step_HM80_HHe,
    log_u_min_HM80_HHe,
    log_u_max_HM80_HHe,
    num_u_HM80_HHe,
    log_u_step_HM80_HHe,
    A2_log_P_HM80_HHe,
    A2_log_T_HM80_HHe,
) = (
    float(0),
    float(0),
    int(0),
    float(0),
    float(0),
    float(0),
    int(0),
    float(0),
    np.zeros((1, 1)),
    np.zeros((1, 1)),
)
(
    log_rho_min_HM80_ice,
    log_rho_max_HM80_ice,
    num_rho_HM80_ice,
    log_rho_step_HM80_ice,
    log_u_min_HM80_ice,
    log_u_max_HM80_ice,
    num_u_HM80_ice,
    log_u_step_HM80_ice,
    A2_log_P_HM80_ice,
    A2_log_T_HM80_ice,
) = (
    float(0),
    float(0),
    int(0),
    float(0),
    float(0),
    float(0),
    int(0),
    float(0),
    np.zeros((1, 1)),
    np.zeros((1, 1)),
)
(
    log_rho_min_HM80_rock,
    log_rho_max_HM80_rock,
    num_rho_HM80_rock,
    log_rho_step_HM80_rock,
    log_u_min_HM80_rock,
    log_u_max_HM80_rock,
    num_u_HM80_rock,
    log_u_step_HM80_rock,
    A2_log_P_HM80_rock,
    A2_log_T_HM80_rock,
) = (
    float(0),
    float(0),
    int(0),
    float(0),
    float(0),
    float(0),
    int(0),
    float(0),
    np.zeros((1, 1)),
    np.zeros((1, 1)),
)


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
    # Choose the arrays from the global variables
    if mat_id == gv.id_HM80_HHe:
        (log_rho_min, num_rho, log_rho_step, log_u_min, num_u, log_u_step, A2_log_P) = (
            log_rho_min_HM80_HHe,
            num_rho_HM80_HHe,
            log_rho_step_HM80_HHe,
            log_u_min_HM80_HHe,
            num_u_HM80_HHe,
            log_u_step_HM80_HHe,
            A2_log_P_HM80_HHe,
        )
    elif mat_id == gv.id_HM80_ice:
        (log_rho_min, num_rho, log_rho_step, log_u_min, num_u, log_u_step, A2_log_P) = (
            log_rho_min_HM80_ice,
            num_rho_HM80_ice,
            log_rho_step_HM80_ice,
            log_u_min_HM80_ice,
            num_u_HM80_ice,
            log_u_step_HM80_ice,
            A2_log_P_HM80_ice,
        )
    elif mat_id == gv.id_HM80_rock:
        (log_rho_min, num_rho, log_rho_step, log_u_min, num_u, log_u_step, A2_log_P) = (
            log_rho_min_HM80_rock,
            num_rho_HM80_rock,
            log_rho_step_HM80_rock,
            log_u_min_HM80_rock,
            num_u_HM80_rock,
            log_u_step_HM80_rock,
            A2_log_P_HM80_rock,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A2_log_P) == 2:
        raise ValueError(
            "Please load the corresponding HM80 table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Convert to log
    log_rho = np.log(rho)
    log_u = np.log(u)

    # 2D interpolation (bilinear with log(rho), log(u)) to find P(rho, u).
    # If rho and/or u are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    idx_rho = int(np.floor((log_rho - log_rho_min) / log_rho_step))
    idx_u = int(np.floor((log_u - log_u_min) / log_u_step))

    # Check if outside the table
    if idx_rho == -1:
        idx_rho = 0
    elif idx_rho >= num_rho - 1:
        idx_rho = num_rho - 2
    if idx_u == -1:
        idx_u = 0
    elif idx_u >= num_u - 1:
        idx_u = num_u - 2

    intp_rho = (log_rho - log_rho_min - idx_rho * log_rho_step) / log_rho_step
    intp_u = (log_u - log_u_min - idx_u * log_u_step) / log_u_step

    log_P_1 = A2_log_P[idx_rho, idx_u]
    log_P_2 = A2_log_P[idx_rho, idx_u + 1]
    log_P_3 = A2_log_P[idx_rho + 1, idx_u]
    log_P_4 = A2_log_P[idx_rho + 1, idx_u + 1]

    # log_P(rho, u)
    log_P = (1 - intp_rho) * ((1 - intp_u) * log_P_1 + intp_u * log_P_2) + intp_rho * (
        (1 - intp_u) * log_P_3 + intp_u * log_P_4
    )

    # Convert back from log
    return np.exp(log_P)


@njit
def T_u_rho(u, rho, mat_id):
    """Compute the temperature from the internal energy and density.

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
    T : float
        Temperature (K).
    """
    # Unpack the parameters
    if mat_id == gv.id_HM80_HHe:
        (log_rho_min, num_rho, log_rho_step, log_u_min, num_u, log_u_step, A2_log_T) = (
            log_rho_min_HM80_HHe,
            num_rho_HM80_HHe,
            log_rho_step_HM80_HHe,
            log_u_min_HM80_HHe,
            num_u_HM80_HHe,
            log_u_step_HM80_HHe,
            A2_log_T_HM80_HHe,
        )
    elif mat_id == gv.id_HM80_ice:
        (log_rho_min, num_rho, log_rho_step, log_u_min, num_u, log_u_step, A2_log_T) = (
            log_rho_min_HM80_ice,
            num_rho_HM80_ice,
            log_rho_step_HM80_ice,
            log_u_min_HM80_ice,
            num_u_HM80_ice,
            log_u_step_HM80_ice,
            A2_log_T_HM80_ice,
        )
    elif mat_id == gv.id_HM80_rock:
        (log_rho_min, num_rho, log_rho_step, log_u_min, num_u, log_u_step, A2_log_T) = (
            log_rho_min_HM80_rock,
            num_rho_HM80_rock,
            log_rho_step_HM80_rock,
            log_u_min_HM80_rock,
            num_u_HM80_rock,
            log_u_step_HM80_rock,
            A2_log_T_HM80_rock,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A2_log_T) == 2:
        raise ValueError(
            "Please load the corresponding HM80 table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Convert to log
    log_rho = np.log(rho)
    log_u = np.log(u)

    # 2D interpolation (bilinear with log(rho), log(u)) to find P(rho, u).
    # If rho and/or u are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    idx_rho = int(np.floor((log_rho - log_rho_min) / log_rho_step))
    idx_u = int(np.floor((log_u - log_u_min) / log_u_step))

    # Check if outside the table
    if idx_rho == -1:
        idx_rho = 0
    elif idx_rho >= num_rho - 1:
        idx_rho = num_rho - 2
    if idx_u == -1:
        idx_u = 0
    elif idx_u >= num_u - 1:
        idx_u = num_u - 2

    intp_rho = (log_rho - log_rho_min - idx_rho * log_rho_step) / log_rho_step
    intp_u = (log_u - log_u_min - idx_u * log_u_step) / log_u_step

    log_T_1 = A2_log_T[idx_rho, idx_u]
    log_T_2 = A2_log_T[idx_rho, idx_u + 1]
    log_T_3 = A2_log_T[idx_rho + 1, idx_u]
    log_T_4 = A2_log_T[idx_rho + 1, idx_u + 1]

    # log_P(rho, u)
    log_T = (1 - intp_rho) * ((1 - intp_u) * log_T_1 + intp_u * log_T_2) + intp_rho * (
        (1 - intp_u) * log_T_3 + intp_u * log_T_4
    )

    # Convert back from log
    return np.exp(log_T)


@njit
def _rho_0(mat_id):
    """Return the density for which the cold internal energy is zero.

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    rho_0 : float
        Density (kg m^-3).
    """
    if mat_id == gv.id_HM80_HHe:
        return 0.0
    elif mat_id == gv.id_HM80_ice:
        return 947.8
    elif mat_id == gv.id_HM80_rock:
        return 2704.8
    else:
        raise ValueError("Invalid material ID")


@njit
def u_cold(rho, mat_id, N):
    """Compute the cold internal energy from the density.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    N : int
        Number of subdivisions for the numerical integral.

    Returns
    -------
    u_cold : float
        Cold specific internal energy (J kg^-1).
    """
    assert rho >= 0
    if mat_id == gv.id_HM80_HHe:
        return 0.0

    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_HM80:
        rho_0 = _rho_0(mat_id)
        drho = (rho - rho_0) / N
        x = rho_0
        u_cold = 1e-10

        for j in range(N):
            x += drho
            u_cold += P_u_rho(u_cold, x, mat_id) * drho / x ** 2

    else:
        raise ValueError("Invalid material ID")

    return u_cold


@njit
def _create_u_cold_array(mat_id):
    """Compute tabulated values of the cold internal energy.

    Ranges from density = 100 to 100000 kg/m^3

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    u_cold_array : [float]
        Array of cold specific internal energies (J kg^-1).
    """
    N_row = 10000
    u_cold_array = np.zeros((N_row,))
    rho_min = 100
    rho_max = 100000
    N_u_cold = 10000

    rho = rho_min
    drho = (rho_max - rho_min) / (N_row - 1)

    for i in range(N_row):
        u_cold_array[i] = u_cold(rho, mat_id, N_u_cold)
        rho = rho + drho

    return u_cold_array


@njit
def u_cold_tab(rho, mat_id):
    """Compute the cold internal energy using premade tabulated values.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    u_cold : float
        Cold specific internal energy (J kg^-1).
    """

    if mat_id == gv.id_HM80_HHe:
        return 0.0
    elif mat_id == gv.id_HM80_ice:
        u_cold_array = A1_u_cold_HM80_ice
    elif mat_id == gv.id_HM80_rock:
        u_cold_array = A1_u_cold_HM80_rock
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(u_cold_array) == 1:
        raise ValueError(
            "Please load the corresponding HM80 table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    N_row = u_cold_array.shape[0]
    rho_min = 100
    rho_max = 100000

    drho = (rho_max - rho_min) / (N_row - 1)

    a = int(((rho - rho_min) / drho))
    b = a + 1

    if a >= 0 and a < (N_row - 1):
        u_cold = u_cold_array[a]
        u_cold += ((u_cold_array[b] - u_cold_array[a]) / drho) * (
            rho - rho_min - a * drho
        )

    elif rho < rho_min:
        u_cold = u_cold_array[0]
    else:
        u_cold = u_cold_array[int(N_row - 1)]
        u_cold += (
            (u_cold_array[int(N_row - 1)] - u_cold_array[int(N_row) - 2]) / drho
        ) * (rho - rho_max)

    return u_cold


@njit
def C_V_HM80(rho, T, mat_id):
    """Return the specific heat capacity from the density and temperature.

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
    C_V : float
        Specific heat capacity (J kg^-1 K^-1).
    """
    # Convert to cgs for HM80's units
    rho_cgs = rho * 1e-3
    R_gas_cgs = gv.R_gas * 1e7

    if mat_id == gv.id_HM80_HHe:

        A1_c = [2.3638, -4.9842e-5, 1.1788e-8, -3.8101e-4, 2.6182, 0.45053]

        C_V = (
            (
                A1_c[0]
                + A1_c[1] * T
                + A1_c[2] * T ** 2
                + A1_c[3] * T * rho_cgs
                + A1_c[4] * rho_cgs
                + A1_c[5] * rho_cgs ** 2
            )
            * gv.R_gas
            * 1e7
            / m_mol_HHe
        )

    elif mat_id == gv.id_HM80_ice:
        # H20, CH4, NH3
        A1_abun = np.array([0.565, 0.325, 0.11])
        A1_nu = np.array([3, 5, 4])
        f_nu = 2.067
        A1_m_mol = np.array([18, 18, 18])
        m_mol = np.sum(A1_m_mol * A1_abun)
        C_V = np.sum(A1_abun * A1_nu) * f_nu * R_gas_cgs / m_mol

    elif mat_id == gv.id_HM80_rock:
        # SiO, MgO, FeS, FeO
        A1_abun = np.array([0.38, 0.25, 0.25, 0.12])
        A1_nu = np.array([3, 2, 2, 2])
        f_nu = 3
        A1_m_mol = np.array([44, 40, 88, 72])
        m_mol = np.sum(A1_m_mol * A1_abun)
        C_V = np.sum(A1_abun * A1_nu) * f_nu * R_gas_cgs / m_mol

    else:
        raise ValueError("Material not fully implemented yet")

    # Convert back to SI
    return C_V * 1e-4


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
    mat_type = mat_id // gv.type_factor

    if mat_type == gv.type_HM80:
        cv = C_V_HM80(rho, T, mat_id)
        u = u_cold_tab(rho, mat_id) + cv * T

    else:
        raise ValueError("Invalid material ID")

    return u


@njit
def T_rho_HM80_HHe(rho, rho_prv, T_prv):
    """Compute the temperature as a function of density for the H-He atmosphere.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    rho_prv, T_prv : float
        The previous density (kg m^-3) and temperature (K).

    Returns
    -------
    T : float
        Temperature (K).
    """
    # Convert to cgs and x,y for HM80's units
    x = np.log(rho / 5)
    x_prv = np.log(rho_prv / 5)
    y_prv = np.log(T_prv)

    # HM80 parameters
    A1_b = [0.328471, 0.0286529, -0.00139609, -0.0231158, 0.0579055, 0.0454488]

    def dy_dx(x, y):
        return (
            A1_b[0]
            + A1_b[1] * y
            + A1_b[2] * y ** 2
            + A1_b[3] * x * y
            + A1_b[4] * x
            + A1_b[5] * x ** 2
        )

    # Integrate from y_prv(x_prv) to y(x)
    y = y_prv
    for x_tmp in np.linspace(x_prv, x, 100):
        y += dy_dx(x_prv, y) * (x_tmp - x_prv)
        x_prv = x_tmp

    return np.exp(y)


@njit
def P_T_rho(T, rho, mat_id):
    """Compute the pressure from the density and temperature.

    Parameters
    ----------
    T : float
        Temperature (K).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    u : float
        Specific internal energy (J kg^-1).
    """

    mat_type = mat_id // gv.type_factor

    if mat_type == gv.type_HM80:

        u = u_rho_T(rho, T, mat_id)
        P = P_u_rho(u, rho, mat_id)

    else:
        raise ValueError("Invalid material ID")

    return P
