""" 
WoMa ideal gas equations of state
"""

from numba import njit

from woma.misc import glob_vars as gv


@njit
def idg_gamma(mat_id):
    """Return the adiabatic index gamma for an ideal gas.

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    gamma : float
        Adiabatic index.
    """
    if mat_id == gv.id_idg_HHe:
        return 1.4
    elif mat_id == gv.id_idg_N2:
        return 1.4
    elif mat_id == gv.id_idg_CO2:
        return 1.29
    else:
        raise ValueError("Invalid material ID")


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
    # Adiabatic constant
    gamma = idg_gamma(mat_id)

    P = (gamma - 1) * u * rho

    return P


@njit
def C_V_idg(mat_id):
    """Return the specific heat capacity.

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    C_V : float
        Specific heat capacity (J kg^-1 K^-1).
    """
    if mat_id == gv.id_idg_HHe:
        return 9093.98
    elif mat_id == gv.id_idg_N2:
        return 742.36
    elif mat_id == gv.id_idg_CO2:
        return 661.38
    else:
        raise ValueError("Invalid material ID")


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
    if mat_type == gv.type_idg:
        return C_V_idg(mat_id) * T
    else:
        raise ValueError("Invalid material ID")
