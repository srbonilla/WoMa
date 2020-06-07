""" 
WoMa ideal gas equations of state
"""

from numba import njit
import glob_vars as gv


@njit
def idg_gamma(mat_id):
    """ Return the adiabatic index gamma for an ideal gas. """
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
    """ Compute the pressure for the ideal gas EoS.

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
def idg_C_V(mat_id):
    """ Returns specific heat capacity for a given material id (SI)

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    C_V : float
        Specific heat capacity (SI).
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
    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        return idg_C_V(mat_id) * T
    else:
        raise ValueError("Invalid material ID")
