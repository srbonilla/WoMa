""" 
WoMa equations of state (EoS)
"""

from numba import njit
import numpy as np
import matplotlib.pyplot as plt

from woma.misc import glob_vars as gv
from woma.eos import tillotson, sesame, idg, hm80
from woma.eos.T_rho import T_rho


@njit
def P_u_rho(u, rho, mat_id):
    """Compute the pressure from the specific internal energy
    and density, for any EoS.

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
    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        P = idg.P_u_rho(u, rho, mat_id)
    elif mat_type == gv.type_Til:
        P = tillotson.P_u_rho(u, rho, mat_id)
    elif mat_type == gv.type_HM80:
        P = hm80.P_u_rho(u, rho, mat_id)
        if np.isnan(P):
            P = 0.0
    elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        P = sesame.P_u_rho(u, rho, mat_id)
        if np.isnan(P):
            P = 0.0
    else:
        raise ValueError("Invalid material ID")
    return P


@njit
def u_rho_T(rho, T, mat_id):
    """Compute the specific internal energy from the density and temperature, for any EoS.

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
        u = idg.u_rho_T(rho, T, mat_id)
    elif mat_type == gv.type_Til:
        u = tillotson.u_rho_T(rho, T, mat_id)
    elif mat_type == gv.type_HM80:
        u = hm80.u_rho_T(rho, T, mat_id)
    elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        u = sesame.u_rho_T(rho, T, mat_id)
    else:
        raise ValueError("Invalid material ID")
    return u


@njit
def s_rho_T(rho, T, mat_id):
    """Compute the specific entropy from the density and temperature, for any EoS.

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
    mat_type = mat_id // gv.type_factor
    if mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        s = sesame.s_rho_T(rho, T, mat_id)
    else:
        raise ValueError("Entropy not implemented for this material type.")
    return s


@njit
def find_rho(P_des, mat_id, T_rho_type, T_rho_args, rho_min, rho_max):
    """Find the density that satisfies P(u(rho), rho) = P_des, for any EoS.

    Parameters
    ----------
    P_des : float
        The required pressure (Pa).

    mat_id : int
        Material id.

    T_rho_type : int
        Relation between T and rho to be used.

    T_rho_args : [?]
        Extra arguments to determine the relation.

    rho_min : float
        Lower bound for where to look the root (kg m^-3).

    rho_max : float
        Upper bound for where to look the root (kg m^-3).

    Returns
    -------
    rho_mid : float
        The density which satisfies P(u(rho), rho) = P_des (kg m^-3).
    """

    assert rho_min > 0
    assert P_des >= 0
    assert rho_min < rho_max

    tolerance = 1e-5

    T_min = T_rho(rho_min, T_rho_type, T_rho_args, mat_id)
    u_min = u_rho_T(rho_min, T_min, mat_id)
    P_min = P_u_rho(u_min, rho_min, mat_id)
    T_max = T_rho(rho_max, T_rho_type, T_rho_args, mat_id)
    u_max = u_rho_T(rho_max, T_max, mat_id)
    P_max = P_u_rho(u_max, rho_max, mat_id)
    rho_mid = (rho_min + rho_max) / 2.0
    T_mid = T_rho(rho_mid, T_rho_type, T_rho_args, mat_id)
    u_mid = u_rho_T(rho_mid, T_mid, mat_id)
    P_mid = P_u_rho(u_mid, rho_mid, mat_id)
    rho_aux = rho_min + 1e-6
    T_aux = T_rho(rho_aux, T_rho_type, T_rho_args, mat_id)
    u_aux = u_rho_T(rho_aux, T_aux, mat_id)
    P_aux = P_u_rho(u_aux, rho_aux, mat_id)

    if (P_min < P_des < P_max) or (P_min > P_des > P_max):
        max_counter = 200
        counter = 0
        while np.abs(rho_max - rho_min) > tolerance and counter < max_counter:
            T_min = T_rho(rho_min, T_rho_type, T_rho_args, mat_id)
            u_min = u_rho_T(rho_min, T_min, mat_id)
            P_min = P_u_rho(u_min, rho_min, mat_id)
            T_max = T_rho(rho_max, T_rho_type, T_rho_args, mat_id)
            u_max = u_rho_T(rho_max, T_max, mat_id)
            P_max = P_u_rho(u_max, rho_max, mat_id)
            T_mid = T_rho(rho_mid, T_rho_type, T_rho_args, mat_id)
            u_mid = u_rho_T(rho_mid, T_mid, mat_id)
            P_mid = P_u_rho(u_mid, rho_mid, mat_id)

            # if np.isnan(P_min): P_min = 0.

            f0 = P_des - P_min
            f2 = P_des - P_mid

            if f0 * f2 > 0:
                rho_min = rho_mid
            else:
                rho_max = rho_mid

            rho_mid = (rho_min + rho_max) / 2.0
            counter += 1

        return rho_mid

    elif P_min == P_des == P_aux != P_max and P_min < P_max:
        while np.abs(rho_max - rho_min) > tolerance:
            rho_mid = (rho_min + rho_max) / 2.0
            T_min = T_rho(rho_min, T_rho_type, T_rho_args, mat_id)
            u_min = u_rho_T(rho_min, T_min, mat_id)
            P_min = P_u_rho(u_min, rho_min, mat_id)
            T_max = T_rho(rho_max, T_rho_type, T_rho_args, mat_id)
            u_max = u_rho_T(rho_max, T_max, mat_id)
            P_max = P_u_rho(u_max, rho_max, mat_id)
            rho_mid = (rho_min + rho_max) / 2.0
            T_mid = T_rho(rho_mid, T_rho_type, T_rho_args, mat_id)
            u_mid = u_rho_T(rho_mid, T_mid, mat_id)
            P_mid = P_u_rho(u_mid, rho_mid, mat_id)

            if P_mid == P_des:
                rho_min = rho_mid
            else:
                rho_max = rho_mid

            rho_mid = rho_mid = (rho_min + rho_max) / 2.0

        return rho_mid

    elif P_des < P_min < P_max:
        return find_rho(P_des, mat_id, T_rho_type, T_rho_args, rho_min/2, rho_max)
    elif P_des > P_max > P_min:
        return find_rho(P_des, mat_id, T_rho_type, T_rho_args, rho_min, 2*rho_max)
    elif P_des > P_min > P_max:
        return rho_min
    elif P_des < P_max < P_min:
        return rho_max
    else:
        e = "Critical error in find_rho."
        raise ValueError(e)


@njit
def P_T_rho(T, rho, mat_id):
    """Compute the pressure from the temperature and density, for any EoS.

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
    P : float
        Pressure (Pa).
    """
    u = u_rho_T(rho, T, mat_id)
    P = P_u_rho(u, rho, mat_id)
    return P


@njit
def rho_P_T(P, T, mat_id):
    """Compute the density from the pressure and temperature, for any EoS.

    Parameters
    ----------
    P : float
        Pressure (Pa).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Returns
    -------
    rho : float
        Density (kg m^-3).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        assert T > 0
        rho_min = 1e-10
        rho_max = 1e5
    elif mat_type == gv.type_Til:
        rho_min = 1e-7
        rho_max = 1e6
    elif mat_type == gv.type_HM80:
        assert T > 0
        if mat_id == gv.id_HM80_HHe:
            rho_min = 1e-1
            rho_max = 1e5
        elif mat_id == gv.id_HM80_ice:
            rho_min = 1e0
            rho_max = 1e5
        elif mat_id == gv.id_HM80_rock:
            rho_min = 1e0
            rho_max = 40000
    elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        assert T > 0
        assert P > 0

        rho_min = 1e-9
        rho_max = 1e5
    else:
        raise ValueError("Invalid material ID")

    return find_rho(P, mat_id, 1, [float(T), 0.0], rho_min, rho_max)


# Visualize EoS
def plot_EoS_P_rho_fixed_T(
    mat_id_1, mat_id_2, T, P_min=0.1, P_max=1e11, rho_min=100, rho_max=15000
):
    """Plot the EoS pressure as a function of density for various temperatures.

    Parameters
    ----------
    mat_id_1 : int
        Material id for the first material.

    mat_id_2 : int
        Material id for the second material.

    T : float
        Fixed temperature (K).

    P_min : float
        Minimum pressure (Pa) to consider.

    P_max : float
        Maximum pressure (Pa) to consider.

    rho_min : float
        Minimum density (kg m^-3) to consider.

    rho_min : float
        Maximum density (kg m^-3) to consider.
    """

    rho = np.linspace(rho_min, rho_max, 1000)
    P_1 = np.zeros_like(rho)
    P_2 = np.zeros_like(rho)
    for i, rho_i in enumerate(rho):
        P_1[i] = P_T_rho(T, rho_i, mat_id_1)
        P_2[i] = P_T_rho(T, rho_i, mat_id_2)

    plt.figure()
    plt.scatter(rho, P_1, label=str(mat_id_1))
    plt.scatter(rho, P_2, label=str(mat_id_2))
    plt.legend(title="Material")
    plt.xlabel(r"$\rho$ [kg m$^{-3}$]")
    plt.ylabel(r"$P$ [Pa]")
    plt.show()
