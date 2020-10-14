""" 
WoMa temperature--density relations
"""

from numba import njit
import numpy as np

from woma.misc import glob_vars as gv
from woma.eos import tillotson, sesame, idg, hm80


@njit
def T_rho(rho, T_rho_type_id, T_rho_args, mat_id):
    """Compute the temperature from the density using the chosen relation.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T_rho_type_id : int
        Relation between T and rho to be used. See woma.Planet.

    T_rho_args : [?]
        Extra arguments to determine the relation. See woma.Planet.

    mat_id : int
        The material ID.

    Returns
    -------
    T : float
        Temperature (K)
    """
    mat_type = mat_id // gv.type_factor

    # Power law, T = K*rho**alpha, T_rho_args = [K, alpha]
    if T_rho_type_id == gv.type_rho_pow:
        K = T_rho_args[0]
        alpha = T_rho_args[1]
        T = K * np.power(rho, alpha)
        return T

    # Adiabatic, T_rho_args = [s_adb,], [rho_prv, T_prv], or [T rho^(1-gamma)]
    elif T_rho_type_id == gv.type_adb:
        if mat_type == gv.type_idg:
            # T rho^(1-gamma) = constant
            gamma = idg.idg_gamma(mat_id)
            return T_rho_args[0] * rho ** (gamma - 1)
        elif mat_id == gv.id_HM80_HHe:
            return hm80.T_rho_HM80_HHe(rho, T_rho_args[0], T_rho_args[1])
        elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
            return sesame.T_rho_s(rho, T_rho_args[0], mat_id)
        else:
            raise ValueError("Adiabatic not implemented for this material type")

    # Fixed entropy, T_rho_args = [s,]
    elif T_rho_type_id == gv.type_ent:
        if mat_type in [gv.type_SESAME, gv.type_ANEOS]:
            return sesame.T_rho_s(rho, T_rho_args[0], mat_id)
        else:
            raise ValueError("Entropy not implemented for this material type")
    else:
        raise ValueError("T_rho_type_id not implemented")


@njit
def set_T_rho_args(T, rho, T_rho_type_id, T_rho_args, mat_id):
    """Set any parameters for the T-rho relation.

    Parameters
    ----------
    T : float
        Temperature (K)

    rho : float
        Density (kg m^-3)

    T_rho_type_id : int
        T-rho relation ID. See woma.Planet.

    T_rho_args : [float]
        T-rho parameters (for a single layer). See woma.Planet.

    mat_id : int
        Material ID

    Returns
    -------
    T_rho_args : [float]
        T-rho parameters (for a single layer). See woma.Planet.
    """
    mat_type = mat_id // gv.type_factor

    # Power: T = K*rho**alpha, T_rho_args = [K, alpha]
    if T_rho_type_id == gv.type_rho_pow:
        T_rho_args[0] = T * rho ** -T_rho_args[1]

    # Adiabatic
    elif T_rho_type_id == gv.type_adb:
        if mat_type == gv.type_idg:
            # T_rho_args = [T rho^(1-gamma),]
            gamma = idg.idg_gamma(mat_id)
            T_rho_args[0] = T * rho ** (1 - gamma)

        elif mat_id == gv.id_HM80_HHe:
            # T_rho_args = [rho_prv, T_prv]
            T_rho_args[0] = rho
            T_rho_args[1] = T

        elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
            # T_rho_args = [s_adb,]
            T_rho_args[0] = sesame.s_rho_T(rho, T, mat_id)

    # Fixed entropy
    elif T_rho_type_id == gv.type_ent:
        if mat_type in [gv.type_SESAME, gv.type_ANEOS]:
            # T_rho_args = [s,]
            T_rho_args[0] = sesame.s_rho_T(rho, T, mat_id)

    else:
        raise ValueError("T-rho relation not implemented")

    return T_rho_args


def T_rho_id_and_args_from_type(A1_T_rho_type):
    """Convert input T-rho types into the internal T-rho arrays.

    Example: ['power=2.4', 'adiabatic'] --> [1, 2], [[None, 2.4], [None, None]]

    Parameters
    ----------
    A1_T_rho_type : [string]
        Array of T-rho relations for each layer. See tutorial.ipynb and
        glob_vars.py for more info.

    Returns
    -------
    A1_T_rho_id : [int]
        Array of T-rho relation ids for each layer. See glob_vars.py.

    A1_T_rho_args : [[float]]
        Array of T-rho arguments for each layer. See glob_vars.py.
    """

    # A1_T_rho_id
    A1_T_rho_type_str = np.copy(A1_T_rho_type)

    for i, string in enumerate(A1_T_rho_type_str):
        A1_T_rho_type_str[i] = string.split("=")[0]

    A1_T_rho_type_id = [gv.Di_T_rho_id[T_rho_id] for T_rho_id in A1_T_rho_type_str]

    # A1_T_rho_args
    A1_T_rho_args = []

    for i, string in enumerate(A1_T_rho_type):
        T_rho_args = [None, None]
        if string.split("=")[0] == "power":
            T_rho_args[1] = float(string.split("=")[1])
        elif string.split("=")[0] == "entropy":
            T_rho_args[0] = float(string.split("=")[1])
        A1_T_rho_args.append(T_rho_args)

    return A1_T_rho_type_id, A1_T_rho_args
