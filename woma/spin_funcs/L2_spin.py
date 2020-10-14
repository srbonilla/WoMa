"""
WoMa 1 layer spinning functions 
"""

import numpy as np
from numba import njit

from woma.spin_funcs import L1_spin
from woma.spin_funcs import utils_spin as us
from woma.eos import eos
from woma.eos.T_rho import T_rho


@njit
def L2_rho_eq_po_from_V(
    A1_r_eq,
    A1_V_eq,
    A1_r_po,
    A1_V_po,
    P_0,
    P_1,
    P_s,
    rho_0,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
):
    """Compute densities of equatorial and polar profiles given the potential
        for a 2 layer planet.

    Parameters
    ----------
    A1_r_eq : [float]
        Points at equatorial profile where the solution is defined (m).

    A1_V_eq : [float]
        Equatorial profile of potential (J).

    A1_r_po : [float]
        Points at equatorial profile where the solution is defined (m).

    A1_V_po : [float]
        Polar profile of potential (J).

    P_0 : float
        Pressure at the center of the planet (Pa).

    P_1 : float
        Pressure at the boundary of the planet (Pa).

    P_s : float
        Pressure at the surface of the planet (Pa).

    rho_0 : float
        Density at the center of the planet (kg m^-3).

    rho_s : float
        Density at the surface of the planet (kg m^-3).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between T and rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between T and rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    Returns
    -------
    A1_rho_eq : [float]
        Equatorial profile of densities (kg m^-3).

    A1_rho_po : [float]
        Polar profile of densities (kg m^-3).
    """

    A1_P_eq = np.zeros(A1_V_eq.shape[0])
    A1_P_po = np.zeros(A1_V_po.shape[0])
    A1_rho_eq = np.zeros(A1_V_eq.shape[0])
    A1_rho_po = np.zeros(A1_V_po.shape[0])

    A1_P_eq[0] = P_0
    A1_P_po[0] = P_0
    A1_rho_eq[0] = rho_0
    A1_rho_po[0] = rho_0

    # equatorial profile
    for i in range(A1_r_eq.shape[0] - 1):
        gradV = A1_V_eq[i + 1] - A1_V_eq[i]
        gradP = -A1_rho_eq[i] * gradV
        A1_P_eq[i + 1] = A1_P_eq[i] + gradP

        # avoid overspin
        if A1_P_eq[i + 1] > A1_P_eq[i]:
            A1_rho_eq[i + 1 :] = A1_rho_eq[i]
            break

        # compute density
        if A1_P_eq[i + 1] >= P_s and A1_P_eq[i + 1] >= P_1:
            A1_rho_eq[i + 1] = eos.find_rho(
                A1_P_eq[i + 1],
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                rho_s * 0.1,
                A1_rho_eq[i],
            )

        elif A1_P_eq[i + 1] >= P_s and A1_P_eq[i + 1] < P_1:
            A1_rho_eq[i + 1] = eos.find_rho(
                A1_P_eq[i + 1],
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                rho_s * 0.1,
                A1_rho_eq[i],
            )

        else:
            A1_rho_eq[i + 1] = 0.0
            break

    # polar profile
    for i in range(A1_r_po.shape[0] - 1):
        gradV = A1_V_po[i + 1] - A1_V_po[i]
        gradP = -A1_rho_po[i] * gradV
        A1_P_po[i + 1] = A1_P_po[i] + gradP

        # avoid overspin
        if A1_P_po[i + 1] > A1_P_po[i]:
            A1_rho_po[i + 1 :] = A1_rho_po[i]
            break

        # compute density
        if A1_P_po[i + 1] >= P_s and A1_P_po[i + 1] >= P_1:
            A1_rho_po[i + 1] = eos.find_rho(
                A1_P_po[i + 1],
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                rho_s * 0.1,
                A1_rho_po[i],
            )

        elif A1_P_po[i + 1] >= P_s and A1_P_po[i + 1] < P_1:
            A1_rho_po[i + 1] = eos.find_rho(
                A1_P_po[i + 1],
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                rho_s * 0.1,
                A1_rho_po[i],
            )

        else:
            A1_rho_po[i + 1] = 0.0
            break

    return A1_rho_eq, A1_rho_po


def L2_spin(
    num_attempt,
    A1_r_eq,
    A1_rho_eq,
    A1_r_po,
    A1_rho_po,
    period,
    P_0,
    P_1,
    P_s,
    rho_0,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    verbosity=1,
):
    """Compute spining profile of densities for a 2 layer planet.

    Parameters
    ----------
    num_attempt : int
        Number of num_attempt to run.

    A1_r_eq : [float]
        Points at equatorial profile where the solution is defined (m).

    A1_rho_eq : [float]
        Densitity values at corresponding A1_r_eq points (kg m^-3).

    A1_r_po : [float]
        Points at polar profile where the solution is defined (m).

    A1_rho_po : [float]
        Densitity values at corresponding A1_r_po points (kg m^-3).

    period : float
        Period of the planet (hours).

    P_0 : float
        Pressure at the center of the planet (Pa).

    P_1 : float
        Pressure at the boundary of the planet (Pa).

    P_s : float
        Pressure at the surface of the planet (Pa).

    rho_0 : float
        Density at the center of the planet (kg m^-3).

    rho_s : float
        Density at the surface of the planet (kg m^-3).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between T and rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between T and rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    Returns
    -------
    profile_eq : ([[float]])
        List of the num_attempt of the equatorial density profile (kg m^-3).

    profile_po : [[float]]
        List of the num_attempt of the polar density profile (kg m^-3).

    """

    profile_eq = []
    profile_po = []

    profile_eq.append(A1_rho_eq)
    profile_po.append(A1_rho_po)

    for i in range(num_attempt):
        A1_V_eq, A1_V_po = L1_spin.V_eq_po_from_rho(
            A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po, period
        )
        A1_rho_eq, A1_rho_po = L2_rho_eq_po_from_V(
            A1_r_eq,
            A1_V_eq,
            A1_r_po,
            A1_V_po,
            P_0,
            P_1,
            P_s,
            rho_0,
            rho_s,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
        )
        profile_eq.append(A1_rho_eq)
        profile_po.append(A1_rho_po)

    return profile_eq, profile_po
