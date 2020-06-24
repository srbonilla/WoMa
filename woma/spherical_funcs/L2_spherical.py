"""
WoMa 1 layer spherical functions 
"""

import numpy as np
from numba import njit
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from woma.misc import glob_vars as gv
from woma.spherical_funcs import L1_spherical
from woma.eos import eos
from woma.eos.T_rho import T_rho, set_T_rho_args


@njit
def L2_integrate(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    R1,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
):
    """ Integration of a 2 layer spherical planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            R1 (float):
                Boundary between layers 1 and 2 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

        Returns:
            A1_r ([float]):
                Array of radii (SI).

            A1_m_enc ([float]):
                Array of cumulative mass (SI).

            A1_P ([float]):
                Array of pressures (SI).

            A1_T ([float]):
                Array of temperatures (SI).

            A1_rho ([float]):
                Array of densities (SI).

            A1_u ([float]):
                Array of internal energy (SI).

            A1_mat_id ([float]):
                Array of material ids (SI).
    """
    A1_r = np.linspace(R, 0, int(num_prof))
    A1_m_enc = np.zeros(A1_r.shape)
    A1_P = np.zeros(A1_r.shape)
    A1_T = np.zeros(A1_r.shape)
    A1_rho = np.zeros(A1_r.shape)
    A1_u = np.zeros(A1_r.shape)
    A1_mat_id = np.zeros(A1_r.shape)

    u_s = eos.u_rho_T(rho_s, T_s, mat_id_L2)
    # Set the T-rho relation
    T_rho_args_L2 = set_T_rho_args(
        T_s, rho_s, T_rho_type_id_L2, T_rho_args_L2, mat_id_L2
    )

    dr = A1_r[0] - A1_r[1]

    A1_m_enc[0] = M
    A1_P[0] = P_s
    A1_T[0] = T_s
    A1_rho[0] = rho_s
    A1_u[0] = u_s
    A1_mat_id[0] = mat_id_L2

    for i in range(1, A1_r.shape[0]):
        # Layer 2
        if A1_r[i] > R1:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L2
            T_rho_type_id = T_rho_type_id_L2
            T_rho_args = T_rho_args_L2
            rho0 = rho
        # Layer 1, 2 boundary
        elif A1_r[i] <= R1 and A1_r[i - 1] > R1:
            rho = eos.find_rho(
                A1_P[i - 1], mat_id_L1, 1, [A1_T[i - 1], 0.0], A1_rho[i - 1], 100000
            )
            rho = eos.rho_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L1)
            T_rho_args_L1 = set_T_rho_args(
                A1_T[i - 1], rho, T_rho_type_id_L1, T_rho_args_L1, mat_id_L1
            )

            mat_id = mat_id_L1
            T_rho_type_id = T_rho_type_id_L1
            T_rho_args = T_rho_args_L1
            rho0 = A1_rho[i - 1]

        # Layer 1
        elif A1_r[i] <= R1:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L1
            T_rho_type_id = T_rho_type_id_L1
            T_rho_args = T_rho_args_L1
            rho0 = A1_rho[i - 1]

        A1_m_enc[i] = A1_m_enc[i - 1] - 4 * np.pi * A1_r[i - 1] ** 2 * rho * dr
        A1_P[i] = A1_P[i - 1] + gv.G * A1_m_enc[i - 1] * rho / (A1_r[i - 1] ** 2) * dr
        A1_rho[i] = eos.find_rho(
            A1_P[i], mat_id, T_rho_type_id, T_rho_args, rho0, 1.1 * rho,
        )
        A1_T[i] = T_rho(A1_rho[i], T_rho_type_id, T_rho_args, mat_id)
        A1_u[i] = eos.u_rho_T(A1_rho[i], A1_T[i], mat_id)
        A1_mat_id[i] = mat_id
        # Update the T-rho parameters
        if mat_id == gv.id_HM80_HHe and T_rho_type_id == gv.type_adb:
            T_rho_args = set_T_rho_args(
                A1_T[i], A1_rho[i], T_rho_type_id, T_rho_args, mat_id
            )

        if A1_m_enc[i] < 0:
            break

    return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id


@njit
def L2_find_mass(
    num_prof,
    R,
    M_max,
    P_s,
    T_s,
    rho_s,
    R1,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
):
    """ Finder of the total mass of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M_max (float):
                Upper bound for the mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            R1 (float):
                Boundary between layers 1 and 2 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).

            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).

        Returns:
            M_max ([float]):
                Mass of the planet (SI).
    """
    M_min = 0.0

    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
        num_prof,
        R,
        M_max,
        P_s,
        T_s,
        rho_s,
        R1,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
    )

    if A1_m_enc[-1] > 0.0:
        while np.abs(M_min - M_max) > 1e-10 * M_min:
            M_try = (M_min + M_max) * 0.5

            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
                num_prof,
                R,
                M_try,
                P_s,
                T_s,
                rho_s,
                R1,
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
            )

            if A1_m_enc[-1] > 0.0:
                M_max = M_try
            else:
                M_min = M_try

    else:
        raise Exception(
            "M_max is too low, ran out of mass in first iteration.\nPlease increase M_max.\n"
        )
        return 0.0

    return M_max


def L2_find_radius(
    num_prof,
    R_max,
    M,
    P_s,
    T_s,
    rho_s,
    R1,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    num_attempt=40,
    verbosity=1,
):
    """ Finder of the total radius of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R_max (float):
                Maximum radius of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            R1 (float):
                Boundary between layers 1 and 2 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

        Returns:
            M_max ([float]):
                Mass of the planet (SI).
    """
    R_min = R1

    A1_r, A1_m_enc_1, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
        num_prof,
        R_max,
        M,
        P_s,
        T_s,
        rho_s,
        R1,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
    )

    rho_s_L1 = eos.rho_P_T(P_s, T_s, mat_id_L1)

    A1_r, A1_m_enc_2, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_spherical.L1_integrate(
        num_prof, R1, M, P_s, T_s, rho_s_L1, mat_id_L1, T_rho_type_id_L1, T_rho_args_L1
    )

    if A1_m_enc_1[-1] > 0:
        raise Exception(
            "R_max too low, excess of mass for R = R_max.\nPlease increase R_max.\n"
        )

    if A1_m_enc_2[-1] == 0:
        e = (
            "R = R1 yields a planet which already lacks mass.\n"
            + "Try increase M or reduce R1.\n"
        )
        raise Exception(e)
        return -1

    if A1_m_enc_1[-1] == 0.0:
        for i in tqdm(
            range(num_attempt), desc="Finding R given M, R1", disable=verbosity == 0
        ):
            R_try = (R_min + R_max) * 0.5

            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
                num_prof,
                R_try,
                M,
                P_s,
                T_s,
                rho_s,
                R1,
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
            )

            if A1_m_enc[-1] > 0.0:
                R_min = R_try
            else:
                R_max = R_try

    return R_min


def L2_find_R1(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    num_attempt=40,
    verbosity=1,
):
    """ Finder of the boundary of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Temperature at the surface (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

        Returns:
            R1_min ([float]):
                Boundary of the planet (SI).
    """
    R1_min = 0.0
    R1_max = R

    # Check all material 2, should be too low density overall
    rho_s_L2 = eos.rho_P_T(P_s, T_s, mat_id_L2)

    A1_r, A1_m_enc_1, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_spherical.L1_integrate(
        num_prof, R, M, P_s, T_s, rho_s_L2, mat_id_L2, T_rho_type_id_L2, T_rho_args_L2
    )

    # Check all material 1, should be too dense overall
    rho_s_L1 = eos.rho_P_T(P_s, T_s, mat_id_L1)

    A1_r, A1_m_enc_2, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_spherical.L1_integrate(
        num_prof, R, M, P_s, T_s, rho_s_L1, mat_id_L1, T_rho_type_id_L1, T_rho_args_L1
    )

    if A1_m_enc_1[-1] == 0:
        e = (
            "Ran out of mass for a planet made of layer 2 material %s.\n"
            "Try increasing the mass (M) or decreasing the radius (R).\n"
            % gv.Di_id_mat[mat_id_L2]
        )
        raise Exception(e)

    elif A1_m_enc_2[-1] > 0:
        e = (
            "Excess of mass for a planet made of layer 1 material %s.\n"
            "Try decreasing the mass (M) or increasing the radius (R).\n"
            % gv.Di_id_mat[mat_id_L2]
        )
        raise Exception(e)

    for i in tqdm(
        range(num_attempt), desc="Finding R1 given R, M", disable=verbosity == 0
    ):
        R1_try = (R1_min + R1_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_try,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
        )

        if A1_m_enc[-1] > 0.0:
            R1_min = R1_try
        else:
            R1_max = R1_try

    return R1_min


def L2_find_R1_R(
    num_prof,
    R_max,
    M1,
    M2,
    P_s,
    T_s,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    num_attempt=40,
    verbosity=1,
):
    """ Finder of the boundary and radius of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R_max (float):
                Max. radius of the planet (SI).

            M1 (float):
                Mass of the core (SI).
                
            M2 (float):
                Mass of the mantle (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Temperature at the surface (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

        Returns:
            R1_min ([float]):
                Boundary of the planet (SI).
    """

    M = M1 + M2
    rho_s_L1 = eos.rho_P_T(P_s, T_s, mat_id_L1)

    # Build planet made of core material
    try:
        R_min = L1_spherical.L1_find_radius(
            num_prof,
            R_max,
            M,
            P_s,
            T_s,
            rho_s_L1,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            verbosity=0,
        )
    except:
        raise Exception(
            "Could not build a planet made of core material.\nPlease increase R_max.\n"
        )

    # Build planet made of mantle material
    try:
        R_max = L1_spherical.L1_find_radius(
            num_prof,
            R_max,
            M,
            P_s,
            T_s,
            rho_s,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            verbosity=0,
        )
    except:
        raise Exception(
            "Could not build a planet made of mantle material.\nPlease increase R_max.\n"
        )

    if R_min > R_max:
        raise Exception(
            "A planet made of core material is bigger than one made of mantle material.\n"
        )

    for i in tqdm(
        range(num_attempt), desc="Finding R1 and R given M1, M2", disable=verbosity == 0
    ):
        R_try = (R_min + R_max) * 0.5

        R1_try = L2_find_R1(
            num_prof,
            R_try,
            M,
            P_s,
            T_s,
            rho_s,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            num_attempt=20,
            verbosity=0,
        )

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
            num_prof,
            R_try,
            M,
            P_s,
            T_s,
            rho_s,
            R1_try,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
        )

        M1_try = A1_m_enc[A1_mat_id == mat_id_L1][0]

        if M1_try > M1:
            R_min = R_try
        else:
            R_max = R_try

    return R1_try, R_try
