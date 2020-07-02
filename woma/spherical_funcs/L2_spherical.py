"""
WoMa 1 layer spherical functions 
"""

import numpy as np
from numba import njit
import sys
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


# @njit
def L2_find_M_given_R_R1(
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
    num_attempt=40,
    tol=0.01,
    verbosity=1,
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

            num_attempt (float):
                Maximum number of iterations to perform.
                
            tol (float):
                Tolerance level. Relative difference between two consecutive masses.
                
            verbosity (int):
                Printing options.

        Returns:
            M_max ([float]):
                Mass of the planet (SI).
    """
    min_tol = 1e-7
    if tol > min_tol:
        tol = min_tol

    M_min = 0.0

    M_max_input = np.copy(M_max)

    for i in range(num_attempt):

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

        tol_reached = np.abs(M_min - M_max) / M_min

        # print info (cannot do it with numba)
        if verbosity >= 1:

            string = (
                "Iteration "
                + str(i)
                + "/"
                + str(num_attempt)
                + ". Tolerance reached "
                + "{:.2e}".format(tol_reached)
                + "/"
                + str(tol)
            )
            sys.stdout.write("\r" + string)

        if tol_reached < tol:

            break

    if verbosity >= 1:
        sys.stdout.write("\n")

    if (M_max_input - M_max) < tol:
        raise ValueError("M tends to M_max")

    return M_max


def L2_find_R_given_M_R1(
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
    tol=0.01,
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
                
            num_attempt (float):
                Maximum number of iterations to perform.
                
            tol (float):
                Tolerance level. Relative difference between two consecutive radius.
                
            verbosity (int):
                Printing options.

        Returns:
            M_max ([float]):
                Mass of the planet (SI).
    """
    R_min = R1
    R_max_input = np.copy(R_max)

    for i in range(num_attempt):
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

        tol_reached = np.abs(R_min - R_max) / R_max

        # print info
        if verbosity >= 1:

            string = (
                "Iteration "
                + str(i)
                + "/"
                + str(num_attempt)
                + ". Tolerance reached "
                + "{:.2e}".format(tol_reached)
                + "/"
                + str(tol)
            )
            sys.stdout.write("\r" + string)

        if tol_reached < tol:

            break

    if verbosity >= 1:
        sys.stdout.write("\n")
        
    if np.abs(R_min - R_max_input) / R_max_input < 2 * tol:
        raise ValueError("R tends to R_max.")

    if np.abs(R_min - R1) / R_min < 2 * tol:
        raise ValueError("R tends to R1.")

    return R_min


def L2_find_R1_given_M_R(
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
    tol=0.01,
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
            
            num_attempt (float):
                Maximum number of iterations to perform.
                
            tol (float):
                Tolerance level. Relative difference between two consecutive radius.
                
            verbosity (int):
                Printing options.

        Returns:
            R1_min ([float]):
                Boundary of the planet (SI).
    """
    R1_min = 0.0
    R1_max = np.copy(R)

    for i in range(num_attempt):
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

        tol_reached = np.abs(R1_min - R1_max) / R1_max

        # print info
        if verbosity >= 1:

            string = (
                "Iteration "
                + str(i)
                + "/"
                + str(num_attempt)
                + ". Tolerance reached "
                + "{:.2e}".format(tol_reached)
                + "/"
                + str(tol)
            )
            sys.stdout.write("\r" + string)

        if tol_reached < tol:

            break

    if verbosity >= 1:
        sys.stdout.write("\n")
        
    if (R - R1_min) / R < 2 * tol or (R - R1_min) / R < 2 / (num_prof - 1):
        raise ValueError("R1 tends to R.")

    if R1_min / R < 2 * tol or R1_min / R < 2 / (num_prof - 1):
        raise ValueError("R1 tends to 0.")

    return R1_min


def L2_find_R_R1_given_M1_M2(
    num_prof,
    R_min,
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
    tol=0.01,
    verbosity=1,
):
    """ Finder of the boundary and radius of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.
                
            R_min (float):
                Min. radius of the planet (SI).

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
                
            num_attempt (float):
                Maximum number of iterations to perform.
                
            tol (float):
                Tolerance level. Relative difference between two consecutive radius.
                
            verbosity (int):
                Printing options.

        Returns:
            R1_min ([float]):
                Boundary of the planet (SI).
    """

    M = M1 + M2
    # rho_s_L1 = eos.rho_P_T(P_s, T_s, mat_id_L1)

    # Build planet made of core material
    if verbosity >= 1:
        print("Trying to build a planet with R_min with gen_prof_L2_find_R1_given_M_R.")
    try:
        _ = L2_find_R1_given_M_R(
            num_prof,
            R_min,
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
            tol=tol,
            num_attempt=num_attempt,
            verbosity=0,
        )
    except:
        raise ValueError("Could not build a planet with R_min.")

    # Build planet made of mantle material
    if verbosity >= 1:
        print("Trying to build a planet with R_max with gen_prof_L2_find_R1_given_M_R.")
    try:
        _ = L2_find_R1_given_M_R(
            num_prof,
            R_max,
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
            tol=tol,
            num_attempt=num_attempt,
            verbosity=0,
        )
    except:
        raise ValueError("Could not build a planet with R_max.")

    for i in range(num_attempt):
        R_try = (R_min + R_max) * 0.5

        R1_try = L2_find_R1_given_M_R(
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
            num_attempt=num_attempt,
            tol=tol,
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

        tol_reached = np.abs(M1_try - M1) / M1

        # print info
        if verbosity >= 1:

            string = (
                "Iteration "
                + str(i)
                + "/"
                + str(num_attempt)
                + ". Tolerance reached "
                + "{:.2e}".format(tol_reached)
                + "/"
                + str(tol)
            )
            sys.stdout.write("\r" + string)

        if tol_reached < tol:

            break

    if verbosity >= 1:
        sys.stdout.write("\n")

        if tol_reached > tol:
            print("Tolerance level not reached. Please modify R_min and R_max.")

    return R1_try, R_try
