"""
WoMa 3 layer spherical functions 
"""

import numpy as np
from numba import njit
import warnings

warnings.filterwarnings("ignore")

from woma.misc import glob_vars as gv
from woma.misc import utils
from woma.eos import eos
from woma.eos.T_rho import T_rho, set_T_rho_args


@njit
def L3_integrate(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    R1,
    R2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
):
    """Integration of a 2 layer spherical planet.

    Parameters
    ----------
    num_prof : int
        Number of profile integration steps.

    R : float
        Radii of the planet (m).

    M : float
        Mass of the planet (kg).

    P_s : float
        Pressure at the surface (Pa).

    T_s : float
        Temperature at the surface (K).

    rho_s : float
        Density at the surface (kg m^-3).

    R1 : float
        Boundary between layers 1 and 2 (m).

    R2 : float
        Boundary between layers 2 and 3 (m).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between A1_T and A1_rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between A1_T and A1_rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    mat_id_L3 : int
        Material id for layer 3.

    T_rho_type_id_L3 : int
        Relation between A1_T and A1_rho to be used in layer 3.

    T_rho_args_L3 : [float]
        Extra arguments to determine the relation in layer 3.

    Returns
    -------
    A1_r : [float]
        The profile radii, in increasing order (m).

    A1_m_enc : [float]
        The cummulative mass at each profile radius (kg).

    A1_P : [float]
        The pressure at each profile radius (Pa).

    A1_T : [float]
        The temperature at each profile radius (K).

    A1_rho : [float]
        The density at each profile radius (kg m^-3).

    A1_u : [float]
        The specific internal energy at each profile radius (J kg^-1).

    A1_mat_id : [float]
        The ID of the material at each profile radius.
    """
    A1_r = np.linspace(R, 0, int(num_prof))
    A1_m_enc = np.zeros(A1_r.shape)
    A1_P = np.zeros(A1_r.shape)
    A1_T = np.zeros(A1_r.shape)
    A1_rho = np.zeros(A1_r.shape)
    A1_u = np.zeros(A1_r.shape)
    A1_mat_id = np.zeros(A1_r.shape)

    u_s = eos.u_rho_T(rho_s, T_s, mat_id_L3)
    T_rho_args_L3 = set_T_rho_args(
        T_s, rho_s, T_rho_type_id_L3, T_rho_args_L3, mat_id_L3
    )

    dr = A1_r[0] - A1_r[1]

    A1_m_enc[0] = M
    A1_P[0] = P_s
    A1_T[0] = T_s
    A1_rho[0] = rho_s
    A1_u[0] = u_s
    A1_mat_id[0] = mat_id_L3

    for i in range(1, A1_r.shape[0]):
        # Layer 3
        if A1_r[i] > R2:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L3
            T_rho_type_id = T_rho_type_id_L3
            T_rho_args = T_rho_args_L3
            rho0 = rho

        # Layer 2, 3 boundary
        elif A1_r[i] <= R2 and A1_r[i - 1] > R2:
            # New density, continuous temperature unless fixed entropy
            if T_rho_type_id_L2 == gv.type_ent:
                rho = eos.find_rho(
                    A1_P[i - 1],
                    mat_id_L2,
                    T_rho_type_id_L2,
                    T_rho_args_L2,
                    A1_rho[i - 1],
                    1e5,
                )
            else:
                rho = eos.rho_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L2)
                T_rho_args_L2 = set_T_rho_args(
                    A1_T[i - 1], rho, T_rho_type_id_L2, T_rho_args_L2, mat_id_L2
                )

            mat_id = mat_id_L2
            T_rho_type_id = T_rho_type_id_L2
            T_rho_args = T_rho_args_L2
            rho0 = A1_rho[i - 1]

        # Layer 2
        elif A1_r[i] > R1:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L2
            T_rho_type_id = T_rho_type_id_L2
            T_rho_args = T_rho_args_L2
            rho0 = rho

        # Layer 1, 2 boundary
        elif A1_r[i] <= R1 and A1_r[i - 1] > R1:
            # New density, continuous temperature unless fixed entropy
            if T_rho_type_id_L1 == gv.type_ent:
                rho = eos.find_rho(
                    A1_P[i - 1],
                    mat_id_L1,
                    T_rho_type_id_L1,
                    T_rho_args_L1,
                    A1_rho[i - 1],
                    1e5,
                )
            else:
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
            A1_P[i], mat_id, T_rho_type_id, T_rho_args, rho0, 1.1 * rho
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


def L3_find_M_given_R_R1_R2(
    num_prof,
    R,
    M_max,
    P_s,
    T_s,
    rho_s,
    R1,
    R2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    num_attempt=40,
    tol=0.01,
    verbosity=1,
):
    """Finder of the total mass of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

    Parameters
    ----------
    num_prof : int
        Number of profile integration steps.

    R : float
        Radii of the planet (m).

    M : float
        Mass of the planet (kg).

    P_s : float
        Pressure at the surface (Pa).

    T_s : float
        Temperature at the surface (K).

    rho_s : float
        Density at the surface (kg m^-3).

    R1 : float
        Boundary between layers 1 and 2 (m).

    R2 : float
        Boundary between layers 2 and 3 (m).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between A1_T and A1_rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between A1_T and A1_rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    mat_id_L3 : int
        Material id for layer 3.

    T_rho_type_id_L3 : int
        Relation between A1_T and A1_rho to be used in layer 3.

    T_rho_args_L3 : [float]
        Extra arguments to determine the relation in layer 3.

    num_attempt : float
        Maximum number of iterations to perform.

    tol : float
        Tolerance level. Relative difference between two consecutive masses.

    verbosity : int
        Printing options.

    Returns
    -------
    M_max : float
        Mass of the planet (kg).

    """
    # Need this tolerance to avoid peaks in the centre of the planet for the density profile
    tol_max = 1e-7
    if tol > tol_max:
        if verbosity >= 1:
            print("Tolerance overwritten to maximum: %g" % tol_max)
        tol = tol_max

    M_min = 0.0
    M_max_input = np.copy(M_max)

    for i in range(num_attempt):

        M_try = (M_min + M_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R,
            M_try,
            P_s,
            T_s,
            rho_s,
            R1,
            R2,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        if A1_m_enc[-1] > 0.0:
            M_max = M_try
        else:
            M_min = M_try

        tol_reached = np.abs(M_min - M_max) / M_min

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): M=%.5gM_E: tol=%.2g(%.2g)"
                % (i + 1, num_attempt, M_try / gv.M_earth, tol_reached, tol),
                end="  ",
                flush=True,
            )

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    # Message if there is not convergence after num_attempt iterations
    if i == num_attempt - 1 and verbosity >= 1:
        print("\nWarning: Convergence not reached after %d iterations." % (num_attempt))

    # Error messages
    if (M_max_input - M_max) / M_max < tol:
        raise ValueError("M tends to M_max. Please increase M_max")

    return M_max


def L3_find_R_given_M_R1_R2(
    num_prof,
    R_max,
    M,
    P_s,
    T_s,
    rho_s,
    R1,
    R2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    num_attempt=40,
    tol=0.01,
    verbosity=1,
):
    """Finder of the total mass of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

    Parameters
    ----------
    num_prof : int
        Number of profile integration steps.

    R_max : float
        Maximum radius of the planet (m).

    M : float
        Mass of the planet (kg).

    P_s : float
        Pressure at the surface (Pa).

    T_s : float
        Temperature at the surface (K).

    rho_s : float
        Density at the surface (kg m^-3).

    R1 : float
        Boundary between layers 1 and 2 (m).

    R2 : float
        Boundary between layers 2 and 3 (m).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between A1_T and A1_rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between A1_T and A1_rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    mat_id_L3 : int
        Material id for layer 3.

    T_rho_type_id_L3 : int
        Relation between A1_T and A1_rho to be used in layer 3.

    T_rho_args_L3 : [float]
        Extra arguments to determine the relation in layer 3.

    num_attempt : float
        Maximum number of iterations to perform.

    tol : float
        Tolerance level. Relative difference between two consecutive masses or radius.

    verbosity : int
        Printing options.

    Returns
    -------
    R_min : float
        Mass of the planet (m).

    """
    if R1 > R2:
        e = "R1 should not be greater than R2"
        raise ValueError(e)

    R_min = R2
    R_max_input = np.copy(R_max)

    for i in range(num_attempt):
        R_try = (R_min + R_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R_try,
            M,
            P_s,
            T_s,
            rho_s,
            R1,
            R2,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        if A1_m_enc[-1] > 0.0:
            R_min = R_try
        else:
            R_max = R_try

        tol_reached = np.abs(R_min - R_max) / R_max

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): R=%.5gR_E: tol=%.2g(%.2g)"
                % (i + 1, num_attempt, R_try / gv.R_earth, tol_reached, tol),
                end="  ",
                flush=True,
            )

        # Error messages
        if np.abs(R_try - R2) / R_try < 1 / (num_prof - 1):
            raise ValueError("R tends to R2. Please decrease R2.")

        if np.abs(R_try - R_max_input) / R_try < 1 / (num_prof - 1):
            raise ValueError("R tends to R_max. Please increase R_max.")

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    # Message if there is not convergence after num_attempt iterations
    if i == num_attempt - 1 and verbosity >= 1:
        print("\nWarning: Convergence not reached after %d iterations." % (num_attempt))

    # Error messages
    if np.abs(R_min - R_max_input) / R_max_input < 2 * tol:
        raise ValueError("R tends to R_max. Please increase R_max.")

    if np.abs(R_min - R2) / R_min < 2 * tol:
        raise ValueError("R tends to R2. Please decrease R2.")

    return R_min


def L3_find_R2_given_M_R_R1(
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
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    num_attempt=40,
    tol=0.01,
    verbosity=1,
):
    """Finder of the boundary between layers 2 and 3 of the planet for
        fixed boundary between layers 1 and 2.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

    Parameters
    ----------
    num_prof : int
        Number of profile integration steps.

    R : float
        Radii of the planet (m).

    M : float
        Mass of the planet (kg).

    P_s : float
        Pressure at the surface (Pa).

    T_s : float
        Temperature at the surface (K).

    rho_s : float
        Density at the surface (kg m^-3).

    R1 : float
        Boundary between layers 1 and 2 (m).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between A1_T and A1_rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between A1_T and A1_rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    mat_id_L3 : int
        Material id for layer 3.

    T_rho_type_id_L3 : int
        Relation between A1_T and A1_rho to be used in layer 3.

    T_rho_args_L3 : [float]
        Extra arguments to determine the relation in layer 3.

    num_attempt : float
        Maximum number of iterations to perform.

    tol : float
        Tolerance level. Relative difference between two consecutive masses or radius.

    verbosity : int
        Printing options.

    Returns
    -------
    R2_max : float
        Boundary between layers 2 and 3 of the planet (m).
    """
    R2_min = R1
    R2_max = R

    for i in range(num_attempt):

        R2_try = (R2_min + R2_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1,
            R2_try,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        if A1_m_enc[-1] > 0.0:
            R2_min = R2_try
        else:
            R2_max = R2_try

        tol_reached = np.abs(R2_min - R2_max) / R2_max

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): R2=%.5gR_E: tol=%.2g(%.2g)"
                % (i + 1, num_attempt, R2_try / gv.R_earth, tol_reached, tol),
                end="  ",
                flush=True,
            )

        # Error messages
        if np.abs(R2_try - R1) / R < 1 / (num_prof - 1):
            raise ValueError("R2 tends to R1. Please decrease R1.")

        if np.abs(R2_try - R) / R < 1 / (num_prof - 1):
            raise ValueError("R2 tends to R. Please increase R.")

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    # Message if there is not convergence after num_attempt iterations
    if i == num_attempt - 1 and verbosity >= 1:
        print("\nWarning: Convergence not reached after %d iterations." % (num_attempt))

    # Error messages
    if np.abs(R2_max - R) / R < 2 * tol:
        raise ValueError("R2 tends to R. Please increase R.")

    if np.abs(R2_max - R1) / R1 < 2 * tol:
        raise ValueError("R2 tends to R1. Please decrease R1.")

    return R2_max


def L3_find_R1_given_M_R_R2(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    R2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    num_attempt=40,
    tol=0.01,
    verbosity=1,
):
    """Finder of the boundary between layers 2 and 3 of the planet for
        fixed boundary between layers 1 and 2.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

    Parameters
    ----------
    num_prof : int
        Number of profile integration steps.

    R : float
        Radii of the planet (m).

    M : float
        Mass of the planet (kg).

    P_s : float
        Pressure at the surface (Pa).

    T_s : float
        Temperature at the surface (K).

    rho_s : float
        Density at the surface (kg m^-3).

    R2 : float
        Boundary between layers 2 and 3 (m).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between A1_T and A1_rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between A1_T and A1_rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    mat_id_L3 : int
        Material id for layer 3.

    T_rho_type_id_L3 : int
        Relation between A1_T and A1_rho to be used in layer 3.

    T_rho_args_L3 : [float]
        Extra arguments to determine the relation in layer 3.

    num_attempt : float
        Maximum number of iterations to perform.

    tol : float
        Tolerance level. Relative difference between two consecutive masses or radius.

    verbosity : int
        Printing options.

    Returns
    -------
    R1_max : float
        Boundary between layers 1 and 2 of the planet (m).
    """
    R1_min = 0.0
    R1_max = R2

    for i in range(num_attempt):

        R1_try = (R1_min + R1_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_try,
            R2,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        if A1_m_enc[-1] > 0.0:
            R1_min = R1_try
        else:
            R1_max = R1_try

        tol_reached = np.abs(R1_min - R1_max) / R1_max

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): R1=%.5gR_E: tol=%.2g(%.2g)"
                % (i + 1, num_attempt, R1_try / gv.R_earth, tol_reached, tol),
                end="  ",
                flush=True,
            )

        # Error messages
        if np.abs(R1_try - R2) / R < 1 / (num_prof - 1):
            raise ValueError("R1 tends to R2. Please increase R2.")

        if R1_try / R < 1 / (num_prof - 1):
            raise ValueError("R1 tends to 0. Please decrease R1 or R.")

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    # Message if there is not convergence after num_attempt iterations
    if i == num_attempt - 1 and verbosity >= 1:
        print("\nWarning: Convergence not reached after %d iterations." % (num_attempt))

    # Error messages
    if np.abs(R1_max - R2) / R2 < 2 * tol:
        raise ValueError("R1 tends to R2. Please increase R2.")

    if R1_max / R < 2 * tol:
        raise ValueError("R1 tends to 0. Please decrease R1 or R.")

    return R1_max


def L3_find_R1_R2_given_M_R_I(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    I_MR2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    R1_min=None,
    R1_max=None,
    num_attempt=40,
    num_attempt_2=40,
    tol=0.01,
    verbosity=1,
):
    """Finder of the boundaries of the planet for a
        fixed moment of inertia factor.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

    Parameters
    ----------
    num_prof : int
        Number of profile integration steps.

    R : float
        Radii of the planet (m).

    M : float
        Mass of the planet (kg).

    P_s : float
        Pressure at the surface (Pa).

    T_s : float
        Temperature at the surface (K).

    rho_s : float
        Density at the surface (kg m^-3).

    I_MR2 : float
        Moment of inertia factor.

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between A1_T and A1_rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between A1_T and A1_rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    mat_id_L3 : int
        Material id for layer 3.

    T_rho_type_id_L3 : int
        Relation between A1_T and A1_rho to be used in layer 3.

    T_rho_args_L3 : [float]
        Extra arguments to determine the relation in layer 3.

    R1_min : float
        Minimum core-mantle boundary to consider (m).

    R1_max : float
        Maximum core-mantle boundary to consider (m).

    num_attempt : float
        Maximum number of iterations to perform. Outer loop.

    num_attempt_2 : float
        Maximum number of iterations to perform. Inner loop.

    tol : float
        Tolerance level. Relative difference between two consecutive masses or radius.

    verbosity : int
        Printing options.

    Returns
    -------
    R1, R2 : [float]
        Boundaries between layers 1 and 2 and between layers 2 and 3 of
        the planet (m).
    """
    # Normalisation for moment of inertia factor
    MR2 = M * R ** 2

    try:
        if verbosity >= 1:
            print("Creating a planet with R1_min")
        R2_min = L3_find_R2_given_M_R_R1(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_min,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
            num_attempt=num_attempt,
            tol=tol,
            verbosity=verbosity,
        )

    except:
        raise ValueError("Could not build a planet with R1_min.")

    try:
        if verbosity >= 1:
            print("Creating a planet with R1_max")
        R2_max = L3_find_R2_given_M_R_R1(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_max,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
            num_attempt=num_attempt,
            tol=tol,
            verbosity=verbosity,
        )

    except:
        raise ValueError("Could not build a planet with R1_max.")

    (
        A1_r_min,
        A1_m_enc_min,
        A1_P_min,
        A1_T_min,
        A1_rho_min,
        A1_u_min,
        A1_mat_id_min,
    ) = L3_integrate(
        num_prof,
        R,
        M,
        P_s,
        T_s,
        rho_s,
        R1_min,
        R2_min,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
        mat_id_L3,
        T_rho_type_id_L3,
        T_rho_args_L3,
    )

    I_MR2_R1_min = utils.moi(A1_r_min, A1_rho_min) / MR2

    (
        A1_r_max,
        A1_m_enc_max,
        A1_P_max,
        A1_T_max,
        A1_rho_max,
        A1_u_max,
        A1_mat_id_max,
    ) = L3_integrate(
        num_prof,
        R,
        M,
        P_s,
        T_s,
        rho_s,
        R1_max,
        R2_max,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
        mat_id_L3,
        T_rho_type_id_L3,
        T_rho_args_L3,
    )

    I_MR2_R1_max = utils.moi(A1_r_max, A1_rho_max) / MR2

    # compute I_MR2_min and _max
    I_MR2_min = np.min([I_MR2_R1_min, I_MR2_R1_max])
    I_MR2_max = np.max([I_MR2_R1_min, I_MR2_R1_max])

    if verbosity >= 1:
        print("Minimum moment of inertia factor found: {:.3f}".format(I_MR2_min))
        print("Maximum moment of inertia factor found: {:.3f}".format(I_MR2_max))

    if I_MR2 < I_MR2_min or I_MR2_max < I_MR2:
        e = (
            "I_MR2 outside the values found for R1_min and R1_max.\n"
            + "Try modifying R1_min, R1_max or I_MR2."
        )

        raise ValueError(e)

    for i in range(num_attempt):
        R1_try = (R1_min + R1_max) * 0.5

        R2_try = L3_find_R2_given_M_R_R1(
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
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
            num_attempt_2,
            tol,
            verbosity=0,
        )

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_try,
            R2_try,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        I_MR2_iter = utils.moi(A1_r, A1_rho) / MR2

        if I_MR2_R1_min == I_MR2_max:
            if I_MR2_iter < I_MR2:
                R1_max = R1_try
            else:
                R1_min = R1_try

        else:
            if I_MR2_iter > I_MR2:
                R1_max = R1_try
            else:
                R1_min = R1_try

        tol_reached = np.abs(I_MR2_iter - I_MR2) / I_MR2

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): R1=%.5gR_E R2=%.5gR_E: tol=%.2g(%.2g)"
                % (
                    i,
                    num_attempt,
                    R1_try / gv.R_earth,
                    R2_try / gv.R_earth,
                    tol_reached,
                    tol,
                ),
                end="  ",
                flush=True,
            )

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    # Message if there is not convergence after num_attempt iterations
    if i == num_attempt - 1 and verbosity >= 1:
        print("\nWarning: Convergence not reached after %d iterations." % (num_attempt))

    return R1_try, R2_try
