"""
WoMa 1 layer spherical functions
"""

import numpy as np
from numba import njit
import warnings

warnings.filterwarnings("ignore")

from woma.misc import glob_vars as gv
from woma.eos import eos
from woma.eos.T_rho import T_rho, set_T_rho_args


@njit
def L1_integrate(num_prof, R, M, P_s, T_s, rho_s, mat_id, T_rho_type_id, T_rho_args):
    """Integration of a 1 layer spherical planet.

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

    mat_id : int
        Material id.

    T_rho_type_id : int
        Relation between A1_T and A1_rho to be used.

    T_rho_args : [float]
        Extra arguments to determine the relation.

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
    # Initialise the profile arrays
    A1_r = np.linspace(R, 0, int(num_prof))
    A1_m_enc = np.zeros(A1_r.shape)
    A1_P = np.zeros(A1_r.shape)
    A1_T = np.zeros(A1_r.shape)
    A1_rho = np.zeros(A1_r.shape)
    A1_u = np.zeros(A1_r.shape)
    A1_mat_id = np.ones(A1_r.shape) * mat_id

    u_s = eos.u_rho_T(rho_s, T_s, mat_id)
    # Set the T-rho relation parameters
    T_rho_args = set_T_rho_args(T_s, rho_s, T_rho_type_id, T_rho_args, mat_id)

    dr = A1_r[0] - A1_r[1]

    # Set the surface values
    A1_m_enc[0] = M
    A1_P[0] = P_s
    A1_T[0] = T_s
    A1_rho[0] = rho_s
    A1_u[0] = u_s

    # Integrate inwards
    for i in range(1, A1_r.shape[0]):
        A1_m_enc[i] = (
            A1_m_enc[i - 1] - 4 * np.pi * A1_r[i - 1] ** 2 * A1_rho[i - 1] * dr
        )
        A1_P[i] = (
            A1_P[i - 1]
            + gv.G * A1_m_enc[i - 1] * A1_rho[i - 1] / (A1_r[i - 1] ** 2) * dr
        )
        A1_rho[i] = eos.find_rho(
            A1_P[i],
            mat_id,
            T_rho_type_id,
            T_rho_args,
            A1_rho[i - 1],
            1.1 * A1_rho[i - 1],
        )
        A1_T[i] = T_rho(A1_rho[i], T_rho_type_id, T_rho_args, mat_id)
        A1_u[i] = eos.u_rho_T(A1_rho[i], A1_T[i], mat_id)
        # Update the T-rho parameters
        if mat_id == gv.id_HM80_HHe and T_rho_type_id == gv.type_adb:
            T_rho_args = set_T_rho_args(
                A1_T[i], A1_rho[i], T_rho_type_id, T_rho_args, mat_id
            )

        # Stop if run out of mass
        if A1_m_enc[i] < 0:
            return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id

    return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id


@njit
def L1_integrate_out(
    r, dr, m_enc, P, T, u, mat_id, T_rho_type_id, T_rho_args, rho_min=0, P_min=0
):
    """Integrate a new layer of a spherical planet outwards.

    Parameters
    ----------
    r : float
        The radius at the base (m).

    dr : float
        The radius step for the integration (m).

    m_enc : float
        The enclosed mass at the base (Pa).

    P : float
        The pressure at the base (Pa).

    T : float
        The temperature at the base (K).

    u : float
        The specific internal energy at the base (J kg^-1).

    mat_id : int
        The material ID.

    T_rho_type_id : int
        The ID for the temperature-density relation.

    T_rho_args : [float]
        Extra arguments for the temperature-density relation.

    rho_min : float
        The minimum density (must be >= 0) at which the new layer will stop.

    P_min : float
        The minimum pressure (must be >= 0) at which the new layer will stop.

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
    # Initialise the profile arrays
    A1_r = [r]
    A1_m_enc = [m_enc]
    A1_P = [P]
    A1_T = [T]
    A1_u = [u]
    A1_mat_id = [mat_id]
    A1_rho = [eos.rho_P_T(A1_P[0], A1_T[0], mat_id)]
    if T_rho_type_id == gv.type_adb and (T_rho_args[0] is None or np.isnan(T_rho_args[0])):
        T_rho_args = set_T_rho_args(
            A1_T[-1],
            A1_rho[-1],
            T_rho_type_id,
            T_rho_args,
            mat_id,
        )

    # Integrate outwards until the minimum density (or zero pressure)
    while A1_rho[-1] > rho_min and A1_P[-1] > P_min:
        A1_r.append(A1_r[-1] + dr)
        A1_m_enc.append(
            A1_m_enc[-1] + 4 * np.pi * A1_r[-1] * A1_r[-1] * A1_rho[-1] * dr
        )
        A1_P.append(A1_P[-1] - gv.G * A1_m_enc[-1] * A1_rho[-1] / (A1_r[-1] ** 2) * dr)
        if A1_P[-1] <= 0:
            # Add dummy values which will be removed along with the -ve P
            A1_rho.append(0)
            A1_T.append(0)
            A1_u.append(0)
            A1_mat_id.append(0)
            break
        # Update the T-rho parameters
        if T_rho_type_id == gv.type_adb and mat_id == gv.id_HM80_HHe:
            T_rho_args = set_T_rho_args(
                A1_T[-1],
                A1_rho[-1],
                T_rho_type_id,
                T_rho_args,
                mat_id,
            )
        rho = eos.find_rho(
            A1_P[-1],
            mat_id,
            T_rho_type_id,
            T_rho_args,
            0.9 * A1_rho[-1],
            A1_rho[-1],
        )
        A1_rho.append(rho)
        A1_T.append(T_rho(rho, T_rho_type_id, T_rho_args, mat_id))
        A1_u.append(eos.u_rho_T(rho, A1_T[-1], mat_id))
        A1_mat_id.append(mat_id)

    # Remove the duplicate first step and the final too-low density or too-low
    # pressure step
    return (
        A1_r[1:-1],
        A1_m_enc[1:-1],
        A1_P[1:-1],
        A1_T[1:-1],
        A1_rho[1:-1],
        A1_u[1:-1],
        A1_mat_id[1:-1],
    )


def L1_find_M_given_R(
    num_prof,
    R,
    M_max,
    P_s,
    T_s,
    rho_s,
    mat_id,
    T_rho_type_id,
    T_rho_args,
    num_attempt=40,
    tol=1e-7,
    verbosity=1,
):
    """Finder of the total mass of the planet.
        The correct value yields m_enc -> 0 at the center of the planet.

    Parameters
    ----------
    num_prof : int
        Number of profile integration steps.

    R : float
        Radii of the planet (m).

    M_max : float
        Upper bound for the mass of the planet (kg).

    P_s : float
        Pressure at the surface (Pa).

    T_s : float
        Temperature at the surface (K).

    rho_s : float
        Density at the surface (kg m^-3).

    mat_id : int
        Material id.

    T_rho_type_id : int
        Relation between A1_T and A1_rho to be used.

    T_rho_args : [float]
        Extra arguments to determine the relation.

    num_attempt : float
        Maximum number of iterations to perform.

    tol : float
        Tolerance level. Relative difference between consecutive masses.

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

    # Iterate the mass
    for i in range(num_attempt):

        M_try = (M_min + M_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
            num_prof, R, M_try, P_s, T_s, rho_s, mat_id, T_rho_type_id, T_rho_args
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

    # Error message is M tends to M_max
    if (M_max_input - M_max) / M_max < tol:
        raise ValueError("M tends to M_max. Please increase M_max")

    return M_max


def L1_find_R_given_M(
    num_prof,
    R_max,
    M,
    P_s,
    T_s,
    rho_s,
    mat_id,
    T_rho_type_id,
    T_rho_args,
    num_attempt=40,
    tol=0.01,
    verbosity=1,
):
    """Finder of the total radius of the planet.
        The correct value yields m_enc -> 0 at the center of the planet.

    Parameters
    ----------
    num_prof : int
        Number of profile integration steps.

    R_max : float
        Maximuum radius of the planet (m).

    M : float
        Mass of the planet (kg).

    P_s : float
        Pressure at the surface (Pa).

    T_s : float
        Temperature at the surface (K).

    rho_s : float
        Density at the surface (kg m^-3).

    mat_id : int
        Material id.

    T_rho_type_id : int
        Relation between A1_T and A1_rho to be used.

    T_rho_args : [float]
        Extra arguments to determine the relation.

    num_attempt : float
        Maximum number of iterations to perform.

    tol : float
        Tolerance level. Relative difference between two consecutive radius

    verbosity : int
        Printing options.

    Returns
    -------
    R_min : float
        Radius of the planet (m).
    """
    R_min = 0.0

    R_max_input = np.copy(R_max)

    # Iterate the radius
    for i in range(num_attempt):

        R_try = (R_min + R_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
            num_prof,
            R_try,
            M,
            P_s,
            T_s,
            rho_s,
            mat_id,
            T_rho_type_id,
            T_rho_args,
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
        if np.abs(R_try - R_max_input) / R_try < 1 / (num_prof - 1):
            raise ValueError("R tends to R_max. Please increase R_max.")

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    # Message if there is not convergence after num_attempt iterations
    if i == num_attempt - 1 and verbosity >= 1:
        print("\nWarning: Convergence not reached after %d iterations." % (num_attempt))

    # Error message if R tends to R_max
    if np.abs(R_try - R_max_input) / R_max_input < tol:
        raise ValueError("R tends to R_max. Please increase R_max")

    return R_min
