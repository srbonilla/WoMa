"""
WoMa general spinning functions
"""

import numpy as np
from numba import njit, jit
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from tqdm import tqdm
import seagen

from woma.misc import glob_vars as gv
from woma.spin_funcs import L1_spin, L2_spin, L3_spin

# Spining model functions
@njit
def Kellogg_V_r_indef(r, R, Z, x):
    """ Indefinite integral, analytic solution of the potential
        of an oblate spheroid evaluated at x with z = 0. Computed
        from Kellogg's formula.

    Parameters
    ----------
    r : float
        Cylindrical r coordinate where to compute the potential (SI).

    R : float
        Major axis of the oblate spheroid (SI).

    Z : float
        Minor axis of the oblate spheroid (SI).

    x : float
        Integration variable (SI).
        
    Returns
    -------
    A1 + A2 : float
        Analytic solution.
    
    """
    if R == Z:
        return 2 * (r ** 2 - 3 * (R ** 2 + x)) / 3 / np.sqrt((R ** 2 + x) ** 3)
    else:
        A1 = -(r ** 2) * np.sqrt(x + Z ** 2) / (R ** 2 + x) / (R ** 2 - Z ** 2)
        A2 = -(r ** 2 - 2 * R ** 2 + 2 * Z ** 2)
        A2 = A2 * np.arctan(np.sqrt((x + Z ** 2) / (R ** 2 - Z ** 2)))
        A2 = A2 / ((R ** 2 - Z ** 2) ** (3 / 2))
        return A1 + A2

    return 0


@njit
def Kellogg_V_z_indef(z, R, Z, x):
    """ Indefinite integral, analytic solution of the potential
        of an oblate spheroid evaluated at x with r = 0. Computed
        from Kellogg's formula.

    Parameters
    ----------
    z : float
        Cylindrical z coordinate where to compute the potential (SI).

    R : float
        Major axis of the oblate spheroid (SI).

    Z : float
        Minor axis of the oblate spheroid (SI).

    x : float
        Integration variable (SI).
    
    Returns
    -------
    A1 + A2 : float
        Analytic solution.
    """

    if R == Z:
        return 2 * (z ** 2 - 3 * (R ** 2 + x)) / 3 / np.sqrt((R ** 2 + x) ** 3)
    else:
        A1 = 2 * z ** 2 / (R ** 2 - Z ** 2) / np.sqrt(Z ** 2 + x)
        A2 = 2 * (R ** 2 + z ** 2 - Z ** 2)
        A2 = A2 * np.arctan(np.sqrt((x + Z ** 2) / (R ** 2 - Z ** 2)))
        A2 = A2 / ((R ** 2 - Z ** 2) ** (3 / 2))
        return A1 + A2

    return 0


@njit
def V_grav_eq(r, R, Z, rho):
    """ Gravitational potential due to an oblate spheroid with constant density
        at r, theta = 0, z = 0.

    Parameters
    ----------
    r : float
        Cylindrical r coordinate where to compute the potential (SI).

    R : float
        Major axis of the oblate spheroid (SI).

    Z : float
        Minor axis of the oblate spheroid (SI).

    rho : float
        Density of the spheroid (SI).

    Returns
    -------
    V : float
        Gravitational potential (SI).
    """

    V = 0

    # Control R and Z
    if R == 0.0 or Z == 0:
        return 0

    elif np.abs((R - Z) / max(R, Z)) < 1e-6:
        R = max(R, Z)
        Z = R

    elif Z > R:
        Z = R

    if R == Z:
        if r >= R:
            vol = 4 * np.pi * R ** 2 * Z / 3
            return -gv.G * vol * rho / r
        else:
            M = 4 / 3 * np.pi * R ** 3 * rho
            return -gv.G * M / 2 / R ** 3 * (3 * R ** 2 - r ** 2)

    if r <= R:
        V = np.pi * R ** 2 * Z * rho
        V = V * (Kellogg_V_r_indef(r, R, Z, 1e30) - Kellogg_V_r_indef(r, R, Z, 0))
        return -gv.G * V

    else:
        A = r ** 2 - R ** 2
        V = np.pi * R ** 2 * Z * rho
        V = V * (Kellogg_V_r_indef(r, R, Z, 1e30) - Kellogg_V_r_indef(r, R, Z, A))
        return -gv.G * V

    return V


@njit
def V_grav_po(z, R, Z, rho):
    """ Gravitational potential due to an oblate spheroid with constant density
        at r = 0, theta = 0, z.

    Parameters
    ----------
    z : float
        Cylindrical z coordinate where to compute the potential (SI).

    R : float
        Major axis of the oblate spheroid (SI).

    Z : float
        Minor axis of the oblate spheroid (SI).

    rho : float
        Density of the spheroid (SI).

    Returns
    -------
    V : float
        Gravitational potential (SI).
    """

    V = 0

    if R == 0.0 or Z == 0:
        return 0

    elif np.abs((R - Z) / max(R, Z)) < 1e-6:
        R = max(R, Z)
        Z = R

    elif Z > R:
        Z = R

    if R == Z:
        if z >= R:
            vol = 4 * np.pi * R ** 2 * Z / 3
            return -gv.G * vol * rho / z
        else:
            M = 4 / 3 * np.pi * R ** 3 * rho
            return -gv.G * M / 2 / R ** 3 * (3 * R ** 2 - z ** 2)

    if z <= Z:
        V = np.pi * R ** 2 * Z * rho
        V = V * (Kellogg_V_z_indef(z, R, Z, 1e40) - Kellogg_V_z_indef(z, R, Z, 0))
        return -gv.G * V

    else:
        A = z ** 2 - Z ** 2
        V = np.pi * R ** 2 * Z * rho
        V = V * (Kellogg_V_z_indef(z, R, Z, 1e40) - Kellogg_V_z_indef(z, R, Z, A))
        return -gv.G * V

    return V


@njit
def ellipse_eqn(r, z, R, Z):
    return r ** 2 / R ** 2 + z ** 2 / Z ** 2


# @jit(nopython=False)
def rho_at_r_z(r, z, A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po):
    """ Computes the density at any point r, z given a spining profile.

    Parameters
    ----------
    r : float
        Cylindrical r coordinate where to compute the density (SI).

    z : float
        Cylindrical z coordinate where to compute the density (SI).

    A1_r_eq : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_eq : [float]
        Equatorial profile of densities (SI).

    A1_r_po : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_po : [float]
        Polar profile of densities (SI).

    Returns
    -------
    rho : float
        Density at r, z (SI).
    """
    z = np.abs(z)

    rho_model_eq = interp1d(A1_r_eq, A1_rho_eq, bounds_error=False, fill_value=0)
    rho_model_po = interp1d(A1_r_po, A1_rho_po, bounds_error=False, fill_value=0)
    index = np.where(A1_rho_po == 0)[0][0] + 1
    rho_model_po_inv = interp1d(A1_rho_po[:index], A1_r_po[:index])

    r_0 = r
    r_1 = A1_r_eq[(A1_rho_eq > 0).sum() - 1]

    rho_0 = rho_model_eq(r_0)
    rho_1 = rho_model_eq(r_1)

    R_0 = r_0
    Z_0 = rho_model_po_inv(rho_0)
    R_1 = r_1
    Z_1 = rho_model_po_inv(rho_1)

    if r == 0 and z == 0:
        return rho_0

    elif r == 0 and z != 0:
        return rho_model_po(z)

    elif r != 0 and z == 0:
        return rho_model_eq(r)

    elif ellipse_eqn(r, z, R_1, Z_1) > 1:
        return 0

    elif ellipse_eqn(r, z, R_1, Z_1) == 1:
        return rho_1

    elif ellipse_eqn(r, z, R_0, Z_0) == 1:
        return rho_0

    elif ellipse_eqn(r, z, R_0, Z_0) > 1 and ellipse_eqn(r, z, R_1, Z_1) < 1:
        r_2 = (r_0 + r_1) * 0.5
        rho_2 = rho_model_eq(r_2)
        R_2 = r_2
        Z_2 = rho_model_po_inv(rho_2)
        tol = 1e-2

        while np.abs(rho_1 - rho_0) > tol:
            if ellipse_eqn(r, z, R_2, Z_2) > 1:
                r_0 = r_2
                rho_0 = rho_2
                R_0 = R_2
                Z_0 = Z_2
            else:
                r_1 = r_2
                rho_1 = rho_2
                R_1 = R_2
                Z_1 = Z_2

            r_2 = (r_0 + r_1) * 0.5
            rho_2 = rho_model_eq(r_2)
            R_2 = r_2
            Z_2 = rho_model_po_inv(rho_2)

        return rho_2

    else:
        raise ValueError("Error finding density")


@njit
def vol_spheroid(R, Z):
    """ Computes the volume of a spheroid of parameters R, Z.

    Parameters
    ----------
    R : float
        Equatorial radius (SI).

    Z : float
        Polar radius (SI).

    Returns
    -------
    V : float
        Volume (SI).

    """

    return np.pi * 4 / 3 * R ** 2 * Z


@njit
def cart_to_spher(x, y, z):

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return r, theta, phi


@njit
def spher_to_cart(r, theta, phi):

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


@njit
def vol_i_partial(theta, R, Z):

    i = -np.sqrt(2) * R ** 2 * np.cos(theta)
    i = i / np.sqrt(
        1 / R ** 2 + 1 / Z ** 2 + (-1 / R ** 2 + 1 / Z ** 2) * np.cos(2 * theta)
    )
    i = i + R ** 2 * Z

    return i


def frac_vol_theta_analytical(theta, R_in, Z_in, R_out, Z_out):

    vol_theta = vol_i_partial(theta, R_out, Z_out) - vol_i_partial(theta, R_in, Z_in)
    vol_tot = vol_i_partial(np.pi, R_out, Z_out) - vol_i_partial(np.pi, R_in, Z_in)

    return vol_theta / vol_tot


@jit(nopython=False)
def spheroid_masses(A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po):

    index = np.where(A1_rho_po == 0)[0][0] + 1
    rho_model_po_inv = interp1d(A1_rho_po[:index], A1_r_po[:index])
    A1_R = A1_r_eq
    Z_array = rho_model_po_inv(A1_rho_eq)

    A1_M = np.zeros_like(A1_R)

    for i in range(1, A1_R.shape[0]):

        if A1_rho_eq[i] == 0:
            break

        dvol = vol_spheroid(A1_R[i], Z_array[i]) - vol_spheroid(
            A1_R[i - 1], Z_array[i - 1]
        )
        A1_M[i] = A1_rho_eq[i] * dvol

    return A1_M


@jit(nopython=False)
def M_spin_planet(A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po):

    return np.sum(spheroid_masses(A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po))


def picle_shell_masses(A1_R_shell, A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po):

    R_eq = np.max(A1_r_eq[A1_rho_eq > 0])

    A1_M_shell = np.zeros_like(A1_R_shell)
    A1_M_cum = np.cumsum(spheroid_masses(A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po))
    get_M_cum_model = interp1d(A1_r_eq, A1_M_cum)

    for i in range(A1_M_shell.shape[0]):
        if i == 0:

            R_in = 1e-5
            R_0 = A1_R_shell[i]
            R_out = A1_R_shell[i + 1]
            R_out = (R_out + R_0) / 2

        elif i == A1_M_shell.shape[0] - 1:

            R_in = A1_R_shell[i - 1]
            R_out = R_eq
            R_0 = A1_R_shell[i]
            R_in = (R_in + R_0) / 2

        else:

            R_in = A1_R_shell[i - 1]
            R_out = A1_R_shell[i + 1]
            R_0 = A1_R_shell[i]
            R_in = (R_in + R_0) / 2
            R_out = (R_out + R_0) / 2

        A1_M_shell[i] = get_M_cum_model(R_out) - get_M_cum_model(R_in)

    return A1_M_shell


def place_particles(A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po, N, period, verbosity=1):

    """ Particle placement for a spining profile.

    Parameters
    ----------
    A1_r_eq : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_eq : [float]
        Equatorial profile of densities (SI).

    A1_r_po : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_po : [float]
        Polar profile of densities (SI).

    N (int):
        Number of particles.
        
    period : float
        Period of the planet (hours).

    Returns
    -------
    A1_x : [float]
        Position x of each particle (SI).

    A1_y : [float]
        Position y of each particle (SI).

    A1_z : [float]
        Position z of each particle (SI).

    A1_vx : [float]
        Velocity in x of each particle (SI).

    A1_vy : [float]
        Velocity in y of each particle (SI).

    A1_vz : [float]
        Velocity in z of each particle (SI).

    A1_m : [float]
        Mass of every particle (SI).
        
    A1_rho : [float]
        Density for every particle (SI).

    A1_h : [float]
        Smoothing lenght for every particle (SI).

    A1_R : [float]
        Semi-major axis of the elipsoid for every particle.
        
    A1_Z : [float]
        Semi-minor axis of the elipsoid for every particle.

    """

    assert len(A1_r_eq) == len(A1_rho_eq)
    assert len(A1_r_po) == len(A1_rho_po)

    # mass of the model planet
    M = M_spin_planet(A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po)

    # Equatorial and polar radius radius
    R_eq = np.max(A1_r_eq[A1_rho_eq > 0])
    R_po = np.max(A1_r_po[A1_rho_po > 0])

    # First model - spherical planet from equatorial profile
    radii = np.arange(0, R_eq, R_eq / 1000000)
    rho_model_eq = interp1d(A1_r_eq, A1_rho_eq)
    densities = rho_model_eq(radii)
    particles = seagen.GenSphere(N, radii[1:], densities[1:], verbosity=verbosity)

    index = np.where(A1_rho_po == 0)[0][0] + 1
    rho_model_po_inv = interp1d(A1_rho_po[:index], A1_r_po[:index])

    A1_R_shell = np.unique(particles.A1_r)
    # R_shell_outer = particles.A1_r_outer.copy()
    rho_shell = rho_model_eq(A1_R_shell)
    Z_shell = rho_model_po_inv(rho_shell)

    # Get picle mass of final configuration
    m_picle = M / N

    A1_M_shell = picle_shell_masses(A1_R_shell, A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po)

    # Number of particles per shell
    N_shell = np.round(A1_M_shell / m_picle).astype(int)

    # Tweek mass picle per shell to match total mass
    m_picle_shell = A1_M_shell / N_shell

    # Generate shells and make adjustments
    A1_x = []
    A1_y = []
    A1_z = []
    A1_rho = []
    A1_m = []
    A1_R = []
    A1_Z = []

    # all layers but first and last
    for i in tqdm(
        range(N_shell.shape[0]), desc="Creating shells...", disable=verbosity == 0
    ):

        # First shell
        if i == 0:
            # Create analitical model for the shell
            theta_elip = np.linspace(0, np.pi, 100000)

            particles = seagen.GenShell(N_shell[i], A1_R_shell[i])

            R_0 = A1_R_shell[i]
            Z_0 = Z_shell[i]
            R_out = A1_R_shell[i + 1]
            Z_out = Z_shell[i + 1]

            R_in = 1e-5
            Z_in = 1e-5
            R_out = (R_out + R_0) / 2
            Z_out = (Z_out + Z_0) / 2

            n_theta_elip = frac_vol_theta_analytical(
                theta_elip, R_in, Z_in, R_out, Z_out
            )

        # Last shell
        elif i == N_shell.shape[0] - 1:

            if N_shell[-1] > 0:
                # Create analitical model for the shell
                theta_elip = np.linspace(0, np.pi, 100000)

                particles = seagen.GenShell(N_shell[i], A1_R_shell[i])

                R_0 = A1_R_shell[i]
                Z_0 = Z_shell[i]
                R_in = A1_R_shell[i - 1]
                Z_in = Z_shell[i - 1]

                R_in = (R_in + R_0) / 2
                Z_in = (Z_in + Z_0) / 2
                R_out = R_eq
                Z_out = R_po

                n_theta_elip = frac_vol_theta_analytical(
                    theta_elip, R_in, Z_in, R_out, Z_out
                )

            else:
                break

        # Rest of shells
        else:
            # Create analitical model for the shell
            theta_elip = np.linspace(0, np.pi, 100000)

            particles = seagen.GenShell(N_shell[i], A1_R_shell[i])

            R_0 = A1_R_shell[i]
            Z_0 = Z_shell[i]
            R_in = A1_R_shell[i - 1]
            Z_in = Z_shell[i - 1]
            R_out = A1_R_shell[i + 1]
            Z_out = Z_shell[i + 1]

            R_in = (R_in + R_0) / 2
            Z_in = (Z_in + Z_0) / 2
            R_out = (R_out + R_0) / 2
            Z_out = (Z_out + Z_0) / 2

            n_theta_elip = frac_vol_theta_analytical(
                theta_elip, R_in, Z_in, R_out, Z_out
            )

        # Transfor theta acordingly
        theta_elip_n_model = interp1d(n_theta_elip, theta_elip)

        x = particles.A1_x
        y = particles.A1_y
        z = particles.A1_z

        r, theta, phi = cart_to_spher(x, y, z)

        theta = theta_elip_n_model((1 - np.cos(theta)) / 2)

        x, y, z = spher_to_cart(r, theta, phi)

        # Project on the spheroid without changing theta
        alpha = np.sqrt(1 / (x ** 2 / R_0 ** 2 + y ** 2 / R_0 ** 2 + z ** 2 / Z_0 ** 2))
        x = alpha * x
        y = alpha * y
        z = alpha * z

        # Save results
        A1_x.append(x)
        A1_y.append(y)
        A1_z.append(z)

        A1_rho.append(rho_shell[i] * np.ones(N_shell[i]))
        A1_m.append(m_picle_shell[i] * np.ones(N_shell[i]))
        A1_R.append(A1_R_shell[i] * np.ones(N_shell[i]))
        A1_Z.append(Z_shell[i] * np.ones(N_shell[i]))

    # Flatten
    A1_x = np.concatenate(A1_x)
    A1_y = np.concatenate(A1_y)
    A1_z = np.concatenate(A1_z)
    A1_rho = np.concatenate(A1_rho)
    A1_m = np.concatenate(A1_m)
    A1_R = np.concatenate(A1_R)
    A1_Z = np.concatenate(A1_Z)

    # Compute velocities (T_w in hours)
    A1_vx = np.zeros(A1_m.shape[0])
    A1_vy = np.zeros(A1_m.shape[0])
    A1_vz = np.zeros(A1_m.shape[0])

    hour_to_s = 3600
    wz = 2 * np.pi / period / hour_to_s

    A1_vx = -A1_y * wz
    A1_vy = A1_x * wz

    return A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_R, A1_Z


def spin_escape_vel(A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po, period):
    """ Computes the escape velocity for a spining planet.
        
    Parameters
    ----------
    A1_r_eq : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_eq : [float]
        Equatorial profile of densities (SI).

    A1_r_po : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_po : [float]
        Polar profile of densities (SI).
        
    period : float
        Period of the planet (hours).

    Returns
    -------
    v_esc_eq : [float]
        Escape velocity at the equator (SI).

    v_esc_po : [float]
        Escape velocity at the pole (SI).

        
    """
    V_eq, V_po = L1_spin.V_eq_po_from_rho(
        A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po, period
    )

    i_eq = min(np.where(A1_rho_eq == 0)[0]) - 1
    i_po = min(np.where(A1_rho_po == 0)[0]) - 1
    V_eq = V_eq[i_eq]
    V_po = V_po[i_po]
    v_esc_po = np.sqrt(-2 * V_po)
    w = 2 * np.pi / period / 60 / 60
    R_eq = A1_r_eq[i_eq]
    v_esc_eq = np.sqrt(-2 * V_eq - (w * R_eq) ** 2)

    return v_esc_eq, v_esc_po


def spin_iteration(
    period,
    num_layer,
    A1_r_eq,
    A1_rho_eq,
    A1_r_po,
    A1_rho_po,
    P_0,
    P_s,
    rho_0,
    rho_s,
    A1_mat_id_layer,
    A1_T_rho_type_id,
    A1_T_rho_args,
    P_1=None,
    P_2=None,
    verbosity=0,
):

    # Use correct function
    if num_layer == 1:

        profile_eq, profile_po = L1_spin.L1_spin(
            1,
            A1_r_eq,
            A1_rho_eq,
            A1_r_po,
            A1_rho_po,
            period,
            P_0,
            P_s,
            rho_0,
            rho_s,
            A1_mat_id_layer[0],
            A1_T_rho_type_id[0],
            A1_T_rho_args[0],
            verbosity=verbosity,
        )

    elif num_layer == 2:

        profile_eq, profile_po = L2_spin.L2_spin(
            1,
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
            A1_mat_id_layer[0],
            A1_T_rho_type_id[0],
            A1_T_rho_args[0],
            A1_mat_id_layer[1],
            A1_T_rho_type_id[1],
            A1_T_rho_args[1],
            verbosity=verbosity,
        )

    elif num_layer == 3:

        profile_eq, profile_po = L3_spin.L3_spin(
            1,
            A1_r_eq,
            A1_rho_eq,
            A1_r_po,
            A1_rho_po,
            period,
            P_0,
            P_1,
            P_2,
            P_s,
            rho_0,
            rho_s,
            A1_mat_id_layer[0],
            A1_T_rho_type_id[0],
            A1_T_rho_args[0],
            A1_mat_id_layer[1],
            A1_T_rho_type_id[1],
            A1_T_rho_args[1],
            A1_mat_id_layer[2],
            A1_T_rho_type_id[2],
            A1_T_rho_args[2],
            verbosity=verbosity,
        )

    A1_rho_eq = profile_eq[-1]
    A1_rho_po = profile_po[-1]

    return A1_rho_eq, A1_rho_po


def find_min_period(
    num_layer,
    A1_r_eq,
    A1_rho_eq,
    A1_r_po,
    A1_rho_po,
    P_0,
    P_s,
    rho_0,
    rho_s,
    A1_mat_id_layer,
    A1_T_rho_type_id,
    A1_T_rho_args,
    P_1=None,
    P_2=None,
    max_period=10,
    max_iter=30,
    verbosity=1,
):

    min_period = 0.0001
    tol = 0.00001

    for k in tqdm(
        range(max_iter), desc="Finding minimum period", disable=verbosity == 0
    ):

        try_period = np.mean([min_period, max_period])

        profile_eq, _ = spin_iteration(
            try_period,
            num_layer,
            A1_r_eq,
            A1_rho_eq,
            A1_r_po,
            A1_rho_po,
            P_0,
            P_s,
            rho_0,
            rho_s,
            A1_mat_id_layer,
            A1_T_rho_type_id,
            A1_T_rho_args,
            P_1,
            P_2,
        )

        if profile_eq[-1] > 0:
            min_period = try_period
        else:
            max_period = try_period

        if np.abs(max_period - min_period) / min_period < tol:
            break

    min_period = max_period

    if verbosity >= 1:
        print("Minimum period: %.3f hours" % (min_period))

    return min_period
