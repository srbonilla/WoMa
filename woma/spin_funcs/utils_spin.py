#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:10:28 2019

@author: sergio
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
def _analytic_solution_r(r, R, Z, x):
    """ Indefinite integral, analytic solution of the potential
        of an oblate spheroid evaluated at x with z = 0. Computed
        from Kellogg's formula.

    Parameters
    ----------
    r (float):
        Cylindrical r coordinate where to compute the potential (SI).

    R (float):
        Mayor axis of the oblate spheroid (SI).

    Z (float):
        Minor axis of the oblate spheroid (SI).

    x (float):
        Integration variable (SI).
        
    Returns
    -------
    A1 + A2 : float
        Analitic solution.
    
    """
    if R == Z:
        return 2 * (r * r - 3 * (R * R + x)) / 3 / np.sqrt((R * R + x) ** 3)
    else:
        A1 = -r * r * np.sqrt(x + Z * Z) / (R * R + x) / (R * R - Z * Z)
        A2 = -(r * r - 2 * R * R + 2 * Z * Z)
        A2 = A2 * np.arctan(np.sqrt((x + Z * Z) / (R * R - Z * Z)))
        A2 = A2 / ((R * R - Z * Z) ** (3 / 2))
        return A1 + A2

    return 0


@njit
def _analytic_solution_z(z, R, Z, x):
    """ Indefinite integral, analytic solution of the optential
        of an oblate spheroid evaluated at x with r = 0. Computed
        from Kellogg's formula.

    Parameters
    ----------
    z (float):
        Cylindrical z coordinate where to compute the potential (SI).

    R (float):
        Mayor axis of the oblate spheroid (SI).

    Z (float):
        Minor axis of the oblate spheroid (SI).

    x (float):
        Integration variable (SI).
                
    Returns
    -------
    A1 + A2 : float
        Analitic solution.
    """

    if R == Z:
        return 2 * (z * z - 3 * (R * R + x)) / 3 / np.sqrt((R * R + x) ** 3)
    else:
        A1 = 2 * z * z / (R * R - Z * Z) / np.sqrt(Z * Z + x)
        A2 = 2 * (R * R + z * z - Z * Z)
        A2 = A2 * np.arctan(np.sqrt((x + Z * Z) / (R * R - Z * Z)))
        A2 = A2 / ((R * R - Z * Z) ** (3 / 2))
        return A1 + A2

    return 0


@njit
def _Vgr(r, R, Z, rho):
    """ Gravitational potential due to an oblate spheroid with constant density
        at r, theta = 0, z = 0.

    Parameters
    ----------
    r (float):
        Cylindrical r coordinate where to compute the optential (SI).

    R (float):
        Mayor axis of the oblate spheroid (SI).

    Z (float):
        Minor axis of the oblate spheroid (SI).

    rho (float):
        Density of the spheroid (SI).

    Returns
    -------
        V (float):
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
        # print("exception")
        Z = R

    if R == Z:
        if r >= R:
            vol = 4 * np.pi * R * R * Z / 3
            return -gv.G * vol * rho / r
        else:
            M = 4 / 3 * np.pi * R ** 3 * rho
            return -gv.G * M / 2 / R ** 3 * (3 * R * R - r * r)

    if r <= R:
        V = np.pi * R * R * Z * rho
        V = V * (_analytic_solution_r(r, R, Z, 1e30) - _analytic_solution_r(r, R, Z, 0))
        return -gv.G * V

    else:
        A = r * r - R * R
        V = np.pi * R * R * Z * rho
        V = V * (_analytic_solution_r(r, R, Z, 1e30) - _analytic_solution_r(r, R, Z, A))
        return -gv.G * V

    return V


@njit
def _Vgz(z, R, Z, rho):
    """ Gravitational potential due to an oblate spheroid with constant density
        at r = 0, theta = 0, z.

    Parameters
    ----------
    z (float):
        Cylindrical z coordinate where to compute the optential (SI).

    R (float):
        Mayor axis of the oblate spheroid (SI).

    Z (float):
        Minor axis of the oblate spheroid (SI).

    rho (float):
        Density of the spheroid (SI).

    Returns
    -------
    V (float):
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
            vol = 4 * np.pi * R * R * Z / 3
            return -gv.G * vol * rho / z
        else:
            M = 4 / 3 * np.pi * R ** 3 * rho
            return -gv.G * M / 2 / R ** 3 * (3 * R * R - z * z)

    if z <= Z:
        V = np.pi * R * R * Z * rho
        V = V * (_analytic_solution_z(z, R, Z, 1e40) - _analytic_solution_z(z, R, Z, 0))
        return -gv.G * V

    else:
        A = z * z - Z * Z
        V = np.pi * R * R * Z * rho
        V = V * (_analytic_solution_z(z, R, Z, 1e40) - _analytic_solution_z(z, R, Z, A))
        return -gv.G * V

    return V


@njit
def _el_eq(r, z, R, Z):
    return r * r / R / R + z * z / Z / Z


@jit(nopython=False)
def rho_rz(r, z, r_array, rho_e, z_array, rho_p):
    """ Computes the density at any point r, z given a spining profile.

    Parameters
    ----------
    r (float):
        Cylindrical r coordinte where to compute the density (SI).

    z (float):
        Cylindrical z coordinte where to compute the density (SI).

    r_array ([float]):
        Points at equatorial profile where the solution is defined (SI).

    rho_e ([float]):
        Equatorial profile of densities (SI).

    z_array ([float]):
        Points at equatorial profile where the solution is defined (SI).

    rho_p ([float]):
        Polar profile of densities (SI).

    Returns
    -------
    rho_2 (float):
        Density at r, z (SI).

    """
    z = np.abs(z)

    rho_e_model = interp1d(r_array, rho_e, bounds_error=False, fill_value=0)
    rho_p_model = interp1d(z_array, rho_p, bounds_error=False, fill_value=0)
    index = np.where(rho_p == 0)[0][0] + 1
    rho_p_model_inv = interp1d(rho_p[:index], z_array[:index])

    r_0 = r
    r_1 = r_array[(rho_e > 0).sum() - 1]

    rho_0 = rho_e_model(r_0)
    rho_1 = rho_e_model(r_1)

    R_0 = r_0
    Z_0 = rho_p_model_inv(rho_0)
    R_1 = r_1
    Z_1 = rho_p_model_inv(rho_1)

    if _el_eq(r, z, R_1, Z_1) > 1:
        return 0

    elif _el_eq(r, z, R_1, Z_1) == 1:
        return rho_1

    elif r == 0 and z == 0:
        return rho_0

    elif r == 0 and z != 0:
        return rho_p_model(z)

    elif r != 0 and z == 0:
        return rho_e_model(r)

    elif _el_eq(r, z, R_0, Z_0) == 1:
        return rho_0

    elif _el_eq(r, z, R_0, Z_0) > 1 and _el_eq(r, z, R_1, Z_1) < 1:
        r_2 = (r_0 + r_1) * 0.5
        rho_2 = rho_e_model(r_2)
        R_2 = r_2
        Z_2 = rho_p_model_inv(rho_2)
        tol = 1e-2

        while np.abs(rho_1 - rho_0) > tol:
            if _el_eq(r, z, R_2, Z_2) > 1:
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
            rho_2 = rho_e_model(r_2)
            R_2 = r_2
            Z_2 = rho_p_model_inv(rho_2)

        return rho_2

    return -1


@njit
def V_spheroid(R, Z):
    """ Computes the volume of a spheroid of parameters R, Z.

    Parameters
    ----------
    R (float):
        Equatorial radius (SI).

    Z (float):
        Polar radius (SI).

    Returns
    -------
    V (float):
        Volume (SI).

    """

    return np.pi * 4 / 3 * R * R * Z


# Particle placement functions

@njit
def integrand(theta, R_l, Z_l, R_h, Z_h):

    r_h = np.sin(theta) ** 2 / R_h / R_h + np.cos(theta) ** 2 / Z_h / Z_h
    r_h = np.sqrt(1 / r_h)

    r_l = np.sin(theta) ** 2 / R_l / R_l + np.cos(theta) ** 2 / Z_l / Z_l
    r_l = np.sqrt(1 / r_l)

    I = 2 * np.pi * (r_h ** 3 - r_l ** 3) * np.sin(theta) / 3

    return I


def V_theta(theta_0, theta_1, shell_config):

    R_l, Z_l = shell_config[0]
    R_h, Z_h = shell_config[1]

    assert R_h >= R_l
    assert Z_h >= Z_l
    assert theta_1 > theta_0

    V = integrate.quad(integrand, theta_0, theta_1, args=(R_l, Z_l, R_h, Z_h))

    return V[0]


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
def _i(theta, R, Z):

    i = -np.sqrt(2) * R * R * np.cos(theta)
    i = i / np.sqrt(
        1 / R / R + 1 / Z / Z + (-1 / R / R + 1 / Z / Z) * np.cos(2 * theta)
    )
    i = i + R * R * Z

    return i


def _V_theta_analytical(theta, shell_config):

    Rm1 = shell_config[0][0]
    Zm1 = shell_config[0][1]
    R1 = shell_config[1][0]
    Z1 = shell_config[1][1]

    V = _i(theta, R1, Z1) - _i(theta, Rm1, Zm1)
    V = V / (_i(np.pi, R1, Z1) - _i(np.pi, Rm1, Zm1))

    return V


@jit(nopython=False)
def compute_M_array(r_array, rho_e, z_array, rho_p):

    index = np.where(rho_p == 0)[0][0] + 1
    rho_p_model_inv = interp1d(rho_p[:index], z_array[:index])
    R_array = r_array
    Z_array = rho_p_model_inv(rho_e)

    M = np.zeros_like(R_array)

    for i in range(1, R_array.shape[0]):

        if rho_e[i] == 0:
            break

        dV = V_spheroid(R_array[i], Z_array[i]) - V_spheroid(
            R_array[i - 1], Z_array[i - 1]
        )
        M[i] = rho_e[i] * dV

    return M


@jit(nopython=False)
def compute_spin_planet_M(r_array, rho_e, z_array, rho_p):

    M = compute_M_array(r_array, rho_e, z_array, rho_p)

    return np.sum(M)


def compute_M_shell(R_shell, r_array, rho_e, z_array, rho_p):

    M_shell = np.zeros_like(R_shell)
    M_array = compute_M_array(r_array, rho_e, z_array, rho_p)

    Re = np.max(r_array[rho_e > 0])

    M_cum = np.cumsum(M_array)
    M_cum_model = interp1d(r_array, M_cum)

    for i in range(M_shell.shape[0]):
        if i == 0:

            R_l = 1e-5
            R_0 = R_shell[i]
            R_h = R_shell[i + 1]
            R_h = (R_h + R_0) / 2

        elif i == M_shell.shape[0] - 1:

            R_l = R_shell[i - 1]
            R_h = Re
            R_0 = R_shell[i]
            R_l = (R_l + R_0) / 2

        else:

            R_l = R_shell[i - 1]
            R_h = R_shell[i + 1]
            R_0 = R_shell[i]
            R_l = (R_l + R_0) / 2
            R_h = (R_h + R_0) / 2

        M_shell[i] = M_cum_model(R_h) - M_cum_model(R_l)

    return M_shell


# main function
def picle_placement(r_array, rho_e, z_array, rho_p, N, period):

    """ Particle placement for a spining profile.

    Parameters
    ----------
    r_array ([float]):
        Points at equatorial profile where the solution is defined (SI).

    rho_e ([float]):
        Equatorial profile of densities (SI).

    z_array ([float]):
        Points at equatorial profile where the solution is defined (SI).

    rho_p ([float]):
        Polar profile of densities (SI).

    N (int):
        Number of particles.
        
    period (float):
        Period of the planet (hours).

    Returns
    -------
    A1_x ([float]):
        Position x of each particle (SI).

    A1_y ([float]):
        Position y of each particle (SI).

    A1_z ([float]):
        Position z of each particle (SI).

    A1_vx ([float]):
        Velocity in x of each particle (SI).

    A1_vy ([float]):
        Velocity in y of each particle (SI).

    A1_vz ([float]):
        Velocity in z of each particle (SI).

    A1_m ([float]):
        Mass of every particle (SI).
        
    A1_rho ([float]):
        Density for every particle (SI).

    A1_h ([float]):
        Smoothing lenght for every particle (SI).

    A1_R ([float]):
        Semi-major axis of the elipsoid for every particle.
        
    A1_Z ([float]):
        Semi-minor axis of the elipsoid for every particle.

    """

    assert len(r_array) == len(rho_e)
    assert len(z_array) == len(rho_p)

    # mass of the model planet
    M = compute_spin_planet_M(r_array, rho_e, z_array, rho_p)

    # Equatorial and polar radius radius
    Re = np.max(r_array[rho_e > 0])
    Rp = np.max(z_array[rho_p > 0])

    # First model - spherical planet from equatorial profile
    radii = np.arange(0, Re, Re / 1000000)
    rho_e_model = interp1d(r_array, rho_e)
    densities = rho_e_model(radii)
    particles = seagen.GenSphere(N, radii[1:], densities[1:], verb=0)

    index = np.where(rho_p == 0)[0][0] + 1
    rho_p_model_inv = interp1d(rho_p[:index], z_array[:index])

    R_shell = np.unique(particles.A1_r)
    # R_shell_outer = particles.A1_r_outer.copy()
    rho_shell = rho_e_model(R_shell)
    Z_shell = rho_p_model_inv(rho_shell)

    # Get picle mass of final configuration
    m_picle = M / N

    M_shell = compute_M_shell(R_shell, r_array, rho_e, z_array, rho_p)

    # Number of particles per shell
    N_shell = np.round(M_shell / m_picle).astype(int)

    # Tweek mass picle per shell to match total mass
    m_picle_shell = M_shell / N_shell

    # Generate shells and make adjustments
    A1_x = []
    A1_y = []
    A1_z = []
    A1_rho = []
    A1_m = []
    A1_R = []
    A1_Z = []

    # all layers but first and last
    for i in tqdm(range(N_shell.shape[0]), desc="Creating shells..."):

        # First shell
        if i == 0:
            # Create analitical model for the shell
            theta_elip = np.linspace(0, np.pi, 100000)

            particles = seagen.GenShell(N_shell[i], R_shell[i])

            R_0 = R_shell[i]
            Z_0 = Z_shell[i]
            R_h = R_shell[i + 1]
            Z_h = Z_shell[i + 1]

            R_l = 1e-5
            Z_l = 1e-5
            R_h = (R_h + R_0) / 2
            Z_h = (Z_h + Z_0) / 2

            shell_config = [[R_l, Z_l], [R_h, Z_h]]

            n_theta_elip = _V_theta_analytical(theta_elip, shell_config)

        # Last shell
        elif i == N_shell.shape[0] - 1:

            if N_shell[-1] > 0:
                # Create analitical model for the shell
                theta_elip = np.linspace(0, np.pi, 100000)

                particles = seagen.GenShell(N_shell[i], R_shell[i])

                R_0 = R_shell[i]
                Z_0 = Z_shell[i]
                R_l = R_shell[i - 1]
                Z_l = Z_shell[i - 1]

                R_l = (R_l + R_0) / 2
                Z_l = (Z_l + Z_0) / 2
                R_h = Re
                Z_h = Rp

                shell_config = [[R_l, Z_l], [R_h, Z_h]]

                n_theta_elip = _V_theta_analytical(theta_elip, shell_config)

            else:
                break

        # Rest of shells
        else:
            # Create analitical model for the shell
            theta_elip = np.linspace(0, np.pi, 100000)

            particles = seagen.GenShell(N_shell[i], R_shell[i])

            R_0 = R_shell[i]
            Z_0 = Z_shell[i]
            R_l = R_shell[i - 1]
            Z_l = Z_shell[i - 1]
            R_h = R_shell[i + 1]
            Z_h = Z_shell[i + 1]

            R_l = (R_l + R_0) / 2
            Z_l = (Z_l + Z_0) / 2
            R_h = (R_h + R_0) / 2
            Z_h = (Z_h + Z_0) / 2

            shell_config = [[R_l, Z_l], [R_h, Z_h]]

            n_theta_elip = _V_theta_analytical(theta_elip, shell_config)

        # Transfor theta acordingly
        theta_elip_n_model = interp1d(n_theta_elip, theta_elip)

        x = particles.A1_x
        y = particles.A1_y
        z = particles.A1_z

        r, theta, phi = cart_to_spher(x, y, z)

        theta = theta_elip_n_model((1 - np.cos(theta)) / 2)

        x, y, z = spher_to_cart(r, theta, phi)

        # Project on the spheroid without changing theta
        alpha = np.sqrt(1 / (x * x / R_0 / R_0 + y * y / R_0 / R_0 + z * z / Z_0 / Z_0))
        x = alpha * x
        y = alpha * y
        z = alpha * z

        # Save results
        A1_x.append(x)
        A1_y.append(y)
        A1_z.append(z)

        A1_rho.append(rho_shell[i] * np.ones(N_shell[i]))
        A1_m.append(m_picle_shell[i] * np.ones(N_shell[i]))
        A1_R.append(R_shell[i] * np.ones(N_shell[i]))
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


def spin_escape_vel(r_array, rho_e, z_array, rho_p, period):
    """
        Computes the escape velocity for a spining planet.
        
    Parameters
    ----------
    r_array ([float]):
        Points at equatorial profile where the solution is defined (SI).

    rho_e ([float]):
        Equatorial profile of densities (SI).

    z_array ([float]):
        Points at equatorial profile where the solution is defined (SI).

    rho_p ([float]):
        Polar profile of densities (SI).
        
    period (float):
        Period of the planet (hours).

    Returns
    -------
    v_escape_equator ([float]):
        Escape velocity at the equator (SI).

    v_escape_pole ([float]):
        Escape velocity at the pole (SI).

        
    """
    V_e, V_p = L1_spin._fillV(r_array, rho_e, z_array, rho_p, period)

    i_equator = min(np.where(rho_e == 0)[0]) - 1
    i_pole = min(np.where(rho_p == 0)[0]) - 1
    V_equator = V_e[i_equator]
    V_pole = V_p[i_pole]
    v_escape_pole = np.sqrt(-2 * V_pole)
    w = 2 * np.pi / period / 60 / 60
    R_e = r_array[i_equator]
    v_escape_equator = np.sqrt(-2 * V_equator - (w * R_e) ** 2)

    return v_escape_equator, v_escape_pole


def spin_iteration(
    period,
    num_layer,
    A1_r_equator,
    A1_rho_equator,
    A1_r_pole,
    A1_rho_pole,
    P_0,
    P_s,
    rho_0,
    rho_s,
    A1_mat_id_layer,
    A1_T_rho_type_id,
    A1_T_rho_args,
    P_1=None,
    P_2=None,
):

    # Use correct function
    if num_layer == 1:

        profile_e, profile_p = L1_spin.spin1layer(
            1,
            A1_r_equator,
            A1_rho_equator,
            A1_r_pole,
            A1_rho_pole,
            period,
            P_0,
            P_s,
            rho_0,
            rho_s,
            A1_mat_id_layer[0],
            A1_T_rho_type_id[0],
            A1_T_rho_args[0],
            verbose=0,
        )

    elif num_layer == 2:

        profile_e, profile_p = L2_spin.spin2layer(
            1,
            A1_r_equator,
            A1_rho_equator,
            A1_r_pole,
            A1_rho_pole,
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
            verbose=0,
        )

    elif num_layer == 3:

        profile_e, profile_p = L3_spin.spin3layer(
            1,
            A1_r_equator,
            A1_rho_equator,
            A1_r_pole,
            A1_rho_pole,
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
            verbose=0,
        )

    A1_rho_equator = profile_e[-1]
    A1_rho_pole = profile_p[-1]

    return A1_rho_equator, A1_rho_pole


def find_min_period(
    num_layer,
    A1_r_equator,
    A1_rho_equator,
    A1_r_pole,
    A1_rho_pole,
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
    print_info=False,
):

    min_period = 0.0001
    tol = 0.00001

    for k in tqdm(
        range(max_iter), desc="Finding minimum period", disable=not print_info
    ):

        try_period = np.mean([min_period, max_period])

        profile_e, _ = spin_iteration(
            try_period,
            num_layer,
            A1_r_equator,
            A1_rho_equator,
            A1_r_pole,
            A1_rho_pole,
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

        if profile_e[-1] > 0:
            min_period = try_period
        else:
            max_period = try_period

        if np.abs(max_period - min_period) / min_period < tol:
            break

    min_period = max_period

    if print_info:
        print("Minimum period: %.3f hours" % (min_period))

    return min_period
