#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:10:28 2019

@author: sergio
"""

import numpy as np
from numba import njit
import glob_vars as gv
from scipy.interpolate import interp1d

@njit
def _analytic_solution_r(r, R, Z, x):
    """ Indefinite integral, analytic solution of the optential
        of an oblate spheroid evaluated at x with z = 0.

        Args:

            r (float):
                Cylindrical r coordinate where to compute the potential (SI).

            R (float):
                Mayor axis of the oblate spheroid (SI).

            Z (float):
                Minor axis of the oblate spheroid (SI).

            x (float):
                Integration variable (SI).
    """
    if R == Z:
        return 2*(r*r - 3*(R*R + x))/3/np.sqrt((R*R + x)**3)
    else:
        A1 = -r*r*np.sqrt(x + Z*Z)/(R*R + x)/(R*R - Z*Z)
        A2 = -(r*r - 2*R*R + 2*Z*Z)
        A2 = A2*np.arctan(np.sqrt((x + Z*Z)/(R*R - Z*Z)))
        A2 = A2/((R*R - Z*Z)**(3/2))
        return A1 + A2

    return 0

@njit
def _analytic_solution_z(z, R, Z, x):
    """ Indefinite integral, analytic solution of the optential
        of an oblate spheroid evaluated at x with r = 0.

        Args:

            z (float):
                Cylindrical z coordinate where to compute the potential (SI).

            R (float):
                Mayor axis of the oblate spheroid (SI).

            Z (float):
                Minor axis of the oblate spheroid (SI).

            x (float):
                Integration variable (SI).
    """

    if R == Z:
        return 2*(z*z - 3*(R*R + x))/3/np.sqrt((R*R + x)**3)
    else:
        A1 = 2*z*z/(R*R - Z*Z)/np.sqrt(Z*Z + x)
        A2 = 2*(R*R + z*z - Z*Z)
        A2 = A2*np.arctan(np.sqrt((x + Z*Z)/(R*R - Z*Z)))
        A2 = A2/((R*R - Z*Z)**(3/2))
        return A1 + A2

    return 0

@njit
def _Vgr(r, R, Z, rho):
    """ Gravitational potential due to an oblate spheroid with constant density
        at r, theta = 0, z = 0.

        Args:

            r (float):
                Cylindrical r coordinate where to compute the optential (SI).

            R (float):
                Mayor axis of the oblate spheroid (SI).

            Z (float):
                Minor axis of the oblate spheroid (SI).

            rho (float):
                Density of the spheroid (SI).

        Returns:
            V (float):
                Gravitational potential (SI).
    """

    V = 0

    # Control R and Z
    if R == 0. or Z == 0:
        return 0

    elif np.abs((R - Z)/max(R, Z)) < 1e-6:
        R = max(R, Z)
        Z = R

    elif Z > R:
        #print("exception")
        Z = R


    if R == Z:
        if r >= R:
            vol = 4*np.pi*R*R*Z/3
            return -gv.G*vol*rho/r
        else:
            M = 4/3*np.pi*R**3*rho
            return -gv.G*M/2/R**3*(3*R*R - r*r)


    if r <= R:
        V = np.pi*R*R*Z*rho
        V = V*(_analytic_solution_r(r, R, Z, 1e30)
               - _analytic_solution_r(r, R, Z, 0))
        return -gv.G*V

    else:
        A = r*r - R*R
        V = np.pi*R*R*Z*rho
        V = V*(_analytic_solution_r(r, R, Z, 1e30)
               - _analytic_solution_r(r, R, Z, A))
        return -gv.G*V

    return V

@njit
def _Vgz(z, R, Z, rho):
    """ Gravitational potential due to an oblate spheroid with constant density
        at r = 0, theta = 0, z.

        Args:

            z (float):
                Cylindrical z coordinate where to compute the optential (SI).

            R (float):
                Mayor axis of the oblate spheroid (SI).

            Z (float):
                Minor axis of the oblate spheroid (SI).

            rho (float):
                Density of the spheroid (SI).

        Returns:
            V (float):
                Gravitational potential (SI).
    """

    V = 0

    if R == 0. or Z == 0:
        return 0

    elif np.abs((R - Z)/max(R, Z)) < 1e-6:
        R = max(R, Z)
        Z = R

    elif Z > R:
        Z = R


    if R == Z:
        if z >= R:
            vol = 4*np.pi*R*R*Z/3
            return -gv.G*vol*rho/z
        else:
            M = 4/3*np.pi*R**3*rho
            return -gv.G*M/2/R**3*(3*R*R - z*z)


    if z <= Z:
        V = np.pi*R*R*Z*rho
        V = V*(_analytic_solution_z(z, R, Z, 1e40)
               - _analytic_solution_z(z, R, Z, 0))
        return -gv.G*V

    else:
        A = z*z - Z*Z
        V = np.pi*R*R*Z*rho
        V = V*(_analytic_solution_z(z, R, Z, 1e40)
               - _analytic_solution_z(z, R, Z, A))
        return -gv.G*V

    return V

@njit
def _el_eq(r, z, R, Z):
    return r*r/R/R + z*z/Z/Z

#@njit
def rho_rz(r, z, r_array, rho_e, z_array, rho_p):
    """ Computes the density at any point r, z given a spining profile.

        Args:

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

        Returns:

            rho_2 (float):
                Density at r, z (SI).

    """
    z = np.abs(z)

    rho_e_model = interp1d(r_array, rho_e, bounds_error=False, fill_value=0)
    rho_p_model = interp1d(z_array, rho_p, bounds_error=False, fill_value=0)
    rho_p_model_inv = interp1d(rho_p, z_array)

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
def cubic_spline_kernel(rij, h):

    gamma = 1.825742
    H     = gamma*h
    C     = 16/np.pi
    u     = rij/H

    fu = np.zeros(u.shape)

    mask_1     = u < 1/2
    fu[mask_1] = (3*np.power(u,3) - 3*np.power(u,2) + 0.5)[mask_1]

    mask_2     = np.logical_and(u > 1/2, u < 1)
    fu[mask_2] = (-np.power(u,3) + 3*np.power(u,2) - 3*u + 1)[mask_2]

    return C*fu/np.power(H,3)

@njit
def N_neig_cubic_spline_kernel(eta):

    gamma = 1.825742

    return 4/3*np.pi*(gamma*eta)**3

@njit
def eta_cubic_spline_kernel(N_neig):

    gamma = 1.825742

    return np.cbrt(3*N_neig/4/np.pi)/gamma

@njit
def SPH_density(M, R, H):

    rho_sph = np.zeros(H.shape[0])

    for i in range(H.shape[0]):

        rho_sph[i] = np.sum(M[i,:]*cubic_spline_kernel(R[i,:], H[i]))

    return rho_sph

@njit
def _generate_M(indices, m_enc):

    M = np.zeros(indices.shape)

    for i in range(M.shape[0]):
        M[i,:] = m_enc[indices[i]]

    return M
