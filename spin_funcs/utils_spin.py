#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:10:28 2019

@author: sergio
"""

import numpy as np
from numba import njit, jit
import glob_vars as gv
from scipy.interpolate import interp1d
import seagen
import scipy.integrate as integrate
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import L1_spin

# Spining model functions
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

@jit(nopython=False)
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

@njit
def V_spheroid(R, Z):
    
    return np.pi*4/3*R*R*Z

@jit(nopython=False)
def compute_spin_planet_M(r_array, rho_e, z_array, rho_p):
    
    rho_p_model_inv = interp1d(rho_p, z_array)
    R_array = r_array
    Z_array = rho_p_model_inv(rho_e)
    
    M = 0.
    
    for i in range(1, R_array.shape[0]):
        dV = V_spheroid(R_array[i], Z_array[i]) -  \
             V_spheroid(R_array[i - 1], Z_array[i - 1])
        M += rho_e[i]*dV
        
    return M

# Particle placement functions
@njit
def vol_spheroid(R, Z):
    return 4*np.pi*R*R*Z/3

@njit
def integrand(theta, R_l, Z_l, R_h, Z_h):
    
    r_h = np.sin(theta)**2/R_h/R_h + np.cos(theta)**2/Z_h/Z_h
    r_h = np.sqrt(1/r_h)
    
    r_l = np.sin(theta)**2/R_l/R_l + np.cos(theta)**2/Z_l/Z_l
    r_l = np.sqrt(1/r_l)
    
    I = 2*np.pi*(r_h**3 - r_l**3)*np.sin(theta)/3
    
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
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    
    return r, theta, phi

@njit
def spher_to_cart(r, theta, phi):
    
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    
    return x, y ,z

def _SPH_density(x, y, z, m, N_neig):
    
    x_reshaped = x.reshape((-1,1))
    y_reshaped = y.reshape((-1,1))
    z_reshaped = z.reshape((-1,1))
    
    X = np.hstack((x_reshaped, y_reshaped, z_reshaped))
    
    del x_reshaped, y_reshaped, z_reshaped
    
    nbrs = NearestNeighbors(n_neighbors=N_neig, algorithm='kd_tree', metric='euclidean', leaf_size=15)
    nbrs.fit(X)
    
    distances, indices = nbrs.kneighbors(X)
    
    w_edge = 2
    h = np.max(distances, axis=1)/w_edge
    M = _generate_M(indices, m)
    rho_sph = SPH_density(M, distances, h)
    
    return rho_sph

@njit
def _i(theta, R, Z):
    
    i = - np.sqrt(2)*R*R*np.cos(theta)
    i = i/np.sqrt(1/R/R + 1/Z/Z + (-1/R/R + 1/Z/Z)*np.cos(2*theta))
    i = i + R*R*Z
    
    return i


def _V_theta_analytical(theta, shell_config):
    
    Rm1 = shell_config[0][0]
    Zm1 = shell_config[0][1]
    R1 = shell_config[1][0]
    Z1 = shell_config[1][1]
    
    V = _i(theta, R1, Z1) - _i(theta, Rm1, Zm1)
    V = V/(_i(np.pi, R1, Z1) - _i(np.pi, Rm1, Zm1))
    
    return V

# =============================================================================
# def compute_M_shell(R_shell, Z_shell, r_array, rho_e, z_array, rho_p):
#     
#     M_shell = np.zeros_like(R_shell)
#     Re = np.max(r_array[rho_e > 0])
#     Rp = np.max(z_array[rho_p > 0])
#     
#     rho_e_model     = interp1d(r_array, rho_e)
#     rho_p_model_inv = interp1d(rho_p, z_array)
#     
#     for i in range(M_shell.shape[0]):
#         if i == 0:
#             
#             R_l = 1e-5
#             Z_l = 1e-5
#             R_h = R_shell[i + 1]
#             Z_h = Z_shell[i + 1]
#             R_0 = R_shell[i]
#             Z_0 = Z_shell[i]
#             
#             R_h = (R_h + R_0)/2
#             Z_h = (Z_h + Z_0)/2
#             
#             # set all Zs with model
#             
#         elif i == M_shell.shape[0] - 1:
#             
#             R_l = R_shell[i - 1]
#             Z_l = Z_shell[i - 1]
#             R_h = Re
#             Z_h = Rp
#             R_0 = R_shell[i]
#             Z_0 = Z_shell[i]
#             
#             R_l = (R_l + R_0)/2
#             Z_l = (Z_l + Z_0)/2
#             
#         else:
#             
#             R_l = R_shell[i - 1]
#             Z_l = Z_shell[i - 1]
#             R_h = R_shell[i + 1]
#             Z_h = Z_shell[i + 1]
#             R_0 = R_shell[i]
#             Z_0 = Z_shell[i]
#             
#             R_l = (R_l + R_0)/2
#             Z_l = (Z_l + Z_0)/2
#             R_h = (R_h + R_0)/2
#             Z_h = (Z_h + Z_0)/2
#             
#         rho_e_temp = np.copy(rho_e)
#         rho_p_temp = np.copy(rho_p)
#         
#         rho_e_temp[r_array > R_h] = 0
#         rho_p_temp[z_array > Z_h] = 0
#         
#         M_shell[i] = compute_spin_planet_M(r_array, rho_e_temp,
#                                            z_array, rho_p_temp)
#         
#         rho_e_temp = np.copy(rho_e)
#         rho_p_temp = np.copy(rho_p)
#         
#         rho_e_temp[r_array > R_l] = 0
#         rho_p_temp[z_array > Z_l] = 0
#         
#         M_shell[i] = M_shell[i] - \
#             compute_spin_planet_M(r_array, rho_e_temp,
#                                   z_array, rho_p_temp)
#     return M_shell
# =============================================================================

@jit(nopython=False)
def compute_M_array(r_array, rho_e, z_array, rho_p):
    
    rho_p_model_inv = interp1d(rho_p, z_array)
    R_array = r_array
    Z_array = rho_p_model_inv(rho_e)
    
    M = np.zeros_like(R_array)
    
    for i in range(1, R_array.shape[0]):
        dV = V_spheroid(R_array[i], Z_array[i]) -  \
             V_spheroid(R_array[i - 1], Z_array[i - 1])
        M[i] = rho_e[i]*dV
        
    return M
    
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
            R_h = (R_h + R_0)/2
            
        elif i == M_shell.shape[0] - 1:
            
            R_l = R_shell[i - 1]
            R_h = Re
            R_0 = R_shell[i]
            R_l = (R_l + R_0)/2
            
        else:
            
            R_l = R_shell[i - 1]
            R_h = R_shell[i + 1]
            R_0 = R_shell[i]
            R_l = (R_l + R_0)/2
            R_h = (R_h + R_0)/2
            
        M_shell[i] = M_cum_model(R_h) - M_cum_model(R_l)            
        
    return M_shell

# =============================================================================
# def compute_M_shell(R_shell, R_shell_outer, r_array, rho_e, z_array, rho_p):
#     
#     M_shell = np.zeros_like(R_shell)
#     Re = np.max(r_array[rho_e > 0])
#     Rp = np.max(z_array[rho_p > 0])
#     
#     rho_e_model     = interp1d(r_array, rho_e)
#     rho_p_model_inv = interp1d(rho_p, z_array)
#     
#     for i in range(M_shell.shape[0]):
#         if i == 0:
#             
#             R_l = 1e-5
#             R_h = R_shell_outer[i]
#             
#             Z_l = rho_p_model_inv(rho_e_model(R_l))
#             Z_h = rho_p_model_inv(rho_e_model(R_l))
#             
#         elif i == M_shell.shape[0] - 1:
#             
#             R_l = R_shell_outer[i - 1]
#             R_h = Re
#             
#             Z_l = rho_p_model_inv(rho_e_model(R_l))
#             Z_h = Rp
#             
#         else:
#             
#             R_l = R_shell_outer[i - 1]
#             R_h = R_shell_outer[i]
#             
#             Z_l = rho_p_model_inv(rho_e_model(R_l))
#             Z_h = rho_p_model_inv(rho_e_model(R_l))
#             
#         rho_e_temp = np.copy(rho_e)
#         rho_p_temp = np.copy(rho_p)
#         
#         rho_e_temp[r_array > R_h] = 0
#         rho_p_temp[z_array > Z_h] = 0
#         
#         M_shell[i] = compute_spin_planet_M(r_array, rho_e_temp,
#                                            z_array, rho_p_temp)
#         
#         rho_e_temp = np.copy(rho_e)
#         rho_p_temp = np.copy(rho_p)
#         
#         rho_e_temp[r_array > R_l] = 0
#         rho_p_temp[z_array > Z_l] = 0
#         
#         M_shell[i] = M_shell[i] - \
#             compute_spin_planet_M(r_array, rho_e_temp,
#                                   z_array, rho_p_temp)
#     return M_shell
# =============================================================================

# main function 
def picle_placement(r_array, rho_e, z_array, rho_p, N, Tw):
    
    """ Particle placement for a spining profile.

        Args:

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
                
            Tw (float):
                Period of the planet (hours).

        Returns:

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
                Semi-major axis of the elipsoid for every particle
                
            A1_Z ([float]):
                Semi-minor axis of the elipsoid for every particle

    """
    
    assert len(r_array) == len(rho_e)
    assert len(z_array) == len(rho_p)
    
    # mass of the model planet
    M = compute_spin_planet_M(r_array, rho_e, z_array, rho_p)
    
    # Equatorial and polar radius radius
    Re = np.max(r_array[rho_e > 0])
    Rp = np.max(z_array[rho_p > 0])
    
    # First model - spherical planet from equatorial profile
    radii = np.arange(0, Re, Re/1000000)
    rho_e_model = interp1d(r_array, rho_e)
    densities = rho_e_model(radii)
    particles = seagen.GenSphere(N, radii[1:], densities[1:], verb=0)
    
    rho_p_model_inv = interp1d(rho_p, z_array)
    
    R_shell       = np.unique(particles.A1_r)
    #R_shell_outer = particles.A1_r_outer.copy()
    rho_shell     = rho_e_model(R_shell)
    Z_shell       = rho_p_model_inv(rho_shell)
       
    # Get picle mass of final configuration
    m_picle = M/N
        
    # return M_shell, M_shell_model
    #M_shell = compute_M_shell(R_shell, Z_shell,
    #                          r_array, rho_e,
    #                          z_array, rho_p)
    
    M_shell = compute_M_shell(R_shell, r_array, rho_e, z_array, rho_p)
    
    print(M_shell, flush=True)
    
    # Number of particles per shell
    N_shell = np.round(M_shell/m_picle).astype(int)
    
    # Tweek mass picle per shell to match total mass
    m_picle_shell = M_shell/N_shell
    
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
            R_h = (R_h + R_0)/2
            Z_h = (Z_h + Z_0)/2
            
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
                    
                R_l = (R_l + R_0)/2
                Z_l = (Z_l + Z_0)/2
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
                
            R_l = (R_l + R_0)/2
            Z_l = (Z_l + Z_0)/2
            R_h = (R_h + R_0)/2
            Z_h = (Z_h + Z_0)/2
            
            shell_config = [[R_l, Z_l], [R_h, Z_h]]
            
            n_theta_elip = _V_theta_analytical(theta_elip, shell_config)
            
        # Transfor theta acordingly
        theta_elip_n_model = interp1d(n_theta_elip, theta_elip)
        
        x = particles.A1_x
        y = particles.A1_y
        z = particles.A1_z
        
        r, theta, phi = cart_to_spher(x, y, z)
        
        theta = theta_elip_n_model((1 - np.cos(theta))/2)
        
        x, y, z = spher_to_cart(r, theta, phi)
        
        # Project on the spheroid without changing theta
        alpha = np.sqrt(1/(x*x/R_0/R_0 + y*y/R_0/R_0 + z*z/Z_0/Z_0))
        x = alpha*x
        y = alpha*y
        z = alpha*z
        
        # Save results
        A1_x.append(x)
        A1_y.append(y)
        A1_z.append(z)

        A1_rho.append(rho_shell[i]*np.ones(N_shell[i]))
        A1_m.append(m_picle_shell[i]*np.ones(N_shell[i]))
        A1_R.append(R_shell[i]*np.ones(N_shell[i]))
        A1_Z.append(Z_shell[i]*np.ones(N_shell[i]))
            
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
    wz = 2*np.pi/Tw/hour_to_s

    A1_vx = -A1_y*wz
    A1_vy = A1_x*wz
    
    return A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_R, A1_Z

def spin_escape_vel(r_array, rho_e, z_array, rho_p, Tw):
    """
        Computes the escape velocity for a spining planet.
        
        Args:

            r_array ([float]):
                Points at equatorial profile where the solution is defined (SI).

            rho_e ([float]):
                Equatorial profile of densities (SI).

            z_array ([float]):
                Points at equatorial profile where the solution is defined (SI).

            rho_p ([float]):
                Polar profile of densities (SI).
                
            Tw (float):
                Period of the planet (hours).

        Returns:

            v_escape_equator ([float]):
                Escape velocity at the equator (SI).

            v_escape_pole ([float]):
                Escape velocity at the pole (SI).

        
    """
    V_e, V_p = L1_spin._fillV(r_array, rho_e,
                              z_array, rho_p, Tw)
        
    i_equator = min(np.where(rho_e == 0)[0]) - 1
    i_pole    = min(np.where(rho_p == 0)[0]) - 1
    V_equator = V_e[i_equator]
    V_pole    = V_p[i_pole]
    v_escape_pole    = np.sqrt(-2*V_pole)
    w = 2*np.pi/Tw/60/60
    R_e = r_array[i_equator]
    v_escape_equator = np.sqrt(-2*V_equator - (w*R_e)**2)
    
    return v_escape_equator, v_escape_pole