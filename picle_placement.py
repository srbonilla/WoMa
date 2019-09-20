#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:37:46 2019

@author: sergio
"""

import woma
import numpy as np
import utils_spin as us
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import eos
from tqdm import tqdm
import seagen
from T_rho import T_rho
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from numba import njit

#auxiliary functions
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
    M = us._generate_M(indices, m)
    rho_sph = us.SPH_density(M, distances, h)
    
    return rho_sph
    
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
    M = us.compute_spin_planet_M(r_array, rho_e, z_array, rho_p)
    
    # Equatorial radius
    Re = np.max(r_array[rho_e > 0])
    
    # First model - spherical planet from equatorial profile
    radii = np.arange(0, Re, Re/1000000)
    rho_e_model = interp1d(r_array, rho_e)
    densities = rho_e_model(radii)
    particles = seagen.GenSphere(N, radii[1:], densities[1:], verb=0)
    
    # Compute R, Z, rho and mass for every shell
    
    rho_p_model_inv = interp1d(rho_p, z_array)
    R = particles.A1_r.copy()
    rho_shell = rho_e_model(R)
    Z = rho_p_model_inv(rho_shell)
    
    R_shell = np.unique(R)
    Z_shell = np.unique(Z)
    rho_shell = np.unique(rho_shell)
    rho_shell = -np.sort(-rho_shell)
    N_shell_original = np.zeros_like(rho_shell, dtype='int')
    M_shell = np.zeros_like(rho_shell)
    
    for i in range(N_shell_original.shape[0]):
        N_shell_original[i] = int((R == R_shell[i]).sum())
       
    # Get picle mass of final configuration
    alpha = M/particles.A1_m.sum()
    m_picle = alpha*np.median(particles.A1_m)
    
    # Compute mass of every shell
    for i in range(M_shell.shape[0] - 1):
        if i == 0:
            M_shell[i] = 4*m_picle
            
        else:
            R_l = R_shell[i - 1]
            Z_l = Z_shell[i - 1]
            R_h = R_shell[i + 1]
            Z_h = Z_shell[i + 1]
            R_0 = R_shell[i]
            Z_0 = Z_shell[i]
            
            R_l = (R_l + R_0)/2
            Z_l = (Z_l + Z_0)/2
            R_h = (R_h + R_0)/2
            Z_h = (Z_h + Z_0)/2
            
            M_shell[i] = rho_shell[i]*V_theta(0, np.pi, [[R_l, Z_l], [R_h, Z_h]])
    
    # Last shell
    M_shell[-1] = M - M_shell.sum()
    
    # Number of particles per shell
    N_shell = np.round(M_shell/m_picle).astype(int)
    
    # Tweek mass picle per shell to match total mass
    m_picle_shell = M_shell/N_shell
    
    # n of theta for spherical shell
    N_theta_sph_model = 2000000
    particles = seagen.GenShell(N_theta_sph_model, 1)
    
    x = particles.A1_x
    y = particles.A1_y
    z = particles.A1_z
    r = np.sqrt(x**2 + y**2 + z**2)
    
    theta = np.arccos(z/r)
    theta_sph = np.sort(theta)
    
    assert len(theta) == len(theta_sph)
    
    n_theta_sph = np.arange(1, N_theta_sph_model + 1)/N_theta_sph_model
    
    # Generate shells and make adjustments
    A1_x = []
    A1_y = []
    A1_z = []
    A1_rho = []
    A1_m = []
    A1_R = []
    A1_Z = []
    
    theta_bins = np.linspace(0, np.pi, 10000)
    delta_theta = (theta_bins[1] - theta_bins[0])/2
    theta_elip = theta_bins[:-1] + delta_theta
    
    n_theta_elip = np.zeros_like(theta_bins)
    
    for i in tqdm(range(N_shell.shape[0] - 1), desc="Creating shells..."):
        
        # first layer
        if i == 0:
            particles = seagen.GenShell(N_shell[i], R_shell[i])
            A1_x.append(particles.A1_x)
            A1_y.append(particles.A1_y)
            A1_z.append(particles.A1_z*Z_shell[i]/R_shell[i])
            A1_rho.append(rho_shell[i]*np.ones(N_shell[i]))
            A1_m.append(m_picle_shell[i]*np.ones(N_shell[i]))
            A1_R.append(R_shell[i]*np.ones(N_shell[i]))
            A1_Z.append(Z_shell[i]*np.ones(N_shell[i]))
            
        else:
            
            # Create analitical model for the shell
            theta_elip = theta_bins[:-1] + delta_theta
            n_theta_elip = np.zeros_like(theta_bins)
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
            
            for j in range(theta_bins.shape[0] - 1):
                low = theta_bins[j]
                high = theta_bins[j + 1]
                
                # analitical solution particle number
                n_theta_elip[j] = rho_shell[i]*V_theta(low, high, shell_config)/m_picle_shell[i]
                
            for j in range(1, n_theta_elip.shape[0]):
                n_theta_elip[j] = n_theta_elip[j] + n_theta_elip[j - 1]
                
            n_theta_elip = n_theta_elip[:-1]/N_shell[i]
                
# =============================================================================
#             random_mask = np.random.binomial(1, 0.1, N_theta_sph_model) > 0
#             plt.figure()
#             plt.scatter(theta_sph[random_mask], n_theta_sph[random_mask],
#                         alpha = 0.5, label='spherical', s=5)
#             plt.scatter(theta_elip, n_theta_elip, alpha = 0.5, label='eliptical - theory', s=5)
#             plt.xlabel(r"$\theta$")
#             plt.ylabel(r"cumulative $n(\theta) [\%]$")
#             plt.legend()
#             plt.show()
# =============================================================================
            
            # Transfor theta acordingly
            n_theta_sph_model = interp1d(theta_sph, n_theta_sph)
            theta_elip_n_model = interp1d(n_theta_elip, theta_elip)
            
            x = particles.A1_x
            y = particles.A1_y
            z = particles.A1_z
            
            r, theta, phi = cart_to_spher(x, y, z)
            
            theta = theta_elip_n_model(n_theta_sph_model(theta))
            
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
            
    # last shell: use model from previous shell
    particles = seagen.GenShell(N_shell[-1], R_shell[-1])
    R_0 = R_shell[-1]
    Z_0 = Z_shell[-1]
    
    x = particles.A1_x
    y = particles.A1_y
    z = particles.A1_z
    
    r, theta, phi = cart_to_spher(x, y, z)
    
    theta = theta_elip_n_model(n_theta_sph_model(theta))
    
    x, y, z = spher_to_cart(r, theta, phi)
    alpha = np.sqrt(1/(x*x/R_0/R_0 + y*y/R_0/R_0 + z*z/Z_0/Z_0))
    x = alpha*x
    y = alpha*y
    z = alpha*z
    
    A1_x.append(x)
    A1_y.append(y)
    A1_z.append(z)
    
    A1_rho.append(rho_shell[i]*np.ones(N_shell[-1]))
    A1_m.append(m_picle_shell[i]*np.ones(N_shell[-1]))
    A1_R.append(R_shell[i]*np.ones(N_shell[-1]))
    A1_Z.append(Z_shell[i]*np.ones(N_shell[-1]))
            
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
    A1_vy = A1_z*wz
    
    return A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_R, A1_Z
    
def picle_placement_L1(r_array, rho_e, z_array, rho_p, Tw, N,
                       mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                       N_neig=48):
    """
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

            N (int):
                Number of particles.

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            N_neig (int):
                Number of neighbors in the SPH simulation.
                
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
                
            A1_u ([float]):
                Internal energy for every particle (SI).

            A1_P ([float]):
                Pressure for every particle (SI).
                
            A1_h ([float]):
                Smoothing lenght for every particle (SI).

            A1_mat_id ([int]):
                Material id for every particle.

            A1_id ([int]):
                Identifier for every particle
                
    """
    A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_R, A1_Z = \
        picle_placement(r_array, rho_e, z_array, rho_p, N, Tw)
        
    # internal energy
    A1_u = np.zeros((A1_m.shape[0]))

    A1_P = np.zeros((A1_m.shape[0],))

    for k in range(A1_m.shape[0]):
        T = T_rho(A1_rho[k], T_rho_type_L1, T_rho_args_L1, mat_id_L1)
        A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L1)
        A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L1)

    #print("Internal energy u computed\n")
    # Smoothing lengths, crudely estimated from the densities
    w_edge  = 2     # r/h at which the kernel goes to zero
    A1_h       = np.cbrt(N_neig*A1_m / (4/3*np.pi*A1_rho)) / w_edge

    A1_id     = np.arange(A1_m.shape[0])
    A1_mat_id = np.ones((A1_m.shape[0],))*mat_id_L1

    return A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_u, A1_P, \
           A1_h, A1_mat_id, A1_id

def picle_placement_L2(r_array, rho_e, z_array, rho_p, Tw, N, rho_i,
                       mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                       mat_id_L2, T_rho_type_L2, T_rho_args_L2,
                       N_neig=48):
    """
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

            N (int):
                Number of particles.
                
            rho_i (float):
                Density at the boundary between layers 1 and 2 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_L2 (int)
                Relation between T and rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            N_neig (int):
                Number of neighbors in the SPH simulation.
                
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
                
            A1_u ([float]):
                Internal energy for every particle (SI).

            A1_P ([float]):
                Pressure for every particle (SI).
                
            A1_h ([float]):
                Smoothing lenght for every particle (SI).

            A1_mat_id ([int]):
                Material id for every particle.

            A1_id ([int]):
                Identifier for every particle
                
    """
    A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_R, A1_Z = \
        picle_placement(r_array, rho_e, z_array, rho_p, N, Tw)
        
    # internal energy
    A1_u = np.zeros((A1_m.shape[0]))

    A1_P = np.zeros((A1_m.shape[0],))

    for k in range(A1_m.shape[0]):
        if A1_rho[k] > rho_i:
            T = T_rho(A1_rho[k], T_rho_type_L1, T_rho_args_L1, mat_id_L1)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L1)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L1)
        else:
            T = T_rho(A1_rho[k], T_rho_type_L2, T_rho_args_L2, mat_id_L2)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L2)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L2)

    #print("Internal energy u computed\n")
    # Smoothing lengths, crudely estimated from the densities
    w_edge  = 2     # r/h at which the kernel goes to zero
    A1_h       = np.cbrt(N_neig*A1_m / (4/3*np.pi*A1_rho)) / w_edge

    A1_id     = np.arange(A1_m.shape[0])
    A1_mat_id = (A1_rho > rho_i)*mat_id_L1 + (A1_rho <= rho_i)*mat_id_L2

    return A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_u, A1_P, \
           A1_h, A1_mat_id, A1_id
           
def picle_placement_L3(r_array, rho_e, z_array, rho_p, Tw, N, rho_12, rho_23,
                       mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                       mat_id_L2, T_rho_type_L2, T_rho_args_L2,
                       mat_id_L3, T_rho_type_L3, T_rho_args_L3,
                       N_neig=48):
    """
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

            N (int):
                Number of particles.
                
            rho_12 (float):
                Density at the boundary between layers 1 and 2 (SI).

            rho_23 (float):
                Density at the boundary between layers 2 and 3 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_L2 (int)
                Relation between T and rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_L3 (int)
                Relation between T and rho to be used in layer 3.

            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.

            N_neig (int):
                Number of neighbors in the SPH simulation.
                
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
                
            A1_u ([float]):
                Internal energy for every particle (SI).

            A1_P ([float]):
                Pressure for every particle (SI).
                
            A1_h ([float]):
                Smoothing lenght for every particle (SI).

            A1_mat_id ([int]):
                Material id for every particle.

            A1_id ([int]):
                Identifier for every particle
                
    """
    A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_R, A1_Z = \
        picle_placement(r_array, rho_e, z_array, rho_p, N, Tw)
        
    # internal energy
    A1_u = np.zeros((A1_m.shape[0]))

    A1_P = np.zeros((A1_m.shape[0],))

    for k in range(A1_m.shape[0]):
        if A1_rho[k] > rho_12:
            T = T_rho(A1_rho[k], T_rho_type_L1, T_rho_args_L1, mat_id_L1)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L1)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L1)

        elif A1_rho[k] > rho_23:
            T = T_rho(A1_rho[k], T_rho_type_L2, T_rho_args_L2, mat_id_L2)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L2)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L2)

        else:
            T = T_rho(A1_rho[k], T_rho_type_L3, T_rho_args_L3, mat_id_L3)
            A1_u[k] = eos._find_u(A1_rho[k], T, mat_id_L3)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L3)

    #print("Internal energy u computed\n")
    # Smoothing lengths, crudely estimated from the densities
    w_edge  = 2     # r/h at which the kernel goes to zero
    A1_h       = np.cbrt(N_neig*A1_m / (4/3*np.pi*A1_rho)) / w_edge

    A1_id     = np.arange(A1_m.shape[0])
    A1_mat_id = (A1_rho > rho_12)*mat_id_L1                       \
                + np.logical_and(A1_rho <= rho_12, A1_rho > rho_23)*mat_id_L2 \
                + (A1_rho < rho_23)*mat_id_L3

    return A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_u, A1_P, \
           A1_h, A1_mat_id, A1_id
           
# =============================================================================
# # Test  
# 
# R_earth = 6371000
# M_earth = 5.972E24
# 
# l1_test = woma.Planet(
#     name            = "prof_pE",
#     A1_mat_layer    = ['Til_granite'],
#     A1_T_rho_type   = [1],
#     A1_T_rho_args   = [[None, 0.]],
#     A1_R_layer      = [R_earth],
#     M               = 0.8*M_earth,
#     P_s             = 0,
#     T_s             = 300
#     )
# 
# l1_test.M_max = M_earth
# 
# l1_test.gen_prof_L1_fix_M_given_R()
# 
# l1_test_sp = woma.SpinPlanet(
#     name         = 'sp_planet',
#     planet       = l1_test,
#     Tw           = 3,
#     R_e          = 1.3*R_earth,
#     R_p          = 1.1*R_earth
#     )
# 
# l1_test_sp.spin()    
# 
# l2_test = woma.Planet(
#     name            = "prof_pE",
#     A1_mat_layer    = ['Til_iron', 'Til_granite'],
#     A1_T_rho_type   = [1, 1],
#     A1_T_rho_args   = [[None, 0.], [None, 0.]],
#     A1_R_layer      = [None, R_earth],
#     M               = M_earth,
#     P_s             = 0,
#     T_s             = 300
#     )
# 
# l2_test.gen_prof_L2_fix_R1_given_R_M()
# 
# l2_test_sp = woma.SpinPlanet(
#     name         = 'sp_planet',
#     planet       = l2_test,
#     Tw           = 2.6,
#     R_e          = 1.45*R_earth,
#     R_p          = 1.1*R_earth
#     )
# 
# l2_test_sp.spin()
# 
# r_array = l2_test_sp.A1_r_equator
# z_array = l2_test_sp.A1_r_pole
# rho_e = l2_test_sp.A1_rho_equator
# rho_p = l2_test_sp.A1_rho_pole
# N = 100000
# Tw = l2_test_sp.Tw
# 
# A1_x, A1_y, A1_z, A1_vx, A1_vy, A1_vz, A1_m, A1_rho, A1_R, A1_Z = \
#     picle_placement(r_array, rho_e, z_array, rho_p, N, Tw)
#     
# rho_sph = _SPH_density(A1_x, A1_y, A1_z, A1_m, 48)
#     
# delta_rho = (rho_sph - A1_rho)/A1_rho    
# rc = np.sqrt(A1_x**2 + A1_y**2)
# 
# plt.figure(figsize=(12, 12))
# plt.scatter(rc/R_earth, np.abs(A1_z)/R_earth, s = 40, alpha = 0.5, c = delta_rho, 
#             marker='.', edgecolor='none', cmap = 'coolwarm')
# plt.xlabel(r"$r_c$ $[R_{earth}]$")
# plt.ylabel(r"$z$ $[R_{earth}]$")
# cbar = plt.colorbar()
# cbar.set_label(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
# plt.clim(-0.1, 0.1)
# plt.axes().set_aspect('equal')
# plt.show()
#         
# 
# plt.figure()
# plt.scatter(A1_z/R_earth, delta_rho, s=1, alpha=0.5)
# plt.show()   
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    