#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:13:54 2019

@author: sergio
"""

import numpy as np
import utils_spin as us
from scipy.interpolate import interp1d
from numba import njit, jit
import eos
from tqdm import tqdm
from T_rho import T_rho

@jit(nopython=False)
def _fillV(r_array, rho_e, z_array, rho_p, Tw):
    """ Computes the potential at every point of the equatorial and polar profiles.

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

            V_e ([float]):
                Equatorial profile of the potential (SI).

            V_p ([float]):
                Polar profile of the potential (SI).
    """

    if r_array.shape[0] != rho_e.shape[0] or z_array.shape[0] != rho_p.shape[0]:
        print("dimension error.\n")
        return -1, -1

    rho_p_model_inv = interp1d(rho_p, z_array)

    R_array = r_array
    Z_array = rho_p_model_inv(rho_e)

    V_e = np.zeros(r_array.shape)
    V_p = np.zeros(z_array.shape)

    W = 2*np.pi/Tw/60/60

    for i in range(rho_e.shape[0] - 1):

        if rho_e[i] == 0:
            break

        delta_rho = rho_e[i] - rho_e[i + 1]

        for j in range(V_e.shape[0]):
            V_e[j] += us._Vgr(r_array[j], R_array[i],
                              Z_array[i], delta_rho)

        for j in range(V_p.shape[0]):
            V_p[j] += us._Vgz(z_array[j], R_array[i],
                              Z_array[i], delta_rho)

    for i in range(V_e.shape[0]):
        V_e[i] += -(1/2)*(W*r_array[i])**2

    return V_e, V_p

@njit
def _fillrho1(r_array, V_e, z_array, V_p, P_c, P_s, rho_c, rho_s,
              mat_id_L1, T_rho_type_id_L1, T_rho_args_L1):
    """ Compute densities of equatorial and polar profiles given the potential
        for a 1 layer planet.

        Args:

            r_array ([float]):
                Points at equatorial profile where the solution is defined (SI).

            V_e ([float]):
                Equatorial profile of potential (SI).

            z_array ([float]):
                Points at equatorial profile where the solution is defined (SI).

            V_p ([float]):
                Polar profile of potential (SI).

            P_c (float):
                Pressure at the center of the planet (SI).

            P_s (float):
                Pressure at the surface of the planet (SI).

            rho_c (float):
                Density at the center of the planet (SI).

            rho_s (float):
                Density at the surface of the planet (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between T and rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

        Returns:

            rho_e ([float]):
                Equatorial profile of densities (SI).

            rho_p ([float]):
                Polar profile of densities (SI).
    """

    P_e = np.zeros(V_e.shape[0])
    P_p = np.zeros(V_p.shape[0])
    rho_e = np.zeros(V_e.shape[0])
    rho_p = np.zeros(V_p.shape[0])

    P_e[0] = P_c
    P_p[0] = P_c
    rho_e[0] = rho_c
    rho_p[0] = rho_c

    for i in range(r_array.shape[0] - 1):
        gradV = V_e[i + 1] - V_e[i]
        gradP = -rho_e[i]*gradV
        P_e[i + 1] = P_e[i] + gradP

        if P_e[i + 1] >= P_s:
            rho_e[i + 1] = eos.find_rho(
                P_e[i + 1], mat_id_L1, T_rho_type_id_L1, T_rho_args_L1,
                rho_s*0.1, rho_e[i]
                )
        else:
            rho_e[i + 1] = 0.
            break

    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP

        if P_p[i + 1] >= P_s:
            rho_p[i + 1] = eos.find_rho(
                P_p[i + 1], mat_id_L1, T_rho_type_id_L1, T_rho_args_L1, 
                rho_s*0.1, rho_p[i]
                )
        else:
            rho_p[i + 1] = 0.
            break

    return rho_e, rho_p

def spin1layer(num_attempt, r_array, z_array, radii, densities, Tw,
               P_c, P_s, rho_c, rho_s,
               mat_id_L1, T_rho_type_id_L1, T_rho_args_L1,
               verbose = 1
               ):
    """ Compute spining profile of densities for a 1 layer planet.

        Args:

            num_attempt (int):
                Number of num_attempt to run.

            r_array ([float]):
                Points at equatorial profile where the solution is defined (SI).

            z_array ([float]):
                Points at equatorial profile where the solution is defined (SI).

            radii ([float]):
                Radii of the spherical profile (SI).

            densities ([float]):
                Densities of the spherical profile (SI).

            Tw (float):
                Period of the planet (hours).

            P_c (float):
                Pressure at the center of the planet (SI).

            P_s (float):
                Pressure at the surface of the planet (SI).

            rho_c (float):
                Density at the center of the planet (SI).

            rho_s (float):
                Density at the surface of the planet (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between T and rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            vervose (int):
                Printing options.

        Returns:

            profile_e ([[float]]):
                List of the num_attempt of the equatorial density profile (SI).

            profile_p ([[float]]):
                List of the num_attempt of the polar density profile (SI).

    """

    spherical_model = interp1d(radii, densities, bounds_error=False, fill_value=0)

    rho_e = spherical_model(r_array)
    rho_p = spherical_model(z_array)

    profile_e = []
    profile_p = []

    profile_e.append(rho_e)
    profile_p.append(rho_p)

    for i in tqdm(range(num_attempt), desc="Solving spining profile", disable = (not verbose>=1)):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho1(r_array, V_e, z_array, V_p, P_c, P_s, rho_c, rho_s,
                                 mat_id_L1, T_rho_type_id_L1, T_rho_args_L1)
        profile_e.append(rho_e)
        profile_p.append(rho_p)

    return profile_e, profile_p

def picle_placement_L1(r_array, rho_e, z_array, rho_p, Tw, N,
                       mat_id_L1, T_rho_type_id_L1, T_rho_args_L1,
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

            T_rho_type_id_L1 (int)
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
        us.picle_placement(r_array, rho_e, z_array, rho_p, N, Tw)
        
    # internal energy
    A1_u = np.zeros((A1_m.shape[0]))

    A1_P = np.zeros((A1_m.shape[0],))

    for k in range(A1_m.shape[0]):
        T = T_rho(A1_rho[k], T_rho_type_id_L1, T_rho_args_L1, mat_id_L1)
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
     