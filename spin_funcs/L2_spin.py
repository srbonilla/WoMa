#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:27:14 2019

@author: sergio
"""

import numpy as np
import utils_spin as us
from scipy.interpolate import interp1d
from numba import njit
import eos
from tqdm import tqdm
import seagen
from T_rho import T_rho
from sklearn.neighbors import NearestNeighbors
import L1_spin

@njit
def _fillrho2(r_array, V_e, z_array, V_p, P_c, P_i, P_s, rho_c, rho_s,
             mat_id_L1, T_rho_type_L1, T_rho_args_L1,
             mat_id_L2, T_rho_type_L2, T_rho_args_L2
             ):
    """ Compute densities of equatorial and polar profiles given the potential
        for a 2 layer planet.

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

            P_i (float):
                Pressure at the boundary of the planet (SI).

            P_s (float):
                Pressure at the surface of the planet (SI).

            rho_c (float):
                Density at the center of the planet (SI).

            rho_s (float):
                Density at the surface of the planet (SI).

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

        if P_e[i + 1] >= P_s and P_e[i + 1] >= P_i:
            rho_e[i + 1] = eos.find_rho(
                P_e[i + 1], mat_id_L1, T_rho_type_L1, T_rho_args_L1, rho_s - 10,
                rho_e[i]
                )

        elif P_e[i + 1] >= P_s and P_e[i + 1] < P_i:
            rho_e[i + 1] = eos.find_rho(
                P_e[i + 1], mat_id_L2, T_rho_type_L2, T_rho_args_L2, rho_s - 10,
                rho_e[i]
                )

        else:
            rho_e[i + 1] = 0.
            break

    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP

        if P_p[i + 1] >= P_s and P_p[i + 1] >= P_i:
            rho_p[i + 1] = eos.find_rho(
                P_p[i + 1], mat_id_L1, T_rho_type_L1, T_rho_args_L1, rho_s - 10,
                rho_p[i]
                )

        elif P_p[i + 1] >= P_s and P_p[i + 1] < P_i:
            rho_p[i + 1] = eos.find_rho(
                P_p[i + 1], mat_id_L2, T_rho_type_L2, T_rho_args_L2, rho_s - 10,
                rho_p[i]
                )

        else:
            rho_p[i + 1] = 0.
            break

    return rho_e, rho_p

def spin2layer(num_attempt, r_array, z_array, radii, densities, Tw,
               P_c, P_i, P_s, rho_c, rho_s,
               mat_id_L1, T_rho_type_L1, T_rho_args_L1,
               mat_id_L2, T_rho_type_L2, T_rho_args_L2
               ):
    """ Compute spining profile of densities for a 2 layer planet.

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

            P_i (float):
                Pressure at the boundary of the planet (SI).

            P_s (float):
                Pressure at the surface of the planet (SI).

            rho_c (float):
                Density at the center of the planet (SI).

            rho_s (float):
                Density at the surface of the planet (SI).

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

    for i in tqdm(range(num_attempt), desc="Solving spining profile"):
        V_e, V_p = L1_spin._fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho2(r_array, V_e, z_array, V_p, P_c, P_i, P_s, rho_c, rho_s,
                                 mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                                 mat_id_L2, T_rho_type_L2, T_rho_args_L2)
        profile_e.append(rho_e)
        profile_p.append(rho_p)

    return profile_e, profile_p

def picle_placement_L2(r_array, rho_e, z_array, rho_p, Tw, N, rho_i,
                           mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                           mat_id_L2, T_rho_type_L2, T_rho_args_L2,
                           u_cold_array_L1, u_cold_array_L2, N_neig=48,
                           num_attempt=10):
    """ Particle placement for a 2 layer spining profile.

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
                Density at the boundary (SI).

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

            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).

            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).

            N_neig (int):
                Number of neighbors in the SPH simulation.

        Returns:

            x ([float]):
                Position x of each particle (SI).

            y ([float]):
                Position y of each particle (SI).

            z ([float]):
                Position z of each particle (SI).

            vx ([float]):
                Velocity in x of each particle (SI).

            vy ([float]):
                Velocity in y of each particle (SI).

            vz ([float]):
                Velocity in z of each particle (SI).

            m ([float]):
                Mass of every particle (SI).

            h ([float]):
                Smoothing lenght for every particle (SI).

            rho ([float]):
                Density for every particle (SI).

            P ([float]):
                Pressure for every particle (SI).

            u ([float]):
                Internal energy for every particle (SI).

            mat_id ([int]):
                Material id for every particle.

            id ([int]):
                Identifier for every particle

    """

    rho_e_model = interp1d(r_array, rho_e)
    rho_p_model_inv = interp1d(rho_p, z_array)

    Re = np.max(r_array[rho_e > 0])

    radii = np.arange(0, Re, Re/1000000)
    densities = rho_e_model(radii)

    particles = seagen.GenSphere(N, radii[1:], densities[1:], verb=0)

    particles_r = np.sqrt(particles.x**2 + particles.y**2 + particles.z**2)
    rho = rho_e_model(particles_r)

    R = particles.A1_r.copy()
    rho_layer = rho_e_model(R)
    Z = rho_p_model_inv(rho_layer)

    f = Z/R
    zP = particles.z*f

    mP = particles.m*f

    # Compute velocities (T_w in hours)
    vx = np.zeros(mP.shape[0])
    vy = np.zeros(mP.shape[0])
    vz = np.zeros(mP.shape[0])

    hour_to_s = 3600
    wz = 2*np.pi/Tw/hour_to_s

    vx = -particles.y*wz
    vy = particles.x*wz

    # internal energy
    u = np.zeros((mP.shape[0]))

    x = particles.x
    y = particles.y

    P = np.zeros((mP.shape[0],))

    for k in range(mP.shape[0]):
        if rho[k] > rho_i:
            T = T_rho(rho[k], T_rho_type_L1, T_rho_args_L1, mat_id_L1)
            u[k] = eos.u_rho_T(rho[k], T, mat_id_L1)
            P[k] = eos.P_u_rho(u[k], rho[k], mat_id_L1)
        else:
            T = T_rho(rho[k], T_rho_type_L2, T_rho_args_L2, mat_id_L2)
            u[k] = eos.u_rho_T(rho[k], T, mat_id_L2)
            P[k] = eos.P_u_rho(u[k], rho[k], mat_id_L2)

    #print("Internal energy u computed\n")

    # Smoothing lengths, crudely estimated from the densities
    num_ngb = N_neig    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge

    A1_id = np.arange(mP.shape[0])
    A1_mat_id = (rho > rho_i)*mat_id_L1 + (rho <= rho_i)*mat_id_L2

    unique_R_L1   = np.unique(R[A1_mat_id == mat_id_L1])
    unique_R_L2 = np.unique(R[A1_mat_id == mat_id_L2])

    x_reshaped  = x.reshape((-1,1))
    y_reshaped  = y.reshape((-1,1))
    zP_reshaped = zP.reshape((-1,1))

    X = np.hstack((x_reshaped, y_reshaped, zP_reshaped))

    del x_reshaped, y_reshaped, zP_reshaped

    nbrs = NearestNeighbors(n_neighbors=N_neig, algorithm='kd_tree', metric='euclidean', leaf_size=15)
    nbrs.fit(X)

    N_mem = int(1e6)

    if particles.N_picle < N_mem:

        print("Finding neighbors of all particles...")
        distances, indices = nbrs.kneighbors(X)

        for _ in tqdm(range(num_attempt), desc="Tweaking mass of every particle"):

            M = us._generate_M(indices, mP)

            rho_sph = us.SPH_density(M, distances, h)

            diff = (rho_sph - rho)/rho
            mP_next = (1 - diff)*mP
            # do not change values of inter-boundary layers
            mP_next[R == unique_R_L1[-1]]   = mP[R == unique_R_L1[-1]]  # outer layer 1
            mP_next[R == unique_R_L2[0]]    = mP[R == unique_R_L2[0]]   # inner layer 2
            mP_next[R == unique_R_L2[-1]]   = mP[R == unique_R_L2[-1]]  # outer layer 2

            mP = mP_next

    else:

        k    = particles.N_picle // N_mem

        for _ in tqdm(range(num_attempt), desc="Tweaking mass of every particle"):

            mP_prev = mP.copy()

            for i in range(int(k)):

                distances_i, indices_i = nbrs.kneighbors(X[i*N_mem:(i + 1)*N_mem,:])

                M_i  = us._generate_M(indices_i, mP_prev)

                rho_sph_i = us.SPH_density(M_i, distances_i, h[i*N_mem:(i + 1)*N_mem])

                diff_i = (rho_sph_i - rho[i*N_mem:(i + 1)*N_mem])/rho[i*N_mem:(i + 1)*N_mem]
                mP_next_i = (1 - diff_i)*mP[i*N_mem:(i + 1)*N_mem]
                # do not change values of inter-boundary layers
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L1[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L1[-1]]  # outer layer 1
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[0]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[0]]   # inner layer 2
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[-1]]  # outer layer 2

                mP[i*N_mem:(i + 1)*N_mem] = mP_next_i

            distances_k, indices_k = nbrs.kneighbors(X[k*N_mem:,:])

            M_k  = us._generate_M(indices_k, mP_prev)

            rho_sph_k = us.SPH_density(M_k, distances_k, h[k*N_mem:])

            diff_k = (rho_sph_k - rho[k*N_mem:])/rho[k*N_mem:]
            mP_next_k = (1 - diff_k)*mP[k*N_mem:]
            # do not change values of inter-boundary layers
            mP_next_k[R[k*N_mem:] == unique_R_L1[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L1[-1]]    # outer layer 1
            mP_next_k[R[k*N_mem:] == unique_R_L2[0]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L2[0]]     # inner layer 2
            mP_next_k[R[k*N_mem:] == unique_R_L2[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L2[-1]]    # outer layer 2

            mP[k*N_mem:] = mP_next_k

# =============================================================================
#     ######
#     import matplotlib.pyplot as plt
#
#     diff = (rho_sph - rho)/rho
#     fig, ax = plt.subplots(1,2, figsize=(12,6))
#     ax[0].hist(diff, bins = 500)
#     ax[0].set_xlabel(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
#     ax[0].set_ylabel('Counts')
#     ax[0].set_yscale("log")
#     ax[1].scatter(zP/R_earth, diff, s = 0.5, alpha=0.5)
#     ax[1].set_xlabel(r"z [$R_{earth}$]")
#     ax[1].set_ylabel(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
#     #ax[1].set_ylim(-0.03, 0.03)
#     plt.tight_layout()
#     plt.show()
# =============================================================================
    #####

    print("\nDone!")

    return x, y, zP, vx, vy, vz, mP, h, rho, P, u, A1_mat_id, A1_id
