#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:32:55 2019

@author: sergio
"""

import numpy as np
from scipy.interpolate import interp1d
from numba import njit
from tqdm import tqdm

from woma.spin_funcs import L1_spin
from woma.spin_funcs import utils_spin as us
from woma.eos import eos
from woma.eos.T_rho import T_rho


@njit
def _fillrho3(
    A1_r_equator,
    V_e,
    A1_r_pole,
    V_p,
    P_c,
    P_12,
    P_23,
    P_s,
    rho_c,
    rho_s,
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
    """ Compute densities of equatorial and polar profiles given the potential
        for a 3 layer planet.

        Args:

            A1_r_equator ([float]):
                Points at equatorial profile where the solution is defined (SI).

            V_e ([float]):
                Equatorial profile of potential (SI).

            A1_r_pole ([float]):
                Points at equatorial profile where the solution is defined (SI).

            V_p ([float]):
                Polar profile of potential (SI).

            P_c (float):
                Pressure at the center of the planet (SI).

            P_12 (float):
                Pressure at the boundary between layers 1 and 2 of the planet (SI).

            P_23 (float):
                Pressure at the boundary between layers 2 and 3 of the planet (SI).

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

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between T and rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_id_L3 (int)
                Relation between T and rho to be used in layer 3.

            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.


        Returns:

            A1_rho_equator ([float]):
                Equatorial profile of densities (SI).

            A1_rho_pole ([float]):
                Polar profile of densities (SI).
    """

    P_e = np.zeros(V_e.shape[0])
    P_p = np.zeros(V_p.shape[0])
    A1_rho_equator = np.zeros(V_e.shape[0])
    A1_rho_pole = np.zeros(V_p.shape[0])

    P_e[0] = P_c
    P_p[0] = P_c
    A1_rho_equator[0] = rho_c
    A1_rho_pole[0] = rho_c

    # equatorial profile
    for i in range(A1_r_equator.shape[0] - 1):
        gradV = V_e[i + 1] - V_e[i]
        gradP = -A1_rho_equator[i] * gradV
        P_e[i + 1] = P_e[i] + gradP

        # avoid overspin
        if P_e[i + 1] > P_e[i]:
            A1_rho_equator[i + 1 :] = A1_rho_equator[i]
            break

        # compute density
        if P_e[i + 1] >= P_s and P_e[i + 1] >= P_12:
            A1_rho_equator[i + 1] = eos.find_rho(
                P_e[i + 1],
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                rho_s * 0.1,
                A1_rho_equator[i],
            )

        elif P_e[i + 1] >= P_s and P_e[i + 1] >= P_23:
            A1_rho_equator[i + 1] = eos.find_rho(
                P_e[i + 1],
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                rho_s * 0.1,
                A1_rho_equator[i],
            )

        elif P_e[i + 1] >= P_s:
            A1_rho_equator[i + 1] = eos.find_rho(
                P_e[i + 1],
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                rho_s * 0.1,
                A1_rho_equator[i],
            )

        else:
            A1_rho_equator[i + 1] = 0.0
            break

    # polar profile
    for i in range(A1_r_pole.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -A1_rho_pole[i] * gradV
        P_p[i + 1] = P_p[i] + gradP

        # avoid overspin
        if P_e[i + 1] > P_e[i]:
            A1_rho_equator[i + 1 :] = A1_rho_equator[i]
            break

        # compute density
        if P_p[i + 1] >= P_s and P_p[i + 1] >= P_12:
            A1_rho_pole[i + 1] = eos.find_rho(
                P_p[i + 1],
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                rho_s * 0.1,
                A1_rho_pole[i],
            )

        elif P_p[i + 1] >= P_s and P_p[i + 1] >= P_23:
            A1_rho_pole[i + 1] = eos.find_rho(
                P_p[i + 1],
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                rho_s * 0.1,
                A1_rho_pole[i],
            )

        elif P_p[i + 1] >= P_s:
            A1_rho_pole[i + 1] = eos.find_rho(
                P_p[i + 1],
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                rho_s * 0.1,
                A1_rho_pole[i],
            )

        else:
            A1_rho_pole[i + 1] = 0.0
            break

    return A1_rho_equator, A1_rho_pole


def spin3layer(
    num_attempt,
    A1_r_equator,
    A1_rho_equator,
    A1_r_pole,
    A1_rho_pole,
    period,
    P_c,
    P_12,
    P_23,
    P_s,
    rho_c,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    verbose=1,
):
    """ Compute spining profile of densities for a 3 layer planet.

        Args:

            num_attempt (int):
                Number of num_attempt to run.

            A1_r_equator ([float]):
                Points at equatorial profile where the solution is defined (SI).

            A1_r_pole ([float]):
                Points at equatorial profile where the solution is defined (SI).

            radii ([float]):
                Radii of the spherical profile (SI).

            densities ([float]):
                Densities of the spherical profile (SI).

            period (float):
                Period of the planet (hours).

            P_c (float):
                Pressure at the center of the planet (SI).

            P_12 (float):
                Pressure at the boundary between layers 1 and 2 of the planet (SI).

            P_23 (float):
                Pressure at the boundary between layers 2 and 3 of the planet (SI).

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

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between T and rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_id_L3 (int)
                Relation between T and rho to be used in layer 3.

            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            vervose (int):
                Printing options.


        Returns:

            profile_e ([[float]]):
                List of the num_attempt of the equatorial density profile (SI).

            profile_p ([[float]]):
                List of the num_attempt of the polar density profile (SI).

    """

    profile_e = []
    profile_p = []

    profile_e.append(A1_rho_equator)
    profile_p.append(A1_rho_pole)

    for i in tqdm(
        range(num_attempt), desc="Solving spining profile", disable=(not verbose >= 1)
    ):
        V_e, V_p = L1_spin._fillV(
            A1_r_equator, A1_rho_equator, A1_r_pole, A1_rho_pole, period
        )
        A1_rho_equator, A1_rho_pole = _fillrho3(
            A1_r_equator,
            V_e,
            A1_r_pole,
            V_p,
            P_c,
            P_12,
            P_23,
            P_s,
            rho_c,
            rho_s,
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
        profile_e.append(A1_rho_equator)
        profile_p.append(A1_rho_pole)

    return profile_e, profile_p


def picle_placement_L3(
    A1_r_equator,
    A1_rho_equator,
    A1_r_pole,
    A1_rho_pole,
    period,
    N,
    rho_12,
    rho_23,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    N_neig=48,
):
    """
    Args:

            A1_r_equator ([float]):
                Points at equatorial profile where the solution is defined (SI).

            A1_rho_equator ([float]):
                Equatorial profile of densities (SI).

            A1_r_pole ([float]):
                Points at equatorial profile where the solution is defined (SI).

            A1_rho_pole ([float]):
                Polar profile of densities (SI).

            period (float):
                Period of the planet (hours).

            N (int):
                Number of particles.
                
            rho_12 (float):
                Density at the boundary between layers 1 and 2 (SI).

            rho_23 (float):
                Density at the boundary between layers 2 and 3 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between T and rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between T and rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_id_L3 (int)
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
    (
        A1_x,
        A1_y,
        A1_z,
        A1_vx,
        A1_vy,
        A1_vz,
        A1_m,
        A1_rho,
        A1_R,
        A1_Z,
    ) = us.picle_placement(A1_r_equator, A1_rho_equator, A1_r_pole, A1_rho_pole, N, period)

    # internal energy
    A1_u = np.zeros((A1_m.shape[0]))

    A1_P = np.zeros((A1_m.shape[0],))

    for k in range(A1_m.shape[0]):
        if A1_rho[k] > rho_12:
            T = T_rho(A1_rho[k], T_rho_type_id_L1, T_rho_args_L1, mat_id_L1)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L1)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L1)

        elif A1_rho[k] > rho_23:
            T = T_rho(A1_rho[k], T_rho_type_id_L2, T_rho_args_L2, mat_id_L2)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L2)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L2)

        else:
            T = T_rho(A1_rho[k], T_rho_type_id_L3, T_rho_args_L3, mat_id_L3)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L3)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L3)

    # print("Internal energy u computed\n")
    # Smoothing lengths, crudely estimated from the densities
    w_edge = 2  # r/h at which the kernel goes to zero
    A1_h = np.cbrt(N_neig * A1_m / (4 / 3 * np.pi * A1_rho)) / w_edge

    A1_id = np.arange(A1_m.shape[0])
    A1_mat_id = (
        (A1_rho > rho_12) * mat_id_L1
        + np.logical_and(A1_rho <= rho_12, A1_rho > rho_23) * mat_id_L2
        + (A1_rho < rho_23) * mat_id_L3
    )

    return (
        A1_x,
        A1_y,
        A1_z,
        A1_vx,
        A1_vy,
        A1_vz,
        A1_m,
        A1_rho,
        A1_u,
        A1_P,
        A1_h,
        A1_mat_id,
        A1_id,
    )
