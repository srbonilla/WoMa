#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:13:54 2019

@author: sergio
"""

import numpy as np
from scipy.interpolate import interp1d
from numba import njit, jit
from tqdm import tqdm

from woma.spin_funcs import utils_spin as us
from woma.eos import eos
from woma.eos.T_rho import T_rho


@jit(nopython=False)
def _fillV(A1_r_eq, A1_rho_eq, A1_z_po, A1_rho_po, period):
    """ Computes the potential at every point of the equatorial and polar profiles.

    Parameters
    ----------
    A1_r_eq ([float]):
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_eq ([float]):
        Equatorial profile of densities (SI).

    A1_z_po ([float]):
        Points at polar profile where the solution is defined (SI).

    A1_rho_po ([float]):
        Polar profile of densities (SI).

    period (float):
        Period of the planet (hours).

    Returns
    -------
    A1_V_eq ([float]):
        Equatorial profile of the potential (SI).

    A1_V_po ([float]):
        Polar profile of the potential (SI).
    """

    assert A1_r_eq.shape[0] == A1_rho_eq.shape[0]
    assert A1_z_po.shape[0] == A1_rho_po.shape[0]

    rho_model_po_inv = interp1d(A1_rho_po, A1_z_po)

    R_array = A1_r_eq
    Z_array = rho_model_po_inv(A1_rho_eq)

    A1_V_eq = np.zeros(A1_r_eq.shape)
    A1_V_po = np.zeros(A1_z_po.shape)

    W = 2 * np.pi / period / 60 / 60

    for i in range(A1_rho_eq.shape[0] - 1):

        if A1_rho_eq[i] == 0:
            break

        delta_rho = A1_rho_eq[i] - A1_rho_eq[i + 1]

        for j in range(A1_V_eq.shape[0]):
            A1_V_eq[j] += us._Vgr(A1_r_eq[j], R_array[i], Z_array[i], delta_rho)

        for j in range(A1_V_po.shape[0]):
            A1_V_po[j] += us._Vgz(A1_z_po[j], R_array[i], Z_array[i], delta_rho)

    for i in range(A1_V_eq.shape[0]):
        A1_V_eq[i] += -(1 / 2) * (W * A1_r_eq[i]) ** 2

    return A1_V_eq, A1_V_po


@njit
def _fillrho1(
    A1_r_eq,
    A1_V_eq,
    A1_z_po,
    A1_V_po,
    P_c,
    P_s,
    rho_c,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
):
    """ Compute densities of equatorial and polar profiles given the potential
        for a 1 layer planet.

    Parameters
    ----------
    A1_r_eq ([float]):
        Points at equatorial profile where the solution is defined (SI).

    A1_V_eq ([float]):
        Equatorial profile of potential (SI).

    A1_z_po ([float]):
        Points at equatorial profile where the solution is defined (SI).

    A1_V_po ([float]):
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

    Returns
    -------
    A1_rho_eq ([float]):
        Equatorial profile of densities (SI).

    A1_rho_po ([float]):
        Polar profile of densities (SI).
    """

    A1_P_eq = np.zeros(A1_V_eq.shape[0])
    A1_P_po = np.zeros(A1_V_po.shape[0])
    A1_rho_eq = np.zeros(A1_V_eq.shape[0])
    A1_rho_po = np.zeros(A1_V_po.shape[0])

    A1_P_eq[0] = P_c
    A1_P_po[0] = P_c
    A1_rho_eq[0] = rho_c
    A1_rho_po[0] = rho_c

    # equatorial profile
    for i in range(A1_r_eq.shape[0] - 1):
        gradV = A1_V_eq[i + 1] - A1_V_eq[i]
        gradP = -A1_rho_eq[i] * gradV
        A1_P_eq[i + 1] = A1_P_eq[i] + gradP

        # avoid overspin
        if A1_P_eq[i + 1] > A1_P_eq[i]:
            A1_rho_eq[i + 1 :] = A1_rho_eq[i]
            break

        # compute density
        if A1_P_eq[i + 1] >= P_s:
            A1_rho_eq[i + 1] = eos.find_rho(
                A1_P_eq[i + 1],
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                rho_s * 0.1,
                A1_rho_eq[i],
            )
        else:
            A1_rho_eq[i + 1] = 0.0
            break

    # polar profile
    for i in range(A1_z_po.shape[0] - 1):
        gradV = A1_V_po[i + 1] - A1_V_po[i]
        gradP = -A1_rho_po[i] * gradV
        A1_P_po[i + 1] = A1_P_po[i] + gradP

        # avoid overspin
        if A1_P_po[i + 1] > A1_P_po[i]:
            A1_rho_po[i + 1 :] = A1_rho_po[i]
            break

        # compute density
        if A1_P_po[i + 1] >= P_s:
            A1_rho_po[i + 1] = eos.find_rho(
                A1_P_po[i + 1],
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                rho_s * 0.1,
                A1_rho_po[i],
            )
        else:
            A1_rho_po[i + 1] = 0.0
            break

    return A1_rho_eq, A1_rho_po


def spin1layer(
    num_attempt,
    A1_r_eq,
    A1_rho_eq,
    A1_z_po,
    A1_rho_po,
    period,
    P_c,
    P_s,
    rho_c,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    verbosity=1,
):
    """ Compute spining profile of densities for a 1 layer planet.

    Parameters
    ----------
    num_attempt (int):
        Number of num_attempt to run.

    A1_r_eq ([float]):
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_eq ([float]):
        Densitity values at corresponding A1_r_eq points (SI).

    A1_z_po ([float]):
        Points at polar profile where the solution is defined (SI).

    A1_rho_po ([float]):
        Densitity values at corresponding A1_z_po points (SI).

    period (float):
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

    Returns
    -------
    profile_e ([[float]]):
        List of the num_attempt of the equatorial density profile (SI).

    profile_p ([[float]]):
        List of the num_attempt of the polar density profile (SI).

    """

    profile_e = []
    profile_p = []

    profile_e.append(A1_rho_eq)
    profile_p.append(A1_rho_po)

    for i in tqdm(
        range(num_attempt), desc="Solving spining profile", disable=verbosity == 0
    ):
        A1_V_eq, A1_V_po = _fillV(A1_r_eq, A1_rho_eq, A1_z_po, A1_rho_po, period)
        A1_rho_eq, A1_rho_po = _fillrho1(
            A1_r_eq,
            A1_V_eq,
            A1_z_po,
            A1_V_po,
            P_c,
            P_s,
            rho_c,
            rho_s,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
        )
        profile_e.append(A1_rho_eq)
        profile_p.append(A1_rho_po)

    return profile_e, profile_p


def picle_placement_L1(
    A1_r_eq,
    A1_rho_eq,
    A1_z_po,
    A1_rho_po,
    period,
    N,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    N_ngb=48,
):
    """ Particle placement for 1 layer spinning planet profile.
    Parameters
    ----------
    A1_r_eq ([float]):
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_eq ([float]):
        Equatorial profile of densities (SI).

    A1_z_po ([float]):
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_po ([float]):
        Polar profile of densities (SI).

    period (float):
        Period of the planet (hours).

    N (int):
        Number of particles.

    mat_id_L1 (int):
        Material id for layer 1.

    T_rho_type_id_L1 (int)
        Relation between T and rho to be used in layer 1.

    T_rho_args_L1 (list):
        Extra arguments to determine the relation in layer 1.

    N_ngb (int):
        Number of neighbors in the SPH simulation.
        
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
    ) = us.picle_placement(A1_r_eq, A1_rho_eq, A1_z_po, A1_rho_po, N, period)

    # internal energy
    A1_u = np.zeros((A1_m.shape[0]))

    A1_P = np.zeros((A1_m.shape[0],))

    for k in range(A1_m.shape[0]):
        T = T_rho(A1_rho[k], T_rho_type_id_L1, T_rho_args_L1, mat_id_L1)
        A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L1)
        A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L1)

    # Smoothing lengths, crudely estimated from the densities
    w_edge = 2  # r/h at which the kernel goes to zero
    A1_h = np.cbrt(N_ngb * A1_m / (4 / 3 * np.pi * A1_rho)) / w_edge

    A1_id = np.arange(A1_m.shape[0])
    A1_mat_id = np.ones((A1_m.shape[0],)) * mat_id_L1

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
