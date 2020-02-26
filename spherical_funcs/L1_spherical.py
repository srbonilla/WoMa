#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:15:44 2019

@author: sergio
"""

import numpy as np
from numba import njit
import eos
from tqdm import tqdm
from T_rho import T_rho
from T_rho import set_T_rho_args
import glob_vars as gv

import warnings
warnings.filterwarnings("ignore")

@njit
def L1_integrate(num_prof, R, M, P_s, T_s, rho_s, mat_id, T_rho_type_id,
                 T_rho_args):
    """ Integration of a 1 layer spherical planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            mat_id (int):
                Material id.

            T_rho_type_id (int)
                Relation between A1_T and A1_rho to be used.

            T_rho_args (list):
                Extra arguments to determine the relation.

        Returns:
            A1_r ([float]):
                Array of radii (SI).

            A1_m_enc ([float]):
                Array of cumulative mass (SI).

            A1_P ([float]):
                Array of pressures (SI).

            A1_T ([float]):
                Array of temperatures (SI).

            A1_rho ([float]):
                Array of densities (SI).

            A1_u ([float]):
                Array of internal energy (SI).

            A1_mat_id ([float]):
                Array of material ids (SI).
    """
    # Initialise the profile arrays
    A1_r        = np.linspace(R, 0, int(num_prof))
    A1_m_enc    = np.zeros(A1_r.shape)
    A1_P        = np.zeros(A1_r.shape)
    A1_T        = np.zeros(A1_r.shape)
    A1_rho      = np.zeros(A1_r.shape)
    A1_u        = np.zeros(A1_r.shape)
    A1_mat_id   = np.ones(A1_r.shape) * mat_id

    u_s = eos.u_rho_T(rho_s, T_s, mat_id)
    # Set the T-rho relation parameters
    T_rho_args  = set_T_rho_args(T_s, rho_s, T_rho_type_id, T_rho_args,
                                 mat_id)

    dr = A1_r[0] - A1_r[1]

    # Set the surface values
    A1_m_enc[0] = M
    A1_P[0]     = P_s
    A1_T[0]     = T_s
    A1_rho[0]   = rho_s
    A1_u[0]     = u_s

    # Integrate inwards
    for i in range(1, A1_r.shape[0]):
        A1_m_enc[i] = A1_m_enc[i - 1] - 4*np.pi*A1_r[i - 1]**2*A1_rho[i - 1]*dr
        A1_P[i]     = A1_P[i - 1] + gv.G*A1_m_enc[i - 1]*A1_rho[i - 1]/(A1_r[i - 1]**2)*dr
        A1_rho[i]   = eos.find_rho(A1_P[i], mat_id, T_rho_type_id, T_rho_args, 
                                   A1_rho[i - 1], 1.1*A1_rho[i - 1])
        A1_T[i]     = T_rho(A1_rho[i], T_rho_type_id, T_rho_args, mat_id)
        A1_u[i]     = eos.u_rho_T(A1_rho[i], A1_T[i], mat_id)
        # Update the T-rho parameters
        if mat_id == gv.id_HM80_HHe and T_rho_type_id == gv.type_adb:
            T_rho_args  = set_T_rho_args(A1_T[i], A1_rho[i], T_rho_type_id, 
                                         T_rho_args, mat_id)
        
        # Stop if run out of mass
        if A1_m_enc[i] < 0:
            return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id

    return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id

@njit
def L1_find_mass(num_prof, R, M_max, P_s, T_s, rho_s, mat_id, T_rho_type_id,
                 T_rho_args):
    """ Finder of the total mass of the planet.
        The correct value yields m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M_max (float):
                Upper bound for the mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            mat_id (int):
                Material id.

            T_rho_type_id (int)
                Relation between A1_T and A1_rho to be used.

            T_rho_args (list):
                Extra arguments to determine the relation.

        Returns:
            M_max (float):
                Mass of the planet (SI).
    """
    M_min = 0.

    # Try integrating the profile with the maximum mass
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
        num_prof, R, M_max, P_s, T_s, rho_s, mat_id, T_rho_type_id, T_rho_args
        )

    if A1_m_enc[-1] < 0:
        raise Exception("M_max is too low, ran out of mass in first iteration.\nPlease increase M_max.\n")

    # Iterate the mass
    while np.abs(M_min - M_max) > 1e-8*M_min:
        M_try = (M_min + M_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
            num_prof, R, M_try, P_s, T_s, rho_s, mat_id,
            T_rho_type_id, T_rho_args)

        if A1_m_enc[-1] > 0.:
            M_max = M_try
        else:
            M_min = M_try

    return M_max

def L1_find_radius(num_prof, R_max, M, P_s, T_s, rho_s, mat_id, T_rho_type_id,
                   T_rho_args, num_attempt=40, verbose=1):
    """ Finder of the total radius of the planet.
        The correct value yields m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Maximuum radius of the planet (SI).

            M_max (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            mat_id (int):
                Material id.

            T_rho_type_id (int)
                Relation between A1_T and A1_rho to be used.

            T_rho_args (list):
                Extra arguments to determine the relation.

        Returns:
            M_max (float):
                Mass of the planet (SI).
    """
    R_min = 0.

    # Try integrating the profile with the minimum radius
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
        num_prof, R_max, M, P_s, T_s, rho_s, mat_id, T_rho_type_id, T_rho_args
        )

    if A1_m_enc[-1] != 0:
        raise Exception("R_max is too low, did not ran out of mass in first iteration.\nPlease increase R_max.\n")

    # Iterate the radius
    for i in tqdm(range(num_attempt), desc="Finding R given M", disable=(not verbose)):
        R_try = (R_min + R_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
            num_prof, R_try, M, P_s, T_s, rho_s, mat_id, T_rho_type_id, T_rho_args,
            )

        if A1_m_enc[-1] > 0.:
            R_min = R_try
        else:
            R_max = R_try
            
        if np.abs(R_min - R_max) < 1e-7*R_max:
            return R_min

    return R_min

