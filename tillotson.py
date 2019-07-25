#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 08:48:59 2019

@author: sergio
"""

import numpy as np
from numba import njit
import os
import sys
import glob_vars as gv

# Go to the WoMa directory
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
sys.path.append(dir)


def load_u_cold_array(mat_id):
    """ Load precomputed values of cold internal energy for a given material.

        Returns:
            u_cold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() (SI).
    """
    if mat_id == gv.id_Til_iron:
        u_cold_array = np.load(gv.Fp_u_cold_Til_iron)
    elif mat_id == gv.id_Til_granite:
        u_cold_array = np.load(gv.Fp_u_cold_Til_granite)
    elif mat_id == gv.id_Til_water:
        u_cold_array = np.load(gv.Fp_u_cold_Til_water)
    else:
        raise ValueError("Invalid material ID")

    return u_cold_array

# load u cold arrays
if os.path.isfile(gv.Fp_u_cold_Til_iron):
    A1_u_cold_iron = load_u_cold_array(gv.id_Til_iron)
if os.path.isfile(gv.Fp_u_cold_Til_granite):
    A1_u_cold_granite = load_u_cold_array(gv.id_Til_granite)
if os.path.isfile(gv.Fp_u_cold_Til_water):
    A1_u_cold_water = load_u_cold_array(gv.id_Til_water)

@njit
def P_u_rho(u, rho, mat_id):
    """ Computes pressure for Tillotson EoS.

        Args:
            u (double)
                Specific internal energy (SI).

            rho (double)
                Density (SI).

            mat_id (int)
                Material id.

        Returns:
            P (double)
                Pressure (SI).
    """
    
    # Material constants for Tillotson EoS
    # mat_id, rho0, a, b, A, B, u_0, u_iv, u_cv, alpha, beta, eta_min, P_min, eta_zero
    iron    = np.array([gv.id_Til_iron, 7800, 0.5, 1.5, 1.28e11, 1.05e11, 9.5e9, 2.4e9, 8.67e9, 5, 5, 0, 0, 0])
    granite = np.array([gv.id_Til_granite, 2680, 0.5, 1.3, 1.8e10, 1.8e10, 1.6e10, 3.5e9, 1.8e10, 5, 5, 0, 0, 0])
    water   = np.array([gv.id_Til_water, 998, 0.7, 0.15, 2.18e9, 1.325e10, 7.0e9, 4.19e8, 2.69e9, 10, 5, 0.925, 0, 0.875])

    if (mat_id == gv.id_Til_iron):
        material = iron
    elif (mat_id == gv.id_Til_granite):
        material = granite
    elif (mat_id == gv.id_Til_water):
        material = water
    else:
        raise ValueError("Invalid material ID")

    rho0     = material[1]
    a        = material[2]
    b        = material[3]
    A        = material[4]
    B        = material[5]
    u_0      = material[6]
    u_iv     = material[7]
    u_cv     = material[8]
    alpha    = material[9]
    beta     = material[10]
    eta_min  = material[11]
    P_min    = material[12]
    eta_zero = material[13]

    eta      = rho/rho0
    eta_sq   = eta*eta
    mu       = eta - 1.
    nu       = 1./eta - 1.
    w        = u/(u_0*eta_sq) + 1.
    w_inv    = 1./w

    P_c = 0.
    P_e = 0.
    P   = 0.

    # Condensed or cold
    P_c = (a + b*w_inv)*rho*u + A*mu + B*mu*mu

    if eta < eta_zero:
        P_c = 0.
    elif eta < eta_min:
        P_c *= (eta - eta_zero) / (eta_min - eta_zero)

    # Expanded and hot
    P_e = a*rho*u + (b*rho*u*w_inv + A*mu*np.exp(-beta*nu))                   \
                     *np.exp(-alpha*nu*nu)


    # Condensed or cold state
    if (1. < eta) or (u < u_iv):
        P = P_c

    # Expanded and hot state
    elif ((eta < 1) and (u_cv < u)):
        P = P_e

    # Hybrid state
    else:
        P = ((u - u_iv) * P_e + (u_cv - u) * P_c) /                           \
                                (u_cv - u_iv)

    # Minimum pressure
    if (P < P_min):
        P = P_min

    return P

@njit
def C_V(mat_id):
    """ Returns specific heat capacity for a given material id (SI)

        Args:
            mat_id (int)
                Material id.

        Returns:
            C_V (double)
                Specific heat capacity (SI).

    """
    if mat_id == gv.id_Til_iron:
        return 449.
    elif mat_id == gv.id_Til_granite:
        return 790.
    elif mat_id == gv.id_Til_water:
        return 4186.
    else:
        raise ValueError("Invalid material ID")
        
@njit
def _rho_0_material(mat_id):
    """ Returns rho0 for a given material id. u_{cold}(rho0) = 0

        Args:
            mat_id (int)
                Material id.

        Returns:
            rho0 (double)
                Density (SI).

    """
    if (mat_id == gv.id_Til_iron):
        return 7800.
    elif (mat_id == gv.id_Til_granite):
        return 2680.
    elif (mat_id == gv.id_Til_water):
        return 998.
    else:
        raise ValueError("Invalid material ID")
        
@njit
def u_cold(rho, mat_id, N):
    """ Computes internal energy cold.

        Args:
            rho (float)
                Density (SI).

            mat_id (int)
                Material id.

            N (int)
                Number of subdivisions for the numerical integral.

        Returns:
            u_cold (float)
                Cold Specific internal energy (SI).
    """
    assert(rho >= 0)
    mat_type    = mat_id // gv.type_factor
    if (mat_type == gv.type_Til):
        rho0 = _rho_0_material(mat_id)
        drho = (rho - rho0)/N
        x = rho0
        u_cold = 1e-9

        for j in range(N):
            x += drho
            u_cold += P_u_rho(u_cold, x, mat_id)*drho/x**2
            
    else:
        raise ValueError("Invalid material ID")

    return u_cold

@njit
def _create_u_cold_array(mat_id):
    """ Computes values of the cold internal energy and stores it to save
        computation time in future calculations.
        It ranges from density = 100 kg/m^3 to 100000 kg/m^3

        Args:
            mat_id (int):
                Material id.

        Returns:
            u_cold_array ([float])
    """
    N_row = 10000
    u_cold_array = np.zeros((N_row,))
    rho_min = 100
    rho_max = 100000
    N_u_cold = 10000

    rho = rho_min
    drho = (rho_max - rho_min)/(N_row - 1)

    for i in range(N_row):
        u_cold_array[i] = u_cold(rho, mat_id, N_u_cold)
        rho = rho + drho

    return u_cold_array

@njit
def u_cold_tab(rho, mat_id):
    """ Fast computation of cold internal energy using the table previously
        computed.

        Args:
            rho (float):
                Density (SI).

            mat_id (int):
                Material id.

        Returns:
            interpolation (float):
                cold Specific internal energy (SI).
    """
    
    if mat_id == gv.id_Til_iron:
        u_cold_array = A1_u_cold_iron
    elif mat_id == gv.id_Til_granite:
        u_cold_array = A1_u_cold_granite
    elif mat_id == gv.id_Til_water:
        u_cold_array = A1_u_cold_water
    else:
        raise ValueError("Invalid material ID")

    N_row = u_cold_array.shape[0]
    rho_min = 100
    rho_max = 100000

    drho = (rho_max - rho_min)/(N_row - 1)

    a = int(((rho - rho_min)/drho))
    b = a + 1

    if a >= 0 and a < (N_row - 1):
        interpolation = u_cold_array[a]
        interpolation += ((u_cold_array[b] - u_cold_array[a])/drho)*(rho - rho_min - a*drho)

    elif rho < rho_min:
        interpolation = u_cold_array[0]
    else:
        interpolation = u_cold_array[int(N_row - 1)]
        interpolation += ((u_cold_array[int(N_row - 1)] - u_cold_array[int(N_row) - 2])/drho)*(rho - rho_max)

    return interpolation

@njit
def u_rho_T(rho, T, mat_id):
    mat_type    = mat_id // gv.type_factor

    if (mat_type == gv.type_Til):
        cv = C_V(mat_id)

        u = u_cold_tab(rho, mat_id) + cv*T
        
    else:
        raise ValueError("Invalid material ID")

    return u
