#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:38:18 2019

@author: sergio
"""

import numpy as np
import scipy.integrate
from numba import jit, njit
from mpi4py import MPI
import time
import sys
from scipy.interpolate import interp1d
import os
import pandas as pd
import matplotlib.pyplot as plt
path = '/home/sergio/Documents/SpiPGen/'
os.chdir(path)

import seagen
import swift_io
import h5py

# Global constants
G = 6.67408E-11;
R_earth = 6371000;
M_earth = 5.972E24;

# //////////////////////////////////////////////////////////////////////////// #
#                              II. Functions                                   #
# //////////////////////////////////////////////////////////////////////////// #

@jit(nopython=True)
def _P_EoS_Till(u, rho, mat_id):
    """
    Computes pressure for Tillotson EoS.
    
    Args:
        u (double)
            Internal energy (SI).
            
        rho (double) 
            Density (SI).
        
        mat_id ([int])
            Material id.
            
    Returns:
        P (double)
            Pressure (SI).
    """
    # Material constants for Tillotson EoS
    # mat_id, rho_0, a, b, A, B, u_0, u_iv, u_cv, alpha, beta, eta_min, P_min
    iron    = np.array([100, 7800, 0.5, 1.5, 1.28e11, 1.05e11, 9.5e9, 2.4e9, 8.67e9, 5, 5, 0, 0])
    granite = np.array([101, 2680, 0.5, 1.3, 1.8e10, 1.8e10, 1.6e10, 3.5e9, 1.8e10, 5, 5, 0, 0])
    water   = np.array([102, 998, 0.7, 0.15, 2.18e9, 1.325e10, 7.0e9, 4.19e8, 2.69e9, 10, 5, 0.9, 0])
    
    if (mat_id == 100):
        material = iron
    elif (mat_id == 101):
        material = granite
    elif (mat_id == 102):
        material = water
    else:
        print("Material not implemented")
        return None
        
    rho_0    = material[1]
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

    eta      = rho/rho_0
    eta_sq   = eta*eta
    mu       = eta - 1.
    nu       = 1./eta - 1.
    w        = u/(u_0*eta_sq) + 1.
    w_inv    = 1./w
    
    P_c = 0.
    P_e = 0.
    P   = 0.

    # Condensed or cold
    if eta < eta_min:
        P_c = 0.
    else:
        P_c = (a + b*w_inv)*rho*u + A*mu + B*mu*mu;
        
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
        P = P_min;
        
    return P


@jit(nopython=True)
def P_EoS(u, rho, mat_id):
    """
    Computes pressure for Tillotson EoS.
    
    Args:
        u (double)
            Internal energy (SI).
            
        rho (double) 
            Density (SI).
        
        mat_id ([int])
            Material id.
            
    Returns:
        P (double)
            Pressure (SI).
    """
    if (mat_id == 100):
        return _P_EoS_Till(u, rho, mat_id)
    elif (mat_id == 101):
        return _P_EoS_Till(u, rho, mat_id)
    elif (mat_id == 102):
        return _P_EoS_Till(u, rho, mat_id)
    else:
        print("Material not implemented")
        return None

@jit(nopython=True)
def _rho0_material(mat_id):
    """Returns rho_0 for a given material id
    
    Args:          
        mat_id ([int])
            Material id.
    
    """
    if (mat_id == 100):     # Tillotson iron
        return 7800
    elif (mat_id == 101):   # Tillotson granite   
        return 2680
    elif (mat_id == 102):   # Tillotson water
        return 998
    elif (mat_id == 200):   # H&M80 H-He atmosphere
        return None
    elif (mat_id == 201):   # H&M80 H-He ice mix
        return None
    elif (mat_id == 202):   # H&M80 H-He rock mix
        return None
    elif (mat_id == 300):   # SESAME iron
        return 7902
    elif (mat_id == 301):   # SESAME basalt
        return 2902
    elif (mat_id == 302):   # SESAME water
        return 1402
    elif (mat_id == 303):   # SS08 like-SESAME water
        return 1002
    else:
        print("Material not implemented")
        return None
    
@jit(nopython=True)
def _spec_c(mat_id):
    """Returns specific capacity for a given material id
    
    Args:          
        mat_id ([int])
            Material id.
    
    """
    if (mat_id == 100):     # Tillotson iron
        return 449
    elif (mat_id == 101):   # Tillotson granite   
        return 790
    elif (mat_id == 102):   # Tillotson water
        return 4186
    elif (mat_id == 200):
        return None
    elif (mat_id == 201):
        return None
    elif (mat_id == 202):
        return None
    elif (mat_id == 300):   # SESAME iron
        return 449
    elif (mat_id == 301):   # SESAME basalt
        return 790
    elif (mat_id == 302):   # SESAME water
        return 4186
    elif (mat_id == 303):   # SS08 like-SESAME water
        return 4186
    else:
        print("Material not implemented")
        return None
    
@jit(nopython=True)
def ucold(rho, mat_id, N):
    """
    Computes internal energy cold.
    
    Args:          
        rho (float) 
            Density (SI).
        
        mat_id ([int])
            Material id.
            
        N (int)
            Number of subdivisions for the numerical integral.
            
    Returns:
        uc (float)
            Cold internal energy (SI).
    """

    rho0 = _rho0_material(mat_id)
    drho = (rho - rho0)/N
    x = rho0
    uc = 1e-9

    for j in range(N):
        x += drho
        uc += P_EoS(uc, x, mat_id)*drho/x**2

    return uc

@jit(nopython=True)
def T_rho(rho, T_rho_id, T_rho_args):
    """
    Computes temperature given density (T = f(rho)).
    
    Args:
        rho (float)
            Density (SI).
            
        T_rho_id (int)
            Relation between T and rho to be used.
            
        T_rho_args (list):
            Extra arguments to determine the relation
            
    Returns:
        Temperature (SI)
            
    """
    if (T_rho_id == 1):  # T = K*rho**alpha, T_rho_args = [K, alpha]
        
        K = T_rho_args[0]
        alpha = T_rho_args[1]
        return K*rho**alpha
    
    else:
        print("relation_id not implemented")
        return None

@jit(nopython=True)
def P_rho(rho, mat_id, T_rho_id, T_rho_args):
    """
    Computes pressure using Tillotson EoS, and
    internal energy = internal energy cold + c*Temperature 
    (which depends on rho and the relation between temperature and density).
    
    Args:          
        rho (float) 
            Density (SI).
        
        mat_id ([float])
            Material id.
            
        T_rho_id (int)
            Relation between T and rho to be used.
            
        T_rho_args (list):
            Extra arguments to determine the relation
            
    Returns:
        P (float):
            Pressure (SI).
    """
    
    N = 10000
    c = _spec_c(mat_id)
    u = ucold(rho, mat_id, N) + c*T_rho(rho, T_rho_id, T_rho_args)
    P = P_EoS(u, rho, mat_id)

    return P

@jit(nopython=True)
def _create_ucold_array(mat_id):
    """
    Computes values of the cold internal energy and stores it to save 
    computation time in future calculations.
    
    Args:
        mat_id (int):
            Material id.
            
    Returns:
        ucold_array ([float])
    """

    nrow = 10000
    ucold_array = np.zeros((nrow,))
    rho_min = 100
    rho_max = 100000
    Nucold = 10000

    rho = rho_min
    drho = (rho_max - rho_min)/(nrow - 1)
    
    rho = rho_min
    for i in range(nrow):
        ucold_array[i] = ucold(rho, mat_id, Nucold)
        rho = rho + drho
    
    return ucold_array

@jit(nopython=True)
def _ucold_tab(rho, ucold_array):
    """
    Fast computation of cold internal energy using the table previously
    computed.
    
    Args:
        rho (float):
            Density (SI).
            
        ucold_array ([float])
            Precomputed values of cold internal energy for a particular material
            with function _create_ucold_array() (SI).
            
    Returns:
        interpolation (float):
            cold internal energy (SI).
    """

    nrow = ucold_array.shape[0]
    rho_min = 100
    rho_max = 100000

    drho = (rho_max - rho_min)/(nrow - 1)

    a = int(((rho - rho_min)/drho))
    b = a + 1

    if a >= 0 and a < (nrow - 1):
        interpolation = ucold_array[a]
        interpolation += ((ucold_array[b] - ucold_array[a])/drho)*(rho - rho_min - a*drho)

    elif rho < rho_min:
        interpolation = ucold_array[0]
    else:
        interpolation = ucold_array[int(nrow - 1)]
        interpolation += ((ucold_array[int(nrow - 1)] - ucold_array[int(nrow) - 2])/drho)*(rho - rho_max)

    return interpolation

@jit(nopython=True)
def _find_rho(Ps, mat_id, T_rho_id, T_rho_args, rho0, rho1, ucold_array):
    """
    Root finder of the density for Tillotson EoS using 
    tabulated values of cold internal energy
    
    Args:
        Ps (float):
            Pressure (SI).
            
        mat_id (int):
            Material id (SI).
        
        T_rho_id (int)
            Relation between T and rho to be used.
            
        T_rho_args (list):
            Extra arguments to determine the relation
            
        rho0 (float):
            Lower bound for where to look the root (SI).
            
        rho1 (float):
            Upper bound for where to look the root (SI).
        
        ucold_array ([float])
            Precomputed values of cold internal energy
            with function _create_ucold_array() (SI).
            
    Returns:
        rho2 (float):
            Value of the density which satisfies P(u(rho), rho) = 0 
            (SI).
    """

    c = _spec_c(mat_id)
    tolerance = 1E-5
    
    u0 = _ucold_tab(rho0, ucold_array) + c*T_rho(rho0, T_rho_id, T_rho_args)
    P0 = P_EoS(u0, rho0, mat_id)
    u1 = _ucold_tab(rho1, ucold_array) + c*T_rho(rho1, T_rho_id, T_rho_args)
    P1 = P_EoS(u1, rho1, mat_id)
    rho2 = (rho0 + rho1)/2.
    u2 = _ucold_tab(rho2, ucold_array) + c*T_rho(rho2, T_rho_id, T_rho_args)
    P2 = P_EoS(u2, rho2, mat_id)
    
    rho_aux = rho0 + 1e-6
    u_aux = _ucold_tab(rho_aux, ucold_array) + c*T_rho(rho_aux, T_rho_id, T_rho_args)
    P_aux = P_EoS(u_aux, rho_aux, mat_id)

    if ((P0 < Ps and Ps < P1) or (P0 > Ps and Ps > P1)):
        while np.abs(rho1 - rho0) > tolerance:
            u0 = _ucold_tab(rho0, ucold_array) + c*T_rho(rho0, T_rho_id, T_rho_args)
            P0 = P_EoS(u0, rho0, mat_id)
            u1 = _ucold_tab(rho1, ucold_array) + c*T_rho(rho1, T_rho_id, T_rho_args)
            P1 = P_EoS(u1, rho1, mat_id)
            u2 = _ucold_tab(rho2, ucold_array) + c*T_rho(rho2, T_rho_id, T_rho_args)
            P2 = P_EoS(u2, rho2, mat_id)
            
            f0 = Ps - P0
            #f1 = Ps - P1
            f2 = Ps - P2
            
            if f0*f2 > 0:
                rho0 = rho2
            else:
                rho1 = rho2
                
            rho2 = (rho0 + rho1)/2.
            
        return rho2
    
    elif (P0 == Ps and P_aux == Ps and P1 != Ps):
        while np.abs(rho1 - rho0) > tolerance:
            rho2 = (rho0 + rho1)/2.
            u0 = _ucold_tab(rho0, ucold_array) + c*T_rho(rho0, T_rho_id, T_rho_args)
            P0 = P_EoS(u0, rho0, mat_id)
            u1 = _ucold_tab(rho1, ucold_array) + c*T_rho(rho1, T_rho_id, T_rho_args)
            P1 = P_EoS(u1, rho1, mat_id)
            rho2 = (rho0 + rho1)/2.
            u2 = _ucold_tab(rho2, ucold_array) + c*T_rho(rho2, T_rho_id, T_rho_args)
            P2 = P_EoS(u2, rho2, mat_id)
            
            if P2 == Ps:
                rho0 = rho2
            else:
                rho1 = rho2
            
            rho2 = rho2 = (rho0 + rho1)/2.
            
        return rho2
    
    elif Ps < P0 and P0 < P1:
        print("Exception 1\n")
        #print("P0: %.2f P1 %.2f Ps %.2f" %(round(P0/1e9,2), round(P1/1e9,2), round(Ps/1e9,2)))
        return rho0
    elif Ps > P1 and P0 < P1:
        print("Exception 2\n")
        #print("P0: %.2f P1 %.2f Ps %.2f" %(round(P0/1e9,2), round(P1/1e9,2), round(Ps/1e9,2)))
        return rho1
    else:
        print("Exception 3\n")
        #print("P0: %.2f P1 %.2f Ps %.2f" %(round(P0/1e9,2), round(P1/1e9,2), round(Ps/1e9,2)))
        return rho2

    return rho2;

@jit(nopython=True)
def _find_rho_fixed_T(Ps, mat_id, Ts, rho0, rho1, ucold_array):
    """
    Root finder of the density for Tillotson EoS using 
    tabulated values of cold internal energy
    
    Args:
        Ps (float):
            Pressure (SI).
            
        mat_id (int):
            Material id (SI).
        
        Ts (float):
            Temperature (SI)
            
        rho0 (float):
            Lower bound for where to look the root (SI).
            
        rho1 (float):
            Upper bound for where to look the root (SI).
        
        ucold_array ([float])
            Precomputed values of cold internal energy
            with function _create_ucold_array() (SI).
            
    Returns:
        rho2 (float):
            Value of the density which satisfies P(u(rho), rho) = 0 
            (SI).
    """

    return _find_rho(Ps, mat_id, 1, [Ts, 0], rho0, rho1, ucold_array)

"""
# find rho_0 for a material
import matplotlib.pyplot as plt

ucold_array_101 = _create_ucold_array(101)

drho = (100000-100)/9999
rho = np.arange(100, 100000 + drho, drho)

plt.scatter(rho, ucold_array_101, s=1)
plt.show()

ucold_model     = interp1d(rho, ucold_array_101)
ucold_model_inv = interp1d(ucold_array_101, rho)

ucold_model(100)
ucold_model_inv(1)
"""







