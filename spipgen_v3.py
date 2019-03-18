# -*- coding: utf-8 -*-
"""
Auxiliar functions and constants to compute a density grid with rotation.
"""

# ========
# Contents:
# ========
#   I   Libraries and constants
#   II  Functions
#   III Main

# //////////////////////////////////////////////////////////////////////////// #
#                        I. Libraries and constants                            #
# //////////////////////////////////////////////////////////////////////////// #


# Import necessary libraries
import numpy as np
import scipy.integrate
from numba import jit, njit
from mpi4py import MPI
import time
import sys
from scipy.interpolate import interp1d

import seagen
import swift_io
import h5py

# Material constants for Tillotson EoS + material code and specific capacity (SI units)
iron = np.array([0.5, 1.5, 1.279E11, 1.05E11, 7860, 9.5E6, 1.42E6, 8.45E6, 5, 5, 0, 449])
granite = np.array([0.5, 1.3, 1.8E10, 1.8E10, 2700, 1.6E7, 3.5E6, 1.8E7, 5, 5, 1, 790])
water = np.array([0.5, 0.9, 2.0E10, 1.0E10, 1000, 2.0E6, 4.0E5, 2.0E6, 5, 5, 2, 4186])

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
        return 7860
    elif (mat_id == 101):   # Tillotson granite   
        return 2700
    elif (mat_id == 102):   # Tillotson water
        return 1000
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
def _analytic_solution_r(r, R, Z, x):
    if R == Z:
        return 2*(r*r - 3*(R*R + x))/3/np.sqrt((R*R + x)**3)
    else:
        A1 = -r*r*np.sqrt(x + Z*Z)/(R*R + x)/(R*R - Z*Z)
        A2 = -(r*r - 2*R*R + 2*Z*Z)
        A2 = A2*np.arctan(np.sqrt((x + Z*Z)/(R*R - Z*Z)))
        A2 = A2/((R*R - Z*Z)**(3/2))
        return A1 + A2
    
    return 0

@jit(nopython=True)
def _analytic_solution_z(z, R, Z, x):
    if R == Z:
        return 2*(z*z - 3*(R*R + x))/3/np.sqrt((R*R + x)**3)
    else:
        A1 = 2*z*z/(R*R - Z*Z)/np.sqrt(Z*Z + x)
        A2 = 2*(R*R + z*z - Z*Z)
        A2 = A2*np.arctan(np.sqrt((x + Z*Z)/(R*R - Z*Z)))
        A2 = A2/((R*R - Z*Z)**(3/2))
        return A1 + A2
    
    return 0

@jit(nopython=True)
def _Vgr(r, R, Z, rho):
    
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
            return -G*vol*rho/r
        else:
            M = 4/3*np.pi*R**3*rho
            return -G*M/2/R**3*(3*R*R - r*r)


    if r <= R:
        V = np.pi*R*R*Z*rho
        V = V*(_analytic_solution_r(r, R, Z, 1e30)
               - _analytic_solution_r(r, R, Z, 0))
        return -G*V
    
    else:
        A = r*r - R*R
        V = np.pi*R*R*Z*rho
        V = V*(_analytic_solution_r(r, R, Z, 1e30)
               - _analytic_solution_r(r, R, Z, A))
        return -G*V
    
    return V

@jit(nopython=True)
def _Vgz(z, R, Z, rho):
    
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
            return -G*vol*rho/z
        else:
            M = 4/3*np.pi*R**3*rho
            return -G*M/2/R**3*(3*R*R - z*z)
        
    
    if z <= Z:
        V = np.pi*R*R*Z*rho
        V = V*(_analytic_solution_z(z, R, Z, 1e40)
               - _analytic_solution_z(z, R, Z, 0))
        return -G*V
    
    else:
        A = z*z - Z*Z
        V = np.pi*R*R*Z*rho
        V = V*(_analytic_solution_z(z, R, Z, 1e40)
               - _analytic_solution_z(z, R, Z, A))
        return -G*V
    
    return V

@jit(nopython=False)
def _fillV(r_array, rho_e, z_array, rho_p, Tw):
    
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
            V_e[j] += _Vgr(r_array[j], R_array[i], 
                           Z_array[i], delta_rho)                      
            
        for j in range(V_p.shape[0]):
            V_p[j] += _Vgz(z_array[j], R_array[i], 
                           Z_array[i], delta_rho)
            
    for i in range(V_e.shape[0]):
        V_e[i] += -(1/2)*(W*r_array[i])**2
        
    return V_e, V_p

# From center to surface
@jit(nopython=True)
def _fillrho(r_array, V_e, z_array, V_p, P_c, P_s, rho_c, rho_s,
             mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core):
    
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
        #print(i)
            
        if P_e[i + 1] >= P_s:
            rho_e[i + 1] = _find_rho(P_e[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_e[i], ucold_array_core) 
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s:
            rho_p[i + 1] = _find_rho(P_p[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_p[i], ucold_array_core)
        else:
            rho_p[i + 1] = 0.
            break
        
    return rho_e, rho_p

def spin1layer(iterations, r_array, z_array, radii, densities, Tw,
               P_c, P_s, rho_c, rho_s,
               mat_id_core, T_rho_id_core, T_rho_args_core):
    
    if mat_id_core == 100:
        ucold_array_core = np.load('ucold_array_100.npy')
    elif mat_id_core == 101:
        ucold_array_core = np.load('ucold_array_101.npy')
    elif mat_id_core == 102:
        ucold_array_core = np.load('ucold_array_102.npy')
        
    spherical_model = interp1d(radii, densities, bounds_error=False, fill_value=0)

    rho_e = spherical_model(r_array)
    rho_p = spherical_model(z_array)
    
    profile_e = []
    profile_p = []
    
    profile_e.append(rho_e)
    profile_p.append(rho_p)
    
    for i in range(iterations):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho(r_array, V_e, z_array, V_p, P_c, P_s, rho_c, rho_s,
                                mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core)
        profile_e.append(rho_e)
        profile_p.append(rho_p)
    
    return profile_e, profile_p
   
@jit(nopython=False)
def _el_eq(r, z, R, Z):
    return r*r/R/R + z*z/Z/Z

@jit(nopython=False)
def _rho_rz(r, z, r_array, rho_e, z_array, rho_p):
    
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
        r_2 = (r_0 + r_1)/2.
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
                
            r_2 = (r_0 + r_1)/2.
            rho_2 = rho_e_model(r_2)
            R_2 = r_2
            Z_2 = rho_p_model_inv(rho_2)
            
        return rho_2
    
    return -1
    
#test 1 layer
import pandas as pd
import matplotlib.pyplot as plt
   
data = pd.read_csv("1layer.csv", header=0)

iterations = 10
r_array = np.arange(0, 1.2*R_earth, 1.2*R_earth/1000)
z_array = np.arange(0, 1.1*R_earth, 1.1*R_earth/1000)
radii = np.array(data.R)*R_earth    
densities = np.array(data.rho)
Tw = 4
P_c = np.median(np.sort(data.P)[-100:])
P_s = np.min(data.P)
rho_c = np.median(np.sort(data.rho)[-100:])
rho_s = np.min(data.rho)
mat_id_core = 101
T_rho_id_core = 1
T_rho_args_core = [300, 0]
N = 10**5

profile_e, profile_p = spin1layer(iterations, r_array, z_array, radii, densities, Tw,
                                  P_c, P_s, rho_c, rho_s,
                                  mat_id_core, T_rho_id_core, T_rho_args_core)

plt.scatter(r_array/R_earth, profile_e[0], label = 'original', s = 1)
plt.scatter(r_array/R_earth, profile_e[10], label = 'last iter', s = 1)
plt.legend()
plt.show()

plt.scatter(z_array/R_earth, profile_p[0], label = 'original', s = 1)
plt.scatter(z_array/R_earth, profile_p[10], label = 'last iter', s = 1)
plt.legend()
plt.show()

#2d plot
rho_e = profile_e[10]
rho_p = profile_p[10]

r_array_coarse = np.arange(0, np.max(r_array), np.max(r_array)/100)
z_array_coarse = np.arange(0, np.max(z_array), np.max(z_array)/100)
rho_grid = np.zeros((r_array_coarse.shape[0], z_array_coarse.shape[0]))
for i in range(rho_grid.shape[0]):
    r = r_array_coarse[i]
    for j in range(rho_grid.shape[1]):
        z = z_array_coarse[j]
        rho_grid[i,j] = _rho_rz(r, z, r_array, rho_e, z_array, rho_p)
        
spipgen_plot.plotrho(rho_grid, r_array_coarse, z_array_coarse)

# particle placement 1 layer
def _picle_placement_1layer(r_array, rho_e, z_array, rho_p, Tw, N,
                            mat_id_core, T_rho_id_core, T_rho_args_core):
    
    rho_e_model = interp1d(r_array, rho_e)
    
    rho_e_model_inv = interp1d(rho_e, r_array)
    rho_p_model_inv = interp1d(rho_p, z_array)

    Re = np.max(r_array[rho_e > 0])

    radii = np.arange(0, Re, Re/1000000)
    densities = rho_e_model(radii)

    particles = seagen.GenSphere(N, radii[1:], densities[1:])
    
    particles_r = np.sqrt(particles.x**2 + particles.y**2 + particles.z**2)
    particles_rc = np.sqrt(particles.x**2 + particles.y**2)
    particles_rho = rho_e_model(particles_r)
    
    R = rho_e_model_inv(particles_rho)
    Z = rho_p_model_inv(particles_rho)
    
    zP = np.sqrt(Z**2*(1 - (particles_rc/R)**2))*np.sign(particles.z)

    """ 
    plt.scatter(particles.z/R_earth, zP/particles.z, s = 0.01, c = 'red')
    plt.xlabel(r"$z$ $[R_{earth}]$")
    plt.ylabel(r"$z'/z$")
    plt.show()
    """
    # Tweek masses
    mP = particles.m*zP/particles.z
    print("\nx, y, z, and m computed\n")
    
    # Compute velocities (T_w in hours)
    vx = np.zeros(mP.shape[0])
    vy = np.zeros(mP.shape[0])
    vz = np.zeros(mP.shape[0])
    
    hour_to_s = 3600
    wz = 2*np.pi/Tw/hour_to_s 
        
    vx = -particles.y*wz
    vy = particles.x*wz
    
    # internal energy
    rho = particles_rho
    u = np.zeros((mP.shape[0]))
    
    x = particles.x
    y = particles.y
    
    print("vx, vy, and vz computed\n")
    
    try:
        
        if mat_id_core == 100:
            ucold_array_core = np.load('ucold_array_100.npy')
        elif mat_id_core == 101:
            ucold_array_core = np.load('ucold_array_101.npy')
        elif mat_id_core == 102:
            ucold_array_core = np.load('ucold_array_102.npy')
            
    except ImportError:
        return False
    
    #ucold_array_core = spipgen_v2._create_ucold_array(mat_id_core)
    c_core = _spec_c(mat_id_core)
    
    for k in range(mP.shape[0]):
        u[k] = _ucold_tab(particles_rho[k], ucold_array_core)
        u[k] = u[k] + c_core*T_rho(particles_rho[k], T_rho_id_core, T_rho_args_core)
    
    print("Internal energy u computed\n")
    ## Smoothing lengths, crudely estimated from the densities
    num_ngb = 48    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    A1_h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_P = np.ones((mP.shape[0],)) # not implemented (not necessary)
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = np.ones((mP.shape[0],))*mat_id_core
    
    return x, y, zP, vx, vy, vz, mP, A1_h, rho, A1_P, u, A1_id, A1_mat_id

x, y, z, vx, vy, vz, m, h, rho, P, u, picle_id, mat_id =                      \
_picle_placement_1layer(rho_grid, r_array, z_array, Tw, N,
                        mat_id_core, T_rho_id_core, T_rho_args_core)

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '1layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, np.array([x, y, z]).T, np.array([vx, vy, vz]).T,
                             m, h, rho, P, u, picle_id, mat_id,
                             4*R_earth, swift_to_SI) 