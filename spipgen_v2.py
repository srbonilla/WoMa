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
from scipy import interpolate

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

# Especial functions for the initial spherical case

# Compute rho0 given the spherical radii profile (from high to low as in function before)
# Rs in Earth radii units
def _rho0_grid(radii_sph, densities_sph, r_array = np.nan, z_array = np.nan):
    """
    Creates 2-d array density profile out of a 1-d array radial
    density profile (spherical).
    
    Args: 
        radii_sph ([float]):
            Radii profile (SI). From center to surface.
            
        densities_sph ([float]):
            Radial density profile (SI). From center to surface.
            
        r_array ([float]):
            Array of distances from the z axis to build the grid (SI).
            First element must be 0.
            
        z_array ([float]):
            Array of distances in the z direction to build the grid (SI).
            First element must be 0.
            
    Returns:
        rho0 ([[float]]):
            2-d spherical density profile (SI).
    """

    radii_sph = np.sort(radii_sph)
    densities_sph = np.flip(np.sort(densities_sph))
    
    if np.isnan(np.array(r_array)).any():
        r_array = np.arange(0, 1.5*np.max(radii_sph), 1.5*np.max(radii_sph)/100)
        
    if np.isnan(np.array(z_array)).any():
        z_array = np.arange(0, 1.1*np.max(radii_sph), 1.1*np.max(radii_sph)/100)
        
    rho_model_sph = interpolate.interp1d(radii_sph, densities_sph, kind = 'quadratic')
    
    rho_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
    
    for i in range(r_array.shape[0]):
        for j in range(z_array.shape[0]):
            r = np.sqrt(r_array[i]**2 + z_array[j]**2)
            if r > np.max(radii_sph):
                rho_grid[i,j] = 0.
            else:
                rho_grid[i,j] = rho_model_sph(r)
            
            
    return rho_grid, r_array, z_array
     
def _create_I_array():
    """
    Tabulates the integral respect to angle phi' inside the 3-d
    integral of the gravitational potential.
    
    Returns:
        I_array ([float]):
            Values of the integral for gamma from 0 to 1.
    """
    
    f = lambda x, gamma: 1/np.sqrt(1 - gamma*np.cos(x))
    prec = 1e5

    gamma = np.arange(0, 1 + 1/prec, 1/prec)
    I_array = np.zeros((gamma.shape[0],))
    for i in range(gamma.shape[0]):
        I_array[i] = scipy.integrate.quad(f, 0, 2*np.pi, args=(gamma[i] - 1e-10,))[0]
        
    return I_array

@jit(nopython=True)
def _I_tab(gamma, I_array):
    """
    Computes the integral respect to angle phi inside the 3-d
    integral of the gravitational potential.
    
    Args:
        gamma (float):
            Parameter depending on r and z, of which the integral depends.
            
        I_array ([float]):
            Values of the integral for gamma from 0 to 1.
    
    Returns:
        I_array[i] (float):
            Values of the integral for the specified gamma.
    """
    
    prec = 1e5
    i = int((gamma)*prec)
    return I_array[i]

@jit(nopython=True)
def _dS(r_array, z_array):
    """Computes de differential of surface for any point i, j of the 2d density grid 
    
    """
    
    dS = np.zeros((r_array.shape[0], z_array.shape[0]))
    
    for i in range(r_array.shape[0]):
        for j in range(z_array.shape[0]):
            
            if i == 0 and j == 0:
                dS[i,j] = r_array[1]*z_array[1]/4
                
            elif i == 0 and j == z_array.shape[0] - 1:
                dS[i,j] = (r_array[1] - r_array[0])/2*(z_array[j] - z_array[j - 1])/2
                
            elif i == r_array.shape[0] - 1 and j == 0:
                dS[i,j] = (r_array[i] - r_array[i - 1])/2*(z_array[1] - z_array[0])/2
                
            elif i == r_array.shape[0] - 1 and j == z_array.shape[0] - 1:
                dS[i,j] = (r_array[i] - r_array[i - 1])/2*(z_array[j] - z_array[j - 1])/2
                
            elif i == 0:
                top = (z_array[j + 1] + z_array[j])/2 
                bottom = (z_array[j] + z_array[j - 1])/2 
                dS[i,j] = (r_array[1] - r_array[0])/2*(top - bottom)
            
            elif j == 0:
                right = (r_array[i + 1] + r_array[i])/2 
                left = (r_array[i] + r_array[i - 1])/2 
                dS[i,j] = (z_array[1] - z_array[0])/2*(right - left)
            
            elif i == r_array.shape[0] - 1:
                top = (z_array[j + 1] + z_array[j])/2 
                bottom = (z_array[j] + z_array[j - 1])/2 
                dS[i,j] = (r_array[i] - r_array[i - 1])/2*(top - bottom)
            
            elif j == z_array.shape[0] - 1:
                right = (r_array[i + 1] + r_array[i])/2 
                left = (r_array[i] + r_array[i - 1])/2 
                dS[i,j] = (z_array[j] - z_array[j - 1])/2*(right - left)
            
            else:
                top = (z_array[j + 1] + z_array[j])/2 
                bottom = (z_array[j] + z_array[j - 1])/2 
                right = (r_array[i + 1] + r_array[i])/2 
                left = (r_array[i] + r_array[i - 1])/2 
                dS[i,j] = (top - bottom)*(right - left)
                
    return dS
            
        

# Computes the gravitational potential given a density grid at some point rc, z
@jit(nopython=True)
def _Vg(rc, z, rho_grid, r_array, z_array, I_array, S_grid):
    """
    Computes the gravitational potential given a 2-d density grid at any point
    rc, z.
    
    Args:
        rc (float):
            Cylindrical radii (distance from z axis) (SI).
            
        z (float):
            z coordinate (SI).
            
        rho_grid ([[float]]):
            2-d density grid (SI) of a planet.
            
        r_array ([float]):
            Array of distances from the z axis to build the grid (SI).
            
        z_array ([float]):
            Array of distances in the z direction to build the grid (SI).
            
        I_array ([float]):
            Values of the integral for gamma from 0 to 1.
            
        S_grid ([[float]]):
            Differential of surface for every element i,j of the density grid.
            
    Returns:
        
        V (float):
            Gravitational potential (SI).
        
            
    """
    
    Vsum = 0
    err = 1e-10
    
    for i in range(rho_grid.shape[0]):
        rcp = r_array[i]
        for j in range(rho_grid.shape[1]):
            zp = z_array[j]
            
            if rcp != rc or zp != z:
                a2 = rcp*rcp + zp*zp + rc*rc + z*z
                alpha = 1 - 2*z*zp/(a2 + err)
                beta = 2*rc*rcp/(a2 + err)
                gamma = beta/(alpha + err) 
                    
                Vsum = Vsum + rho_grid[i,j]*rcp*_I_tab(gamma, I_array)/np.sqrt(a2 + err)/np.sqrt(alpha + err)*S_grid[i,j]
                
            zp = -z_array[j]
            
            if rcp != rc or zp != z:
                a2 = rcp*rcp + zp*zp + rc*rc + z*z
                alpha = 1 - 2*z*zp/(a2 + err)
                beta = 2*rc*rcp/(a2 + err)
                gamma = beta/(alpha + err) 
                    
                Vsum = Vsum + rho_grid[i,j]*rcp*_I_tab(gamma, I_array)/np.sqrt(a2 + err)/np.sqrt(alpha + err)*S_grid[i,j]
                
    V = -G*Vsum
    
    return V

@jit(nopython=True)
def _fillV(rho_grid, r_array, z_array, I_array, S_grid, Tw):
    """
    Computes a 2-d potential grid given a 2-d density grid of a rotating planet.
    
    Args:
        rho_grid ([[float]]):
            2-d density grid (SI) of the planet.
            
        r_array ([float]):
            Array of distances from the z axis to build the grid (SI).
            
        z_array ([float]):
            Array of distances in the z direction to build the grid (SI).
            
        I_array ([float]):
            Values of the integral for gamma from 0 to 1.
            
        S_grid ([[float]]):
            Differential of surface for every element i,j of the density grid.
            
        Tw (float):
            Period of the planet (hours).
    
    Returns:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
    
    """

    V_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
    #Tw in hours
    W = 2*np.pi/Tw/60/60
    
    for i in range(V_grid.shape[0]):
        for j in range(V_grid.shape[1]):
            V_grid[i,j] = _Vg(r_array[i], z_array[j], rho_grid, r_array, z_array, I_array, S_grid) - (1/2)*(W*r_array[i])**2
            
    return V_grid

def _fillV_parallel(rho_grid, r_array, z_array, I_array, S_grid, Tw):
    """
    Computes a 2-d potential grid given a 2-d density grid of a rotating planet.
    
    Args:
        rho_grid ([[float]]):
            2-d density grid (SI) of the planet.
            
        r_array ([float]):
            Array of distances from the z axis to build the grid (SI).
            
        z_array ([float]):
            Array of distances in the z direction to build the grid (SI).
            
        I_array ([float]):
            Values of the integral for gamma from 0 to 1.
            
        S_grid ([[float]]):
            Differential of surface for every element i,j of the density grid.
            
        Tw (float):
            Period of the planet (hours).
    
    Returns:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
    
    """
    
    # parallel set-up
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    V_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
    #Tw in hours
    W = 2*np.pi/Tw/60/60
    
    N_rows = z_array.shape[0] 
    
    # Essential: N_rows is a multiple of size (the number of processors). N_rows = 100 by default 
    for i in range(V_grid.shape[0]):
        for j in range(rank*int(N_rows/size), (rank + 1)*int(N_rows/size)):
            V_grid[i,j] = _Vg(r_array[i], z_array[j], rho_grid, r_array, z_array, I_array, S_grid) - (1/2)*(W*r_array[i])**2
            
    V_raw = comm.gather(V_grid)
    V_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
    
    if rank == 0:
        for I in range(size):
            V_grid[:,:] = V_grid[:,:] + V_raw[I]
        #V_grid = np.flip(V_grid, 1)
        
    V_grid = comm.bcast(V_grid, root=0)
    
    return V_grid

# From center to surface
@jit(nopython=True)
def _fillrho(V_grid, r_array, z_array, P_c, P_s, rho_c, rho_s,
             mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core):
    """
    Update 2-d density grid given a 2-d potential grid of the planet.
    
    Args:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
            
        r_array ([float]):
            Array of distances from the z axis to build the grid (SI).
            
        z_array ([float]):
            Array of distances in the z direction to build the grid (SI).
            
        P_c (float):
            Central pressure of the planet (SI).
            
        P_s (float):
            Surface pressure of the planet (SI).
            
        rho_c (float):
            Central density of the planet (SI).
            
        rho_s (float):
            Surface density of the planet (SI).
            
        mat_id_core ([int]):
            Material id.
            
        T_rho_id_core (int)
            Relation between T and rho to be used for core material.
            
        T_rho_args_core (list):
            Extra arguments to determine the relation for core material
        
        ucold_array_core ([[float]]):
            Tabulated values of cold internal energy for core material
            
    Returns:
        rho_grid ([[float]]):
            Updated 2-d density grid (SI).
    """
    
    rho_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
        
    rho_grid[0,0] = rho_c
    
    # Fill pole
    for j in range(0, rho_grid.shape[1] - 1):
        gradV = (V_grid[0, j + 1] - V_grid[0, j])
        gradP = -rho_grid[0, j]*gradV
        P = P_rho(rho_grid[0, j], mat_id_core, T_rho_id_core, T_rho_args_core) + gradP
        
        if P > P_s:
            rho_grid[0, j + 1] = _find_rho(P, mat_id_core, T_rho_id_core, T_rho_args_core,
                                           rho_s - 10, rho_grid[0, j], ucold_array_core)
        else:
            rho_grid[0, j + 1] = 0.
            break
        
    # Fill to the right
    for i in range(0, rho_grid.shape[0] - 1):
        for j in range(0, rho_grid.shape[1]):
            if rho_grid[i, j] == 0:
                break

            gradV = (V_grid[i + 1, j] - V_grid[i, j])
            gradP = -rho_grid[i, j]*gradV
            P = P_rho(rho_grid[i, j], mat_id_core, T_rho_id_core, T_rho_args_core) + gradP
            
            if P > P_s:
                rho_grid[i + 1, j] = _find_rho(P, mat_id_core, T_rho_id_core, T_rho_args_core,
                                               rho_s - 10, rho_grid[i, j], ucold_array_core)
            else:
                rho_grid[i + 1, j] = 0.
                break
        
    return rho_grid

def _fillrho_parallel(V_grid, r_array, z_array, P_c, P_s, rho_c, rho_s,
             mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core):
    """
    Update 2-d density grid given a 2-d potential grid of the planet.
    
    Args:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
            
        r_array ([float]):
            Array of distances from the z axis to build the grid (SI).
            
        z_array ([float]):
            Array of distances in the z direction to build the grid (SI).
            
        P_c (float):
            Central pressure of the planet (SI).
            
        P_s (float):
            Surface pressure of the planet (SI).
            
        rho_c (float):
            Central density of the planet (SI).
            
        rho_s (float):
            Surface density of the planet (SI).
            
        mat_id_core ([int]):
            Material id.
            
        T_rho_id_core (int)
            Relation between T and rho to be used for core material.
            
        T_rho_args_core (list):
            Extra arguments to determine the relation for core material
        
        ucold_array_core ([[float]]):
            Tabulated values of cold internal energy for core material
            
    Returns:
        rho_grid ([[float]]):
            Updated 2-d density grid (SI).
    """
    
    # parallel set-up
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    rho_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
    P_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
        
    rho_grid[0,0] = rho_c
    P_grid[0,0] = P_c
    
    # Fill pole for every processor
    for j in range(0, rho_grid.shape[1] - 1):
        gradV = (V_grid[0, j + 1] - V_grid[0, j])
        gradP = -rho_grid[0, j]*gradV
        #P = P_rho(rho_grid[0, j], mat_id_core, T_rho_id_core, T_rho_args_core) + gradP
        P_grid[0, j + 1] = P_grid[0, j] + gradP
        
        if P_grid[0, j + 1] > P_s:
            rho_grid[0, j + 1] = _find_rho(P_grid[0, j + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                           rho_s - 10, rho_grid[0, j], ucold_array_core)
        else:
            P_grid[0, j + 1] = 0.
            rho_grid[0, j + 1] = 0.
            break
        
    N_rows = z_array.shape[0]
    
    # Fill to the right distributing the work
    # Essential: N_rows is a multiple of size (the number of processors). N_rows = 100 by default 
    for i in range(0, rho_grid.shape[0] - 1):
        for j in range(rank*int(N_rows/size), (rank + 1)*int(N_rows/size)):
            if rho_grid[i, j] == 0:
                break

            gradV = (V_grid[i + 1, j] - V_grid[i, j])
            gradP = -rho_grid[i, j]*gradV
            #P = P_rho(rho_grid[i, j], mat_id_core, T_rho_id_core, T_rho_args_core) + gradP
            P_grid[i + 1, j] = P_grid[i, j] + gradP
            
            if P_grid[i + 1, j] > P_s:
                rho_grid[i + 1, j] = _find_rho(P_grid[i + 1, j], mat_id_core, T_rho_id_core, T_rho_args_core,
                                               rho_s - 10, rho_grid[i, j], ucold_array_core)
            else:
                P_grid[0, j + 1]
                rho_grid[i + 1, j] = 0.
                break
        
    rho_raw = comm.gather(rho_grid)
    rho_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
    
    if rank == 0:
        for I in range(size):
            rho_grid[:,:] = rho_grid[:,:] + rho_raw[I]
        #V_grid = np.flip(V_grid, 1)
        
    rho_grid = comm.bcast(rho_grid, root=0)
    
    rho_grid[0,:] = rho_grid[0,:]/size
    
    return rho_grid

    
def spin1layer(iterations, radii, densities, Tw, mat_id, T_rho_id, T_rho_args,
               P_c, P_s, rho_c, rho_s, r_array = None, z_array = None):
    """
    Spining of a radial planetary profile
    
    Args:
        iterations (int):
            Number of iterations to compute.
            
        rho_sph ([float]):
            Initial density radial profile to start with (SI).
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        Tw (float):
            Period of the planet (hours).
            
        K, alpha (float):
            Parameters from the relation between density and temperature
            T = K*rho**alpha.
    
    Returns:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
    """
    
    times = np.zeros((iterations, 2))
    
    start = time.time()
    
    try:
        I_array = np.load('I_array.npy')
        
        if mat_id == 100:
            ucold_array = np.load('ucold_array_100.npy')
        elif mat_id == 101:
            ucold_array = np.load('ucold_array_101.npy')
        elif mat_id == 102:
            ucold_array = np.load('ucold_array_102.npy')  
            
    except ImportError:
        return False
    
    #ucold_array = _create_ucold_array(mat_id)
        
    #I_array = _create_I_array()
    I_array = np.load('I_array.npy')
    
    rho_grid, r_array, z_array = _rho0_grid(radii, densities, r_array, z_array)
    rho = np.zeros((iterations + 1, r_array.shape[0], z_array.shape[0]))
    rho[0] = rho_grid
    
    end = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0: print(f"Running time for creating I_array and u_cold_table: {end - start:.2f} seconds \n")
    
    dS = _dS(r_array, z_array)
    
    for i in range(1, iterations + 1):
        
        start = time.time()
        V1 = _fillV_parallel(rho[i - 1], r_array, z_array, I_array, dS, Tw)
        end = time.time()
        times[i - 1, 0] = end - start
        start = time.time()
        rho[i] = _fillrho_parallel(V1, r_array, z_array, P_c, P_s, rho_c, rho_s, mat_id, T_rho_id, T_rho_args, ucold_array)
        end = time.time()
        times[i - 1, 1] = end - start
        
        if rank == 0: 
            print(f"Iteration {i} complete in {times[i-1,0]:.2f} + {times[i-1,1]:.2f} seconds")
            print(f"Total time: {times[i-1,0] + times[i-1,1]:.2f} seconds\n")
        
    return rho, r_array, z_array, times

#########################2 layer###############################################
# From center to surface
@jit(nopython=True)
def _fillrho2(V_grid, r_array, z_array, P_c, P_i, P_s, rho_c, rho_s,
             mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core,
             mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle, ucold_array_mantle):
    """
    Update 2-d density grid given a 2-d potential grid of the planet.
    
    Args:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
            
        r_array ([float]):
            Array of distances from the z axis to build the grid (SI).
            
        z_array ([float]):
            Array of distances in the z direction to build the grid (SI).
            
        P_c (float):
            Central pressure of the planet (SI).
            
        P_i (float):
            Pressure at the interphase of the 2 materials of the planet (SI).
            
        P_s (float):
            Surface pressure of the planet (SI).
            
        rho_c (float):
            Central density of the planet (SI).
            
        rho_s (float):
            Surface density of the planet (SI).
            
        mat_id_core ([int]):
            Material id.
            
        T_rho_id_core (int)
            Relation between T and rho to be used for core material.
            
        T_rho_args_core (list):
            Extra arguments to determine the relation for core material
        
        ucold_array_core ([[float]]):
            Tabulated values of cold internal energy for core material
            
    Returns:
        rho_grid ([[float]]):
            Updated 2-d density grid (SI).
    """
    
    rho_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
    P_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
        
    rho_grid[0,0] = rho_c
    P_grid[0,0] = P_c
    
    # Fill pole
    for j in range(0, rho_grid.shape[1] - 1):
        gradV = (V_grid[0, j + 1] - V_grid[0, j])
        gradP = -rho_grid[0, j]*gradV
        
        # Core
        if P_grid[0,j] > P_s and P_grid[0,j] > P_i:
            P_grid[0, j + 1] = P_grid[0, j] + gradP
            rho_grid[0, j + 1] = _find_rho(P_grid[0, j + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                           rho_s - 10, rho_grid[0, j], ucold_array_core)
        # Mantle  
        elif P_grid[0,j] > P_s and P_grid[0,j] < P_i:
            P_grid[0, j + 1] = P_grid[0, j] + gradP
            if P_grid[0, j + 1] > P_s:
                rho_grid[0, j + 1] = _find_rho(P_grid[0, j + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                               rho_s - 1, rho_grid[0, j], ucold_array_mantle)
            else:
                P_grid[0, j + 1] = P_s
                rho_grid[0, j + 1] = rho_s
            
        else:
            P_grid[0, j + 1] = P_s
            rho_grid[0, j + 1] = 0.
            break
    
    # Fill to the right 
    for j in range(0, rho_grid.shape[1]):         
        for i in range(0, rho_grid.shape[0] - 1):
            
            if rho_grid[i, j] == 0:
                break

            gradV = (V_grid[i + 1, j] - V_grid[i, j])
            gradP = -rho_grid[i, j]*gradV
            
            # Core
            if P_grid[i, j] > P_s and P_grid[i, j] > P_i:
                P_grid[i + 1, j] = P_grid[i, j] + gradP
                if P_grid[i + 1, j] > P_s:
                    rho_grid[i + 1, j] = _find_rho(P_grid[i + 1, j], mat_id_core, T_rho_id_core, T_rho_args_core,
                                                   rho_s - 10, rho_grid[i, j], ucold_array_core)
                else:
                    P_grid[i + 1, j] = P_s
                    rho_grid[i + 1, j] + rho_s
            # Mantle  
            elif P_grid[i, j] > P_s and P_grid[i, j] < P_i:
                P_grid[i + 1, j] = P_grid[i, j] + gradP
                if P_grid[i + 1, j] > P_s:
                    rho_grid[i + 1, j] = _find_rho(P_grid[i + 1, j], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                                   rho_s - 10, rho_grid[i, j], ucold_array_mantle)
                else:
                    P_grid[i + 1, j] = P_s
                    rho_grid[i + 1, j] + rho_s
            else:
                P_grid[i + 1, j] = P_s
                rho_grid[i + 1, j] = 0.
                break
        
    return rho_grid, P_grid

def _fillrho2_parallel(V_grid, r_array, z_array, P_c, P_i, P_s, rho_c, rho_s,
             mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core,
             mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle, ucold_array_mantle):
    """
    Update 2-d density grid given a 2-d potential grid of the planet.
    
    Args:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
            
        r_array ([float]):
            Array of distances from the z axis to build the grid (SI).
            
        z_array ([float]):
            Array of distances in the z direction to build the grid (SI).
            
        P_c (float):
            Central pressure of the planet (SI).
            
        P_s (float):
            Surface pressure of the planet (SI).
            
        rho_c (float):
            Central density of the planet (SI).
            
        rho_s (float):
            Surface density of the planet (SI).
            
        mat_id_core ([int]):
            Material id.
            
        T_rho_id_core (int)
            Relation between T and rho to be used for core material.
            
        T_rho_args_core (list):
            Extra arguments to determine the relation for core material
        
        ucold_array_core ([[float]]):
            Tabulated values of cold internal energy for core material
            
    Returns:
        rho_grid ([[float]]):
            Updated 2-d density grid (SI).
    """
    
    # parallel set-up
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    rho_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
    P_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
        
    rho_grid[0,0] = rho_c
    P_grid[0,0] = P_c
    
    # Fill pole
    for j in range(0, rho_grid.shape[1] - 1):
        gradV = (V_grid[0, j + 1] - V_grid[0, j])
        gradP = -rho_grid[0, j]*gradV
        
        # Core
        if P_grid[0,j] > P_s and P_grid[0,j] > P_i:
            P_grid[0, j + 1] = P_grid[0, j] + gradP
            rho_grid[0, j + 1] = _find_rho(P_grid[0, j + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                           rho_s - 10, rho_grid[0, j], ucold_array_core)
        # Mantle  
        elif P_grid[0,j] > P_s and P_grid[0,j] < P_i:
            P_grid[0, j + 1] = P_grid[0, j] + gradP
            if P_grid[0, j + 1] > P_s:
                rho_grid[0, j + 1] = _find_rho(P_grid[0, j + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                               rho_s - 1, rho_grid[0, j], ucold_array_mantle)
            else:
                P_grid[0, j + 1] = P_s
                rho_grid[0, j + 1] = rho_s
            
        else:
            P_grid[0, j + 1] = 0.
            rho_grid[0, j + 1] = 0.
            break
    
    N_rows = z_array.shape[0]
    
    # Fill to the right distributing the work
    # Essential: N_rows is a multiple of size (the number of processors). N_rows = 100 by default 
    for j in range(rank*int(N_rows/size), (rank + 1)*int(N_rows/size)):
        for i in range(0, rho_grid.shape[0] - 1):
            
            if rho_grid[i, j] == 0:
                break

            gradV = (V_grid[i + 1, j] - V_grid[i, j])
            gradP = -rho_grid[i, j]*gradV
            
            # Core
            if P_grid[i, j] > P_s and P_grid[i, j] > P_i:
                P_grid[i + 1, j] = P_grid[i, j] + gradP
                if P_grid[i + 1, j] > P_s:
                    rho_grid[i + 1, j] = _find_rho(P_grid[i + 1, j], mat_id_core, T_rho_id_core, T_rho_args_core,
                                                   rho_s - 10, rho_grid[i, j], ucold_array_core)
                else:
                    P_grid[i + 1, j] = P_s
                    rho_grid[i + 1, j] = rho_s
            # Mantle  
            elif P_grid[i, j] > P_s and P_grid[i, j] < P_i:
                P_grid[i + 1, j] = P_grid[i, j] + gradP
                if P_grid[i + 1, j] > P_s:
                    rho_grid[i + 1, j] = _find_rho(P_grid[i + 1, j], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                                   rho_s - 10, rho_grid[i, j], ucold_array_mantle)
                else:
                    P_grid[i + 1, j] = P_s
                    rho_grid[i + 1, j] = rho_s
            else:
                P_grid[i + 1, j] = 0.
                rho_grid[i + 1, j] = 0.
                break
        
    rho_raw = comm.gather(rho_grid)
    rho_grid = np.zeros((r_array.shape[0], z_array.shape[0]))
    
    if rank == 0:
        for I in range(size):
            rho_grid[:,:] = rho_grid[:,:] + rho_raw[I]
        #V_grid = np.flip(V_grid, 1)
        
    rho_grid = comm.bcast(rho_grid, root=0)
    
    rho_grid[0,:] = rho_grid[0,:]/size
    
    return rho_grid

    
def spin2layer(iterations, radii, densities, Tw,
               mat_id_core, T_rho_id_core, T_rho_args_core,
               mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
               P_c, P_i, P_s, rho_c, rho_s, r_array = None, z_array = None):
    """
    Spining of a radial planetary profile
    
    Args:
        iterations (int):
            Number of iterations to compute.
            
        rho_sph ([float]):
            Initial density radial profile to start with (SI).
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        Tw (float):
            Period of the planet (hours).
            
        K, alpha (float):
            Parameters from the relation between density and temperature
            T = K*rho**alpha.
    
    Returns:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
    """
    
    times = np.zeros((iterations, 2))
    
    start = time.time()
    
    try:
        I_array = np.load('I_array.npy')
        
        if mat_id_core == 100:
            ucold_array_core = np.load('ucold_array_100.npy')
        elif mat_id_core == 101:
            ucold_array_core = np.load('ucold_array_101.npy')
        elif mat_id_core == 102:
            ucold_array_core = np.load('ucold_array_102.npy')
            
        if mat_id_mantle == 100:
            ucold_array_mantle = np.load('ucold_array_100.npy')
        elif mat_id_mantle == 101:
            ucold_array_mantle = np.load('ucold_array_101.npy')
        elif mat_id_mantle == 102:
            ucold_array_mantle = np.load('ucold_array_102.npy')
            
    except ImportError:
        return False
            
    #ucold_array_core = _create_ucold_array(mat_id_core)
    #ucold_array_mantle = _create_ucold_array(mat_id_mantle)
    #I_array = _create_I_array()
    
    rho_grid, r_array, z_array = _rho0_grid(radii, densities, r_array, z_array)
    rho = np.zeros((iterations + 1, r_array.shape[0], z_array.shape[0]))
    rho[0] = rho_grid
    
    end = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0: print(f"Running time for creating I_array and u_cold_table: {end - start:.2f} seconds \n")
    
    dS = _dS(r_array, z_array)
    
    for i in range(1, iterations + 1):
        
        start = time.time()
        V1 = _fillV_parallel(rho[i - 1], r_array, z_array, I_array, dS, Tw)
        end = time.time()
        times[i - 1, 0] = end - start
        start = time.time()
        rho[i] = _fillrho2_parallel(V1, r_array, z_array, P_c, P_i, P_s, rho_c, rho_s,
                                    mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core,
                                    mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle, ucold_array_mantle)
        end = time.time()
        times[i - 1, 1] = end - start
        
        if rank == 0: 
            print(f"Iteration {i} complete in {times[i-1,0]:.2f} + {times[i-1,1]:.2f} seconds")
            print(f"Total time: {times[i-1,0] + times[i-1,1]:.2f} seconds\n")
        
    return rho, r_array, z_array, times