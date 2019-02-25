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
    # Material constants for Tillotson EoS + material code and specific capacity (SI units)
    iron = np.array([0.5, 1.5, 1.279E11, 1.05E11, 7860, 9.5E6, 1.42E6, 8.45E6, 5, 5, 0, 449])
    granite = np.array([0.5, 1.3, 1.8E10, 1.8E10, 2700, 1.6E7, 3.5E6, 1.8E7, 5, 5, 1, 790])
    water = np.array([0.5, 0.9, 2.0E10, 1.0E10, 1000, 2.0E6, 4.0E5, 2.0E6, 5, 5, 2, 4186])
    
    if (mat_id == 100):
        material = iron
    elif (mat_id == 101):
        material = granite
    elif (mat_id == 102):
        material = water
    else:
        print("Material not implemented")
        return None
        
    a = material[0]
    b = material[1]
    A = material[2]
    B = material[3]
    rho0 = material[4]
    u0 = material[5]
    u1 = material[6]
    u2 = material[7]
    alpha = material[8]
    beta = material[9]

    eta = rho/rho0
    mu = eta - 1.

    if (rho >= rho0 or u <= u1):
        P = (a + b*u0*eta**2/(u0*eta**2 + u))*u*rho + A*mu + B*mu**2
        
    elif (rho <= rho0 and u >= u2):
        P21 = a*u*rho
        P221 = b*u*rho*u0*eta**2/(u + u0*eta**2)
        P222 = A*mu*np.exp(-beta*(1/eta - 1))
        P23 = np.exp(-alpha*(1/eta - 1)**2)
        P = P21 + (P221 + P222)*P23
        
    else:
        P1 = (a + b*u0*pow(eta, 2)/(u + u0*pow(eta, 2)))*u*rho + A*mu + B*pow(mu, 2)
        P21 = a*u*rho
        P221 = b*u*rho*u0*eta**2/(u + u0*eta**2)
        P222 = A*mu*np.exp(-beta*(1/eta - 1))
        P23 = np.exp(-alpha*(1/eta - 1)**2)
        P2 = P21 + (P221 + P222)*P23
        P = ((u - u1)*P2 + (u2 - u)*P1)/(u2 - u1)
    
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
    uc = 0

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
    rho2 = (rho0 + rho1)/2
    tolerance = 1E-7

    f0 = Ps - P_EoS((_ucold_tab(rho0, ucold_array) + c*T_rho(rho0, T_rho_id, T_rho_args)), rho0, mat_id)
    f1 = Ps - P_EoS((_ucold_tab(rho1, ucold_array) + c*T_rho(rho1, T_rho_id, T_rho_args)), rho1, mat_id)

    if f0*f1 > 0:
        #print("Cannot find a 0 in the interval\n")
        rho2 = rho1;
    else:
        while np.abs(rho1 - rho0) > tolerance:
            f0 = Ps - P_EoS((_ucold_tab(rho0, ucold_array) + c*T_rho(rho0, T_rho_id, T_rho_args)), rho0, mat_id)
            f1 = Ps - P_EoS((_ucold_tab(rho1, ucold_array) + c*T_rho(rho1, T_rho_id, T_rho_args)), rho1, mat_id)
            f2 = Ps - P_EoS((_ucold_tab(rho2, ucold_array) + c*T_rho(rho2, T_rho_id, T_rho_args)), rho2, mat_id)

            if f0*f2 > 0:
                rho0 = rho2
            elif f0*f2 < 0: 
                rho1 = rho2
            else:
                return rho2

            rho2 = (rho0 + rho1)/2

    return rho2;

# Especial functions for the initial spherical case

# Compute rho0 given the spherical radii profile (from high to low as in function before)
# Rs in Earth radii units
def _rho0_grid(radii_sph, densities_sph, r_array = None, z_array = None):
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
    
    if r_array == None:
        r_array = np.arange(0, 1.5*np.max(radii_sph), 1.5*np.max(radii_sph)/100)
        
    if z_array == None:
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
            Cylindrical radii (distance from z axis) (R_earth).
            
        z (float):
            z coordinate (R_earth).
            
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
            
            a2 = rcp*rcp + zp*zp + rc*rc + z*z
            alpha = 1 - 2*z*zp/(a2 + err)
            beta = 2*rc*rcp/(a2 + err)
            gamma = beta/alpha 
            
            Vsum = Vsum + rho_grid[i,j]*rcp*_I_tab(gamma, I_array)/np.sqrt(a2 + err)/np.sqrt(alpha)*S_grid[i,j]
            
            zp = -z_array[j]
            alpha = 1 - 2*z*zp/(a2 + err)
            gamma = beta/alpha      
            Vsum = Vsum + rho_grid[i,j]*rcp*_I_tab(gamma, I_array)/np.sqrt(a2 + err)/np.sqrt(alpha)*S_grid[i,j]
      
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

# attempt 2, from center to surface
@jit(nopython=True)
def _fillrho(V_grid, Rs, prev_rho_grid, rho_s, material, K, alpha, ucold_table):
    """
    Update 2-d density grid given a 2-d potential grid of the planet.
    
    Args:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        prev_rho_grid ([[float]]):
            2-d density grid (SI) to be updated.
            
        rho_s (float):
            Surface density of the planet (SI).
            
        material ([float]):
            Material constants (SI).
            
        K, alpha (float):
            Parameters from the relation between density and temperature
            T = K*rho**alpha.
        
        ucold_table ([[float]]):
            Tabulated values of cold internal energy.
            
    Returns:
        rho_grid ([[float]]):
            Updated 2-d density grid (SI).
    """
    
    N = V_grid.shape[1]
    rho_grid = np.zeros((2*N, N))
    Ps = P_rho(rho_s, material, K, alpha)
    
    rho_grid[0,0] = prev_rho_grid[0,0]
    #rho_grid[1,0] = prev_rho_grid[1,0]
    #rho_grid[0,1] = prev_rho_grid[0,1]
    #rho_grid[1,1] = prev_rho_grid[1,1]
    ##############
    gradV = -(V_grid[0, 1] - V_grid[0, 0])
    gradP = rho_grid[0, 0]*gradV
    P = gradP + P_rho(rho_grid[0, 0], material, K, alpha)
    rho_grid[0,1] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[0,0], ucold_table)
    
    gradV = -(V_grid[1, 0] - V_grid[0, 0])
    gradP = rho_grid[0, 0]*gradV
    P = gradP + P_rho(rho_grid[0, 0], material, K, alpha)
    rho_grid[1,0] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[0,0], ucold_table)
    
    gradV = -(V_grid[1, 1] - V_grid[1, 0])
    gradP = rho_grid[1, 0]*gradV
    P = gradP + P_rho(rho_grid[1, 0], material, K, alpha)
    rho_grid[1,1] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[0,0], ucold_table)
    
    ##############
    
    # fill polar radii
    for i in range(2):
        for j in range(1, rho_grid.shape[1] - 1):
            gradV = -(V_grid[i, j + 1] - V_grid[i, j - 1])
            gradP = rho_grid[i, j]*gradV
            P = gradP + P_rho(rho_grid[i, j - 1], material, K, alpha)
            if P >= Ps:
                rho_grid[i, j + 1] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[i,j], ucold_table)
            else:
                rho_grid[i, j + 1] = 0
                break       
        
    # fill the rest
    for j in range(0, rho_grid.shape[1]):
        for i in range(1, rho_grid.shape[0] - 1):
            if rho_grid[i - 1, j] == 0:
                break
            
            if (i >= 2 and i < 2*N - 2):
                gradV = -(- V_grid[i+2, j] + 8*V_grid[i+1,j] - 8*V_grid[i-1,j] + V_grid[i-2,j])/6
            else:    
                gradV = -(V_grid[i + 1, j] - V_grid[i - 1, j])
                
            P = rho_grid[i, j]*gradV
            P = P + P_rho(rho_grid[i - 1, j], material, K, alpha)
            if P >= Ps:
                rho_grid[i + 1, j] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[i,j], ucold_table)
            else:
                rho_grid[i + 1, j] = 0
                break
       
    return rho_grid


# Compute V grid (parallelized) ###########
@jit(nopython=True)
def _fillV_rows(rho_grid, Rs, I_array, lower, upper, rank, subsize, N, W, dr):
    """
    Partially complete the computation of the total potential.
    
    Args:
        rho_grid ([[float]]):
            2-d density grid (SI) of the planet.
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        I_array ([float]):
            Tabulated values of the integral of phi' for gamma from 0 to 1.
            
        lower (int):
            Row from which to start the computation.
            
        upper (int):
            Row from which to end the computation (not included).
            
        rank (int):
            Identifier of the processor.
            
        subsize (int):
            int(N/size).
            
        N (int):
            Number of integration steps from the initial density profile.
            
        W (float):
            Angular velocity (SI).
            
        dr (float):
            Rs/(N - 1).
            
    Returns:
        rows ([[float]]):
            2-d potential grid (SI) for the rows specified.
        
    """
    
    rows = np.zeros((2*N, upper - lower))
    for i in range(2*N):
        for j in range(upper - lower):
            rows[i, j] = _Vg(i*dr, (rank*subsize + j)*dr, rho_grid, Rs, I_array) - (1/2)*(W*i*dr*R_earth)**2
    
    return rows

def _fillV_par(rho_grid, Rs, Tw, I_array):
    """
    Parallel computation of 2-d potential grid given 
    a 2-d density grid of a rotating planet.
    
    Args:
        rho_grid ([[float]]):
            2-d density grid (SI) of the planet.
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        Tw (float):
            Period of the planet (hours).
            
        I_array ([float]):
            Tabulated values of the integral of phi' for gamma from 0 to 1.
    
    Returns:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
    
    """

    # parallel set-up
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    N = rho_grid.shape[1]
    W = 2*np.pi/Tw/60/60
    dr = Rs/(N - 1)
    
    # partitioning the work
    subsize = int(N/size)
    partition = [0, subsize]
    
    for i in range(2, size):
        partition.append(i*subsize)
    
    partition.append(N)
    partition = np.array(partition)
    
    # send the work to each precessor
    lower = partition[rank]
    upper = partition[rank + 1]
        
    V_subgrid = _fillV_rows(rho_grid, Rs, I_array, lower, upper, rank, subsize, N, W, dr)
    #V_subgrid = np.flip(V_subgrid, 1)
    
    V_raw = comm.gather(V_subgrid, root=0)
    V_grid = np.zeros((2*N, N))
    
    if rank == 0:
        for I in range(size):
            V_grid[:, partition[I]:partition[I + 1]] = V_raw[I]
        #V_grid = np.flip(V_grid, 1)
        
    V_grid = comm.bcast(V_grid, root=0)
    #V_grid = V_grid.reshape((2*N, N))
    
    return V_grid

###############################################################################
    
def spin(iterations, rho_sph, Rs, material, Tw, K, alpha):
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
    
    N = rho_sph.shape[0]
    rho_s = rho_sph[-1]
    I_array = _create_I_tab()
    ucold_table = _create_ucold_table()

    rho = np.zeros((iterations + 1, 2*N, N))
    rho[0] = rho0(rho_sph, Rs)
    
    end = time.time()
    print("Running time for creating I_array and u_cold_table:", end - start)
    
    for i in range(1, iterations + 1):
        
        start = time.time()
        V1 = _fillV_par(rho[i - 1], Rs, Tw, I_array)
        end = time.time()
        times[i - 1, 0] = end - start
        start = time.time()
        rho[i] = _fillrho(V1, Rs, rho[i - 1], rho_s, material, K, alpha, ucold_table)
        end = time.time()
        times[i - 1, 1] = end - start
        
    return rho, times