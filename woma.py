#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:18:01 2019

@author: Sergio Ruiz-Bonilla
"""

###############################################################################
####################### Libraries and constants ###############################
###############################################################################

import numpy as np
from numba import jit
from scipy.interpolate import interp1d
import seagen
import weos
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Global constants
G = 6.67408E-11;
R_earth = 6371000;
M_earth = 5.972E24;

###############################################################################
####################### Spherical profile functions ###########################
###############################################################################

def set_up():
    """ Creates tabulated values of cold internal energy, 
        and saves the results in the data folder
    
    """
    print('Creating u cold curve for material 100...\n')
    ucold_array_100 = weos._create_ucold_array(100)
    np.save("data/ucold_array_100", ucold_array_100)
    del ucold_array_100
    print('Creating u cold curve for material 101...\n')
    ucold_array_101 = weos._create_ucold_array(101)
    np.save("data/ucold_array_101", ucold_array_101)
    del ucold_array_101
    print('Creating u cold curve for material 102...\n')
    ucold_array_102 = weos._create_ucold_array(102)
    np.save("data/ucold_array_102", ucold_array_102)
    del ucold_array_102

      
############################## 1 layer ########################################
    
@jit(nopython=True)
def integrate_1layer(N, R, M, Ps, Ts, rhos, mat_id, T_rho_id, T_rho_args,
                     ucold_array):
    """ Integration of a 1 layer spherical planet.
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            mat_id (int):
                Material id.
                
            T_rho_id (int)
                Relation between T and rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
            ucold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() (SI).
                
        Returns:
            
            r ([float]):
                Array of radii (SI).
                
            m ([float]):
                Array of cumulative mass (SI).
                
            P ([float]):
                Array of pressures (SI).
                
            T ([float]):
                Array of temperatures (SI).
                
            rho ([float]):
                Array of densities (SI).
                
            u ([float]):
                Array of internal energy (SI).
                
            mat ([float]):
                Array of material ids (SI).
            
    """
    
    r   = np.linspace(R, 0, int(N))
    m   = np.zeros(r.shape)
    P   = np.zeros(r.shape)
    T   = np.zeros(r.shape)
    rho = np.zeros(r.shape)
    u   = np.zeros(r.shape)
    mat = np.ones(r.shape)*mat_id
    
    c = weos._spec_c(mat_id)
    
    us = weos.ucold(rhos, mat_id, 10000) + c*Ts
    if T_rho_id == 1:
        T_rho_args[0] = Ts*rhos**(-T_rho_args[1])
    
    dr = r[0] - r[1]
    
    m[0]    = M
    P[0]    = Ps
    T[0]    = Ts
    rho[0]  = rhos
    u[0]    = us
    
    for i in range(1, r.shape[0]):
            
        m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
        P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
        rho[i] = weos._find_rho(P[i], mat_id, T_rho_id, T_rho_args,
                               rho[i - 1], 1.1*rho[i - 1], ucold_array)
        T[i]   = weos.T_rho(rho[i], T_rho_id, T_rho_args)
        u[i]   = weos._ucold_tab(rho[i], mat_id, ucold_array) + c*T[i]
        
        if m[i] < 0:
            
            return r, m, P, T, rho, u, mat
        
    return r, m, P, T, rho, u, mat

@jit(nopython=True)
def find_mass_1layer(N, R, M_max, Ps, Ts, rhos, mat_id, T_rho_id, T_rho_args,
                     ucold_array):
    
    """ Finder of the total mass of the planet.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M_max (float):
                Upper bound for the mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            mat_id (int):
                Material id.
                
            T_rho_id (int)
                Relation between T and rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
            ucold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() (SI).
                
        Returns:
            
            M_max (float):
                Mass of the planet (SI).
            
    """
    
    M_min = 0.
    
    r, m, P, T, rho, u, mat = integrate_1layer(N, R, M_max, Ps, Ts, rhos, mat_id,
                                               T_rho_id, T_rho_args, ucold_array)
    
    if m[-1] > 0.:
        
        while np.abs(M_min - M_max) > 1e-10*M_min:
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_1layer(N, R, M_try, Ps, Ts, rhos, mat_id,
                                   T_rho_id, T_rho_args, ucold_array)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

#@jit(nopython=True)
def find_radius_1layer(N, R_max, M, Ps, Ts, rhos, mat_id, T_rho_id, T_rho_args,
                       ucold_array, iterations = 40):
    
    """ Finder of the total radius of the planet.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Maximuum radius of the planet (SI).
                
            M_max (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            mat_id (int):
                Material id.
                
            T_rho_id (int)
                Relation between T and rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
            ucold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() (SI).
                
        Returns:
            
            M_max (float):
                Mass of the planet (SI).
            
    """
    
    R_min = 0.
    
    r, m, P, T, rho, u, mat = integrate_1layer(N, R_max, M, Ps, Ts, rhos, mat_id,
                                               T_rho_id, T_rho_args, ucold_array)
    
    if m[-1] == 0.:
        
        for i in tqdm(range(iterations), desc="Finding R given M"):
            
            R_try = (R_min + R_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_1layer(N, R_try, M, Ps, Ts, rhos, mat_id,
                                   T_rho_id, T_rho_args, ucold_array)
            
            if m[-1] > 0.:
                R_min = R_try
            else:
                R_max = R_try
                
    else:
        print("R_max is too low, did not ran out of mass in first iteration")
        return 0.
        
    return R_min
     
############################## 2 layers #######################################
    
@jit(nopython=True)
def integrate_2layer(N, R, M, Ps, Ts, rhos, Bcm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     ucold_array_core, ucold_array_mantle):
    """ Integration of a 2 layer spherical planet.
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            Bcm (float):
                Boundary core-mantle (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
        Returns:
            
            r ([float]):
                Array of radii (SI).
                
            m ([float]):
                Array of cumulative mass (SI).
                
            P ([float]):
                Array of pressures (SI).
                
            T ([float]):
                Array of temperatures (SI).
                
            rho ([float]):
                Array of densities (SI).
                
            u ([float]):
                Array of internal energy (SI).
                
            mat ([float]):
                Array of material ids (SI).
            
    """
    r   = np.linspace(R, 0, int(N))
    m   = np.zeros(r.shape)
    P   = np.zeros(r.shape)
    T   = np.zeros(r.shape)
    rho = np.zeros(r.shape)
    u   = np.zeros(r.shape)
    mat = np.zeros(r.shape)
    
    c_core   = weos._spec_c(mat_id_core)
    c_mantle = weos._spec_c(mat_id_mantle)
    
    us = weos.ucold(rhos, mat_id_mantle, 10000) + c_mantle*Ts
    if T_rho_id_mantle == 1:
        T_rho_args_mantle[0] = Ts*rhos**(-T_rho_args_mantle[1])
    
    dr = r[0] - r[1]
    
    m[0]    = M
    P[0]    = Ps
    T[0]    = Ts
    rho[0]  = rhos
    u[0]    = us
    mat[0]  = mat_id_mantle 
    
    for i in range(1, r.shape[0]):
            
        # mantle
        if r[i] > Bcm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = weos._find_rho(P[i], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_mantle)
            T[i]   = weos.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
            u[i]   = weos._ucold_tab(rho[i], mat_id_mantle, ucold_array_mantle) + c_mantle*T[i]
            mat[i] = mat_id_mantle
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
        # boundary core mantle
        elif r[i] <= Bcm and r[i - 1] > Bcm:
            
            rho_transition = weos._find_rho_fixed_P_T(P[i - 1], T[i - 1],
                                                     mat_id_core, ucold_array_core)
            
            if T_rho_id_core == 1:
                T_rho_args_core[0] = T[i - 1]*rho_transition**(-T_rho_args_core[1])
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho_transition*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho_transition/(r[i - 1]**2)*dr
            rho[i] = weos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho_transition, ucold_array_core)
            T[i]   = weos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = weos._ucold_tab(rho[i], mat_id_core, ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
        # core  
        elif r[i] <= Bcm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = weos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_core)
            T[i]   = weos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = weos._ucold_tab(rho[i], mat_id_core, ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
    return r, m, P, T, rho, u, mat

@jit(nopython=True)
def find_mass_2layer(N, R, M_max, Ps, Ts, rhos, Bcm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     ucold_array_core, ucold_array_mantle):
    """ Finder of the total mass of the planet.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M_max (float):
                Upper bound for the mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            Bcm (float):
                Boundary core-mantle (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
        Returns:
            
            M_max ([float]):
                Mass of the planet (SI).
            
    """
    M_min = 0.
    
    r, m, P, T, rho, u, mat = \
        integrate_2layer(N, R, M_max, Ps, Ts, rhos, Bcm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_core, ucold_array_mantle)
    
    if m[-1] > 0.:
        
        while np.abs(M_min - M_max) > 1e-10*M_min:
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_2layer(N, R, M_try, Ps, Ts, rhos, Bcm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_core, ucold_array_mantle)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

#@jit(nopython=True)
def find_radius_2layer(N, R_max, M, Ps, Ts, rhos, Bcm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     ucold_array_core, ucold_array_mantle, iterations = 40):
    """ Finder of the total radius of the planet.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R_max (float):
                Maximum radius of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            Bcm (float):
                Boundary core-mantle (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
        Returns:
            
            M_max ([float]):
                Mass of the planet (SI).
            
    """
    R_min = Bcm
    
    r, m1, P, T, rho, u, mat = \
        integrate_2layer(N, R_max, M, Ps, Ts, rhos, Bcm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_core, ucold_array_mantle)
    
    rhos_core = weos._find_rho_fixed_P_T(Ps, Ts, mat_id_core, ucold_array_core)
    
    r, m2, P, T, rho, u, mat = \
        integrate_1layer(N, Bcm, M, Ps, Ts, rhos_core,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         ucold_array_core)
        
    if m1[-1] > 0:
        print("R_max too low, excess of mass for R = R_max")
        return R_max
    
    if m2[-1] == 0:
        print("R = Bcm yields a planet which already lacks mass.")
        print("Try increase M or reduce Bcm.")
        return -1
    
    if m1[-1] == 0.:
        
        for i in tqdm(range(iterations), desc="Finding R given M, B"):
            
            R_try = (R_min + R_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_2layer(N, R_try, M, Ps, Ts, rhos, Bcm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_core, ucold_array_mantle)
            
            if m[-1] > 0.:
                R_min = R_try
            else:
                R_max = R_try
        
    return R_min

#@jit(nopython=True)
def find_boundary_2layer(N, R, M, Ps, Ts, rhos,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_core, ucold_array_mantle, iterations = 40):
    """ Finder of the boundary of the planet.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Temperature at the surface (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
        Returns:
            
            b_min ([float]):
                Boundary of the planet (SI).
            
    """
    
    B_min = 0.
    B_max = R 
    
    rhos_mantle = weos._find_rho_fixed_P_T(Ps, Ts, mat_id_mantle, ucold_array_mantle)
    
    r, m1, P, T, rho, u, mat = \
        integrate_1layer(N, R, M, Ps, Ts, rhos_mantle,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_mantle)
        
    rhos_core = weos._find_rho_fixed_P_T(Ps, Ts, mat_id_core, ucold_array_core)
    
    r, m2, P, T, rho, u, mat = \
        integrate_1layer(N, R, M, Ps, Ts, rhos_core,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         ucold_array_core)
        
    if m1[-1] == 0:
        print("Ran out of mass for a planet made of mantle material")
        print("Try increasing the mass or decreasing the radius")
        return B_min
    
    if m2[-1] > 0:
        print("Excess of mass for a planet made of core material")
        print("Try decreasing the mass or increasing the radius")
        return B_max
    
    if m1[-1] > 0.:
        
        for i in tqdm(range(iterations), desc="Finding B given R, M"):
            
            B_try = (B_min + B_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_2layer(N, R, M, Ps, Ts, rhos, B_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_core, ucold_array_mantle)
            
            if m[-1] > 0.:
                B_min = B_try
            else:
                B_max = B_try
        
    return B_min

############################## 3 layers #######################################
    
@jit(nopython=True)
def integrate_3layer(N, R, M, Ps, Ts, rhos, Bcm, Bma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
    """ Integration of a 2 layer spherical planet.
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            Bcm (float):
                Boundary core-mantle (SI).
                
            Bma (float):
                Boundary mantle-atmosphere (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            mat_id_atm (int):
                Material id for the atmosphere.
                
            T_rho_id_atm (int)
                Relation between T and rho to be used at the atmosphere.
                
            T_rho_args_atm (list):
                Extra arguments to determine the relation at the atmosphere.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
            ucold_array_atm ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the atmosphere (SI).
                
        Returns:
            
            r ([float]):
                Array of radii (SI).
                
            m ([float]):
                Array of cumulative mass (SI).
                
            P ([float]):
                Array of pressures (SI).
                
            T ([float]):
                Array of temperatures (SI).
                
            rho ([float]):
                Array of densities (SI).
                
            u ([float]):
                Array of internal energy (SI).
                
            mat ([float]):
                Array of material ids (SI).
            
    """
    
    r   = np.linspace(R, 0, int(N))
    m   = np.zeros(r.shape)
    P   = np.zeros(r.shape)
    T   = np.zeros(r.shape)
    rho = np.zeros(r.shape)
    u   = np.zeros(r.shape)
    mat = np.zeros(r.shape)
    
    c_core   = weos._spec_c(mat_id_core)
    c_mantle = weos._spec_c(mat_id_mantle)
    c_atm    = weos._spec_c(mat_id_atm)
    
    us = weos.ucold(rhos, mat_id_atm, 10000) + c_atm*Ts
    if T_rho_id_atm == 1:
        T_rho_args_atm[0] = Ts*rhos**(-T_rho_args_atm[1])        
    
    dr = r[0] - r[1]
    
    m[0]    = M
    P[0]    = Ps
    T[0]    = Ts
    rho[0]  = rhos
    u[0]    = us
    mat[0]  = mat_id_atm
    
    for i in range(1, r.shape[0]):
            
        # atm
        if r[i] > Bma:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = weos._find_rho(P[i], mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_atm)
            T[i]   = weos.T_rho(rho[i], T_rho_id_atm, T_rho_args_atm)
            u[i]   = weos._ucold_tab(rho[i], mat_id_atm, ucold_array_atm) + c_atm*T[i]
            mat[i] = mat_id_atm
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
        # boundary mantle atm
        elif r[i] <= Bma and r[i - 1] > Bma:
            
            rho_transition = weos._find_rho_fixed_P_T(P[i - 1], T[i - 1],
                                                     mat_id_mantle, ucold_array_mantle)
            
            if T_rho_id_mantle == 1:
                T_rho_args_mantle[0] = T[i - 1]*rho_transition**(-T_rho_args_mantle[1])
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho_transition*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho_transition/(r[i - 1]**2)*dr
            rho[i] = weos._find_rho(P[i], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                   rho[i - 1], 1.1*rho_transition, ucold_array_mantle)
            T[i]   = weos.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
            u[i]   = weos._ucold_tab(rho[i], mat_id_mantle, ucold_array_mantle) + c_mantle*T[i]
            mat[i] = mat_id_mantle
            
        # mantle
        elif r[i] > Bcm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = weos._find_rho(P[i], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_mantle)
            T[i]   = weos.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
            u[i]   = weos._ucold_tab(rho[i], mat_id_mantle, ucold_array_mantle) + c_mantle*T[i]
            mat[i] = mat_id_mantle
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
        # boundary core mantle
        elif r[i] <= Bcm and r[i - 1] > Bcm:
            
            rho_transition = weos._find_rho_fixed_P_T(P[i - 1], T[i - 1],
                                                     mat_id_core, ucold_array_core)
            
            if T_rho_id_core == 1:
                T_rho_args_core[0] = T[i - 1]*rho_transition**(-T_rho_args_core[1])
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho_transition*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho_transition/(r[i - 1]**2)*dr
            rho[i] = weos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho_transition, ucold_array_core)
            T[i]   = weos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = weos._ucold_tab(rho[i], mat_id_core, ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
        # core  
        elif r[i] <= Bcm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = weos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_core)
            T[i]   = weos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = weos._ucold_tab(rho[i], mat_id_core, ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
    return r, m, P, T, rho, u, mat


@jit(nopython=True)
def find_mass_3layer(N, R, M_max, Ps, Ts, rhos, Bcm, Bma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
    """ Finder of the total mass of the planet.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            Bcm (float):
                Boundary core-mantle (SI).
                
            Bma (float):
                Boundary mantle-atmosphere (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            mat_id_atm (int):
                Material id for the atmosphere.
                
            T_rho_id_atm (int)
                Relation between T and rho to be used at the atmosphere.
                
            T_rho_args_atm (list):
                Extra arguments to determine the relation at the atmosphere.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
            ucold_array_atm ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the atmosphere (SI).
                
        Returns:
            
            M_max ([float]):
                Mass of the planet (SI).
            
    """
    
    M_min = 0.
    
    r, m, P, T, rho, u, mat = \
        integrate_3layer(N, R, M_max, Ps, Ts, rhos, Bcm, Bma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
    
    if m[-1] > 0.:
        
        while np.abs(M_min - M_max) > 1e-10*M_min:
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M_try, Ps, Ts, rhos, Bcm, Bma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

#@jit(nopython=True)
def find_radius_3layer(N, R_max, M, Ps, Ts, rhos, Bcm, Bma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm,
                     iterations=40):
    """ Finder of the total mass of the planet.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R_max (float):
                Maximum radius of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            Bcm (float):
                Boundary core-mantle (SI).
                
            Bma (float):
                Boundary mantle-atmosphere (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            mat_id_atm (int):
                Material id for the atmosphere.
                
            T_rho_id_atm (int)
                Relation between T and rho to be used at the atmosphere.
                
            T_rho_args_atm (list):
                Extra arguments to determine the relation at the atmosphere.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
            ucold_array_atm ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the atmosphere (SI).
                
        Returns:
            
            M_max ([float]):
                Mass of the planet (SI).
            
    """
    if Bcm > Bma:
        print("Bcm should not be greater than Bma")
        return -1
    
    R_min = Bma
    
    rhos_mantle = weos._find_rho_fixed_P_T(Ps, Ts, mat_id_mantle, ucold_array_mantle)
    
    r, m, P, T, rho, u, mat = \
        integrate_2layer(N, Bma, M, Ps, Ts, rhos_mantle, Bcm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_core, ucold_array_mantle)
        
    if m[-1] == 0:
        print("Ran out of mass for a 2 layer planet with core and mantle.")
        print("Try increase the mass or reduce Bcm")
        return R_min
        
    r, m, P, T, rho, u, mat = \
        integrate_3layer(N, R_max, M, Ps, Ts, rhos, Bcm, Bma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
    if m[-1] > 0:
        print("Excess of mass for a 3 layer planet with R = R_max.")
        print("Try reduce the mass or increase R_max")
        return R_max
        
    for i in tqdm(range(iterations), desc="Finding R given M, Bcm, Bma"):
            
            R_try = (R_min + R_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R_try, M, Ps, Ts, rhos, Bcm, Bma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if m[-1] > 0.:
                R_min = R_try
            else:
                R_max = R_try
                

        
    return R_min

@jit(nopython=True)
def moi(r, rho):
    """ Computes moment of inertia for a planet with spherical symmetry.
        
        Args:
            r ([float]):
                Radii of the planet (SI).
                
            rho ([float]):
                Densities asociated with the radii (SI)
                
        Returns:
            
            MoI (float):
                Moment of inertia (SI).
    """
    
    dr  = np.abs(r[0] - r[1])
    r4  = np.power(r, 4)
    MoI = 2*np.pi*(4/3)*np.sum(r4*rho)*dr
    
    return MoI

#@jit(nopython=True)
def find_Bma_3layer(N, R, M, Ps, Ts, rhos, Bcm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm,
                     iterations=40):
    """ Finder of the boundary mantle-atmosphere of the planet for
        fixed boundary core-mantle.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            Bcm (float):
                Boundary core-mantle (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            mat_id_atm (int):
                Material id for the atmosphere.
                
            T_rho_id_atm (int)
                Relation between T and rho to be used at the atmosphere.
                
            T_rho_args_atm (list):
                Extra arguments to determine the relation at the atmosphere.
                             
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
            ucold_array_atm ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the atmosphere (SI).
                
        Returns:
            
            b_ma_max ([float]):
                Boundary mantle-atmosphere of the planet (SI).
            
    """
    Bma_min = Bcm
    Bma_max = R
    
    r, m, P, T, rho, u, mat = \
        integrate_2layer(N, R, M, Ps, Ts, rhos, Bcm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_atm)
        
    if m[-1] == 0:
        print("A planet made of core and atm. materials lacks mass.")
        print("Try increasing the mass, increasing Bcm or decreasing R.") 
        return Bma_min
        
    rhos_mantle = weos._find_rho_fixed_P_T(Ps, Ts, mat_id_mantle, ucold_array_mantle)
    
    r, m, P, T, rho, u, mat = \
        integrate_2layer(N, R, M, Ps, Ts, rhos_mantle, Bcm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_core, ucold_array_mantle)
        
    if m[-1] > 0:
        print("A planet made of core and mantle materials excess mass.")  
        print("Try decreasing the mass, decreasing Bcm or increasing R")
        return Bma_max
    
    for i in tqdm(range(iterations), desc="Finding Bma given M, R, Bcm"):
            
        Bma_try = (Bma_min + Bma_max)/2.
        
        r, m, P, T, rho, u, mat = \
            integrate_3layer(N, R, M, Ps, Ts, rhos, Bcm, Bma_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        if m[-1] > 0.:
            Bma_min = Bma_try
        else:
            Bma_max = Bma_try
        
    return Bma_max

#@jit(nopython=True)
def find_Bcm_3layer(N, R, M, Ps, Ts, rhos, Bma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm,
                     iterations=40):
    """ Finder of the boundary mantle-atmosphere of the planet for
        fixed boundary core-mantle.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            Bma (float):
                Boundary mantle-atmosphere (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            mat_id_atm (int):
                Material id for the atmosphere.
                
            T_rho_id_atm (int)
                Relation between T and rho to be used at the atmosphere.
                
            T_rho_args_atm (list):
                Extra arguments to determine the relation at the atmosphere.
                             
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
            ucold_array_atm ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the atmosphere (SI).
                
        Returns:
            
            b_ma_max ([float]):
                Boundary mantle-atmosphere of the planet (SI).
            
    """
    Bcm_min = 0.
    Bcm_max = Bma
    
    r, m, P, T, rho, u, mat = \
        integrate_2layer(N, R, M, Ps, Ts, rhos, Bma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_atm)
        
    if m[-1] > 0:
        print("A planet made of core and atm. materials excess mass.")
        print("Try decreasing the mass, increasing Bma or increasing R")
        return Bcm_min
    
    r, m, P, T, rho, u, mat = \
        integrate_2layer(N, R, M, Ps, Ts, rhos, Bma,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_mantle, ucold_array_atm)
        
    if m[-1] == 0:
        print("A planet made of mantle and atm. materials lacks mass.")  
        print("Try increasing the mass, increasing Bma or decreasing R")
        return Bcm_max
    
    for i in tqdm(range(iterations), desc="Finding Bcm given R, M, Bma"):
            
        Bcm_try = (Bcm_min + Bcm_max)/2.
        
        r, m, P, T, rho, u, mat = \
            integrate_3layer(N, R, M, Ps, Ts, rhos, Bcm_try, Bma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        if m[-1] > 0.:
            Bcm_min = Bcm_try
        else:
            Bcm_max = Bcm_try
        
    return Bcm_max

#@jit(nopython=True)
def find_boundaries_3layer(N, R, M, Ps, Ts, rhos, MoI,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm,
                     iterations=20, subiterations=10):
    """ Finder of the boundaries of the planet for a
        fixed moment of inertia.
        The correct value yields m -> 0 at the center of the planet. 
    
        Args:
            N (int):
                Number of integration steps.
            
            R (float):
                Radii of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            Ps (float):
                Pressure at the surface (SI).
                
            Ts (float):
                Temperature at the surface (SI).
                
            rhos (float):
                Density at the surface (SI).
                
            MoI (float):
                moment of inertia (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            mat_id_atm (int):
                Material id for the atmosphere.
                
            T_rho_id_atm (int)
                Relation between T and rho to be used at the atmosphere.
                
            T_rho_args_atm (list):
                Extra arguments to determine the relation at the atmosphere.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
            ucold_array_atm ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the atmosphere (SI).
                
        Returns:
            
            b_cm_try, b_ma_try ([float]):
                Boundaries core-mantle and mantle-atmosphere of the planet (SI).
            
    """
    rhos_mantle = weos._find_rho_fixed_P_T(Ps, Ts, mat_id_mantle, ucold_array_mantle)
    
    Bcm_I_max = find_boundary_2layer(N, R, M, Ps, Ts, rhos_mantle,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     ucold_array_core, ucold_array_mantle)
    
    Bcm_I_min = find_boundary_2layer(N, R, M, Ps, Ts, rhos,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     ucold_array_core, ucold_array_atm)
    
    r_max, m, P, T, rho_max, u, mat = \
        integrate_2layer(N, R, M, Ps, Ts, rhos_mantle, Bcm_I_max,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         ucold_array_core, ucold_array_mantle)
        
    r_min, m, P, T, rho_min, u, mat = \
        integrate_2layer(N, R, M, Ps, Ts, rhos, Bcm_I_min,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_atm)
        
    moi_min = moi(r_min, rho_min)
    moi_max = moi(r_max, rho_max)
    
    Bcm_min = Bcm_I_max
    Bcm_max = Bcm_I_min
    
    if MoI > moi_min and  MoI < moi_max:
        
        for i in tqdm(range(iterations), desc="Finding Bcm, Bma given R, M, I"):
            
            Bcm_try = (Bcm_min + Bcm_max)/2.
            
            Bma_try = find_Bma_3layer(N, R, M, Ps, Ts, rhos, Bcm_try,
                             mat_id_core, T_rho_id_core, T_rho_args_core,
                             mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                             mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm,
                             subiterations)
                    
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M, Ps, Ts, rhos, Bcm_try, Bma_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if moi(r,rho) < MoI:
                Bcm_max = Bcm_try
            else:
                Bcm_min = Bcm_try
                
    elif MoI > moi_max:
        print("Moment of interia is too high,")
        print("maximum value is:")
        print(moi_max/M_earth/R_earth/R_earth,"[M_earth R_earth^2]")
        Bcm_try = 0.
        Bma_try = 0.
    
    elif MoI < moi_min:
        print("Moment of interia is too low,")
        print("minimum value is:")
        print(moi_min/M_earth/R_earth/R_earth,"[M_earth R_earth^2]")
        Bcm_try = 0.
        Bma_try = 0.
        
    else:
        print("Something went wrong")
        Bcm_try = 0.
        Bma_try = 0.
        
    return Bcm_try, Bma_try

###############################################################################
####################### Spherical profile classes #############################
###############################################################################

def _print_banner():
    
    print("\n")
    print("#  WoMa - World Maker")
    print("#  sergio.ruiz-bonilla@durham.ac.uk")
    print("\n")
    
class _planet():
    
    N_integ_steps = 10000
    iterations    = 40
    
    T_surface     = np.nan
    P_surface     = np.nan
    rho_surface   = np.nan
    
    def __init__(self):
        return None
    
    def set_T_surface(self, T):
        self.T_surface = T
        
    def set_P_surface(self, P):
        self.P_surface = P
        
    def set_rho_surface(self, rho):
        self.rho_surface = rho          
    
class _1l_planet(_planet):
    
    N_layers        = 1
    mat_id_core     = np.nan
    T_rho_id_core   = np.nan
    T_rho_args_core = np.nan
    
    M           = np.nan # Total mass of the planet (SI).
    R           = np.nan # Radius of the planet (SI).
    A1_R        = np.nan # Vector of radial distances (SI).
    A1_M        = np.nan # Vector of cumulative mass of the planet (SI).
    A1_P        = np.nan # Pressure profile (SI).
    A1_T        = np.nan # Temperature profile (SI).
    A1_rho      = np.nan # Density profile (SI).
    A1_u        = np.nan # Internal energy profile (SI).
    A1_material = np.nan # Material profile
    
    def __init__(self):
        return None
    
    def set_core_properties(self, mat_id_core, T_rho_id_core, T_rho_args_core):
        self.mat_id_core     = mat_id_core
        self.T_rho_id_core   = T_rho_id_core
        self.T_rho_args_core = np.array(T_rho_args_core, dtype='float')
    
    def fix_R_given_M(self, M, R_max):
        """ Computes the correct R for a given M.
        """
        ucold_array = weos.load_ucold_array(self.mat_id_core)
        R = find_radius_1layer(self.N_integ_steps, R_max, M,
                               self.P_surface, self.T_surface, self.rho_surface,
                               self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                               ucold_array, self.iterations)
        
        print("Tweeking M to avoid peaks at the center of the planet...")
        
        M_tweek = find_mass_1layer(self.N_integ_steps, R, 2*M,
                                   self.P_surface, self.T_surface, self.rho_surface,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   ucold_array)
        
        print("Done!")
        
        r, m, P, T, rho, u, mat = \
            integrate_1layer(self.N_integ_steps, R, M_tweek,
                             self.P_surface, self.T_surface, self.rho_surface,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             ucold_array)
         
        self.M           = M_tweek
        self.R           = R
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
    
    def fix_M_given_R(self, R, M_max):
        
        ucold_array = weos.load_ucold_array(self.mat_id_core)
        
        print("Finding M given R...")
        
        M = find_mass_1layer(self.N_integ_steps, R, M_max,
                             self.P_surface, self.T_surface, self.rho_surface,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             ucold_array)
        
        r, m, P, T, rho, u, mat = \
            integrate_1layer(self.N_integ_steps, R, M,
                             self.P_surface, self.T_surface, self.rho_surface,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             ucold_array)
            
        print("Done!")
            
        self.M           = M
        self.R           = R
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
    
    def compute_spherical_profile_given_R_M(self, R, M):
        
        ucold_array = weos.load_ucold_array(self.mat_id_core)
        
        r, m, P, T, rho, u, mat = \
            integrate_1layer(self.N_integ_steps, R, M,
                             self.P_surface, self.T_surface, self.rho_surface,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             ucold_array)
            
        self.M           = M
        self.R           = R
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat

class _2l_planet(_planet):
    
    N_layers          = 2
    mat_id_core       = np.nan
    T_rho_id_core     = np.nan
    T_rho_args_core   = np.nan
    mat_id_mantle     = np.nan
    T_rho_id_mantle   = np.nan
    T_rho_args_mantle = np.nan
    
    M           = np.nan # Total mass of the planet (SI).
    R           = np.nan # Radius of the planet (SI).
    Bcm         = np.nan # Boundary core-mantle (SI).
    A1_R        = np.nan # Vector of radial distances (SI).
    A1_M        = np.nan # Vector of cumulative mass of the planet (SI).
    A1_P        = np.nan # Pressure profile (SI).
    A1_T        = np.nan # Temperature profile (SI).
    A1_rho      = np.nan # Density profile (SI).
    A1_u        = np.nan # Internal energy profile (SI).
    A1_material = np.nan # Material profile
        
    def __init__(self):
        return None
    
    def set_core_properties(self, mat_id_core, T_rho_id_core, T_rho_args_core):  
        self.mat_id_core     = mat_id_core
        self.T_rho_id_core   = T_rho_id_core
        self.T_rho_args_core = np.array(T_rho_args_core, dtype='float')
        
    def set_mantle_properties(self, mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle): 
        self.mat_id_mantle     = mat_id_mantle
        self.T_rho_id_mantle   = T_rho_id_mantle
        self.T_rho_args_mantle = np.array(T_rho_args_mantle, dtype='float')
    
    def fix_B_given_R_M(self, R, M):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        
        Bcm = find_boundary_2layer(self.N_integ_steps, R, M,
                                   self.P_surface, self.T_surface, self.rho_surface,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   ucold_array_core, ucold_array_mantle, self.iterations)
        
        print("Tweeking M to avoid peaks at the center of the planet...")
        
        M_tweek = find_mass_2layer(self.N_integ_steps, R, 2*M,
                                   self.P_surface, self.T_surface, self.rho_surface, Bcm,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   ucold_array_core, ucold_array_mantle)
        
        r, m, P, T, rho, u, mat = \
            integrate_2layer(self.N_integ_steps, R, M_tweek,
                             self.P_surface, self.T_surface, self.rho_surface, Bcm,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             ucold_array_core, ucold_array_mantle)
            
        print("Done!")
            
        self.M           = M_tweek
        self.R           = R
        self.Bcm         = Bcm
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
            
    def fix_M_given_B_R(self, B, R, M_max):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        
        print("Finding M given B and R...")
        
        M = find_mass_2layer(self.N_integ_steps, R, M_max,
                             self.P_surface, self.T_surface, self.rho_surface, B,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             ucold_array_core, ucold_array_mantle)
        
        r, m, P, T, rho, u, mat = \
            integrate_2layer(self.N_integ_steps, R, M,
                             self.P_surface, self.T_surface, self.rho_surface, B,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             ucold_array_core, ucold_array_mantle)
            
        print("Done!")
            
        self.M           = M
        self.R           = R
        self.Bcm         = B
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
    
    def fix_R_given_M_B(self, M, B, R_max):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        
        R = find_radius_2layer(self.N_integ_steps, R_max, M,
                               self.P_surface, self.T_surface, self.rho_surface, B,
                               self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                               self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                               ucold_array_core, ucold_array_mantle, self.iterations)
        
        print("Tweeking M to avoid peaks at the center of the planet...")
        
        M_tweek = find_mass_2layer(self.N_integ_steps, R, 2*M,
                                   self.P_surface, self.T_surface, self.rho_surface, B,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   ucold_array_core, ucold_array_mantle)
        
        r, m, P, T, rho, u, mat = \
            integrate_2layer(self.N_integ_steps, R, M_tweek,
                             self.P_surface, self.T_surface, self.rho_surface, B,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             ucold_array_core, ucold_array_mantle)
            
        print("Done!")
            
        self.M           = M_tweek
        self.R           = R
        self.Bcm         = B
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
    
    def compute_spherical_profile_given_R_M_B(self, R, M, B):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        
        r, m, P, T, rho, u, mat = \
            integrate_2layer(self.N_integ_steps, R, M,
                             self.P_surface, self.T_surface, self.rho_surface, B,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             ucold_array_core, ucold_array_mantle)
            
        self.M           = M
        self.R           = R
        self.Bcm         = B
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat

class _3l_planet(_planet):
    
    N_layers          = 3
    mat_id_core       = np.nan 
    T_rho_id_core     = np.nan
    T_rho_args_core   = np.nan
    mat_id_mantle     = np.nan
    T_rho_id_mantle   = np.nan
    T_rho_args_mantle = np.nan
    mat_id_atm        = np.nan
    T_rho_id_atm      = np.nan
    T_rho_args_atm    = np.nan
    
    subiterations = 20
    
    M           = np.nan # Total mass of the planet (SI).
    R           = np.nan # Radius of the planet (SI).
    Bcm         = np.nan # Boundary core-mantle (SI).
    Bma         = np.nan # Boundary mantle-atmosphere (SI).
    I           = np.nan # Moment of inertia (SI).
    A1_R        = np.nan # Vector of radial distances (SI).
    A1_M        = np.nan # Vector of cumulative mass of the planet (SI).
    A1_P        = np.nan # Pressure profile (SI).
    A1_T        = np.nan # Temperature profile (SI).
    A1_rho      = np.nan # Density profile (SI).
    A1_u        = np.nan # Internal energy profile (SI).
    A1_material = np.nan # Material profile
        
    def __init__(self):
        return None
    
    def set_core_properties(self, mat_id_core, T_rho_id_core, T_rho_args_core):  
        self.mat_id_core     = mat_id_core
        self.T_rho_id_core   = T_rho_id_core
        self.T_rho_args_core = np.array(T_rho_args_core, dtype='float')
        
    def set_mantle_properties(self, mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle): 
        self.mat_id_mantle     = mat_id_mantle
        self.T_rho_id_mantle   = T_rho_id_mantle
        self.T_rho_args_mantle = np.array(T_rho_args_mantle, dtype='float')
        
    def set_atmosphere_properties(self, mat_id_atm, T_rho_id_atm, T_rho_args_atm): 
        self.mat_id_atm     = mat_id_atm
        self.T_rho_id_atm   = T_rho_id_atm
        self.T_rho_args_atm = np.array(T_rho_args_atm, dtype='float')
    
    def fix_Bcm_Bma_given_R_M_I(self, R, M, I):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        ucold_array_atm    = weos.load_ucold_array(self.mat_id_atm)
        
        Bcm, Bma = \
            find_boundaries_3layer(self.N_integ_steps, R, M,
                                   self.P_surface, self.T_surface, self.rho_surface, I, 
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                                   ucold_array_core, ucold_array_mantle, ucold_array_atm,
                                   self.iterations, self.subiterations)

        print("Tweeking M to avoid peaks at the center of the planet...")
        
        M_tweek = find_mass_3layer(self.N_integ_steps, R, 2*M,
                                   self.P_surface, self.T_surface, self.rho_surface,
                                   Bcm, Bma,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                                   ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
        
        r, m, P, T, rho, u, mat = \
            integrate_3layer(self.N_integ_steps, R, M_tweek,
                             self.P_surface, self.T_surface, self.rho_surface,
                             Bcm, Bma,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        print("Done!")
            
        self.M           = M_tweek
        self.R           = R
        self.Bcm         = Bcm
        self.Bma         = Bma
        self.I           = moi(r, rho)
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
        
    
    def fix_Bma_given_R_M_Bcm(self, R, M, Bcm):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        ucold_array_atm    = weos.load_ucold_array(self.mat_id_atm)
        
        Bma = find_Bma_3layer(self.N_integ_steps, R, M,
                              self.P_surface, self.T_surface, self.rho_surface, Bcm, 
                              self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                              self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                              self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                              ucold_array_core, ucold_array_mantle, ucold_array_atm,
                              self.iterations)
        
        print("Tweeking M to avoid peaks at the center of the planet...")
        
        M_tweek = find_mass_3layer(self.N_integ_steps, R, 2*M,
                                   self.P_surface, self.T_surface, self.rho_surface,
                                   Bcm, Bma,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                                   ucold_array_core, ucold_array_mantle, ucold_array_atm)

        r, m, P, T, rho, u, mat = \
            integrate_3layer(self.N_integ_steps, R, M_tweek,
                             self.P_surface, self.T_surface, self.rho_surface,
                             Bcm, Bma,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        print("Done!")
            
        self.M           = M_tweek
        self.R           = R
        self.Bcm         = Bcm
        self.Bma         = Bma
        self.I           = moi(r, rho)
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
    
    def fix_Bcm_given_R_M_Bma(self, R, M, Bma):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        ucold_array_atm    = weos.load_ucold_array(self.mat_id_atm)
        
        Bcm = find_Bcm_3layer(self.N_integ_steps, R, M,
                              self.P_surface, self.T_surface, self.rho_surface, Bma, 
                              self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                              self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                              self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                              ucold_array_core, ucold_array_mantle, ucold_array_atm,
                              self.iterations)
        
        print("Tweeking M to avoid peaks at the center of the planet...")
        
        M_tweek = find_mass_3layer(self.N_integ_steps, R, 2*M,
                                   self.P_surface, self.T_surface, self.rho_surface,
                                   Bcm, Bma,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                                   ucold_array_core, ucold_array_mantle, ucold_array_atm)

        r, m, P, T, rho, u, mat = \
            integrate_3layer(self.N_integ_steps, R, M_tweek,
                             self.P_surface, self.T_surface, self.rho_surface,
                             Bcm, Bma,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        print("Done!")
            
        self.M           = M_tweek
        self.R           = R
        self.Bcm         = Bcm
        self.Bma         = Bma
        self.I           = moi(r, rho)
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
    
    def fix_M_given_R_Bcm_Bma(self, R, Bcm, Bma, M_max):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        ucold_array_atm    = weos.load_ucold_array(self.mat_id_atm)
        
        print("Finding M given Bcm, Bma and R...")
        
        M = find_mass_3layer(self.N_integ_steps, R, M_max,
                             self.P_surface, self.T_surface, self.rho_surface,
                             Bcm, Bma,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm)

        r, m, P, T, rho, u, mat = \
            integrate_3layer(self.N_integ_steps, R, M,
                             self.P_surface, self.T_surface, self.rho_surface,
                             Bcm, Bma,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        print("Done!")
        
        self.M           = M
        self.R           = R
        self.Bcm         = Bcm
        self.Bma         = Bma
        self.I           = moi(r, rho)
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
    
    def fix_R_given_M_Bcm_Bma(self, M, Bcm, Bma, R_max):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        ucold_array_atm    = weos.load_ucold_array(self.mat_id_atm)
        
        R = find_radius_3layer(self.N_integ_steps, R_max, M,
                               self.P_surface, self.T_surface, self.rho_surface,
                               Bcm, Bma, 
                               self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                               self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                               self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                               ucold_array_core, ucold_array_mantle, ucold_array_atm,
                               self.iterations)
        
        print("Tweeking M to avoid peaks at the center of the planet...")
        
        M_tweek = find_mass_3layer(self.N_integ_steps, R, 2*M,
                                   self.P_surface, self.T_surface, self.rho_surface,
                                   Bcm, Bma,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                                   ucold_array_core, ucold_array_mantle, ucold_array_atm)

        r, m, P, T, rho, u, mat = \
            integrate_3layer(self.N_integ_steps, R, M_tweek,
                             self.P_surface, self.T_surface, self.rho_surface,
                             Bcm, Bma,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        print("Done!")
            
        self.M           = M_tweek
        self.R           = R
        self.Bcm         = Bcm
        self.Bma         = Bma
        self.I           = moi(r, rho)
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
    
    def compute_spherical_profile_given_R_M_Bcm_Bma(self, R, M, Bcm, Bma):
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        ucold_array_atm    = weos.load_ucold_array(self.mat_id_atm)
        
        r, m, P, T, rho, u, mat = \
            integrate_3layer(self.N_integ_steps, R, M,
                             self.P_surface, self.T_surface, self.rho_surface,
                             Bcm, Bma,
                             self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                             self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                             self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        self.M           = M
        self.R           = R
        self.Bcm         = Bcm
        self.Bma         = Bma
        self.I           = moi(r, rho)
        self.A1_R        = r
        self.A1_M        = m
        self.A1_P        = P
        self.A1_T        = T
        self.A1_rho      = rho
        self.A1_u        = u
        self.A1_material = mat
    
    
def Planet(N_layers):
        
    _print_banner()
    
    if N_layers not in [1, 2, 3]:
        print(f"Can't build a planet with {N_layers} layers!")
        return None
        
    if N_layers == 1:
        print("For a 1 layer planet, please specify:")
        print("pressure, temperature and density at the surface of the planet,")
        print("material, relation between temperature and density with any desired aditional parameters,")
        print("for the core of the planet.")
        planet = _1l_planet()
        return planet
            
    elif N_layers == 2:
        print("For a 2 layer planet, please specify:")
        print("pressure, temperature and density at the surface of the planet,")
        print("materials, relations between temperature and density with any desired aditional parameters,")
        print("for the core and mantle of the planet.")
        planet = _2l_planet()
        return planet
        
    elif N_layers == 3:
        print("For a 3 layer planet, please specify:")
        print("pressure, temperature and density at the surface of the planet,")
        print("materials, relations between temperature and density with any desired aditional parameters,")
        print("for the core, mantle and atmosphere of the planet.")
        planet = _3l_planet()
        return planet


        

###############################################################################
####################### Spining profile functions #############################
###############################################################################

@jit(nopython=True)
def _analytic_solution_r(r, R, Z, x):
    """ Indefinite integral, analytic solution of the optential
        of an oblate spheroid evaluated at x with z = 0.
    
        Args:
            
            r (float):
                Cylindrical r coordinate where to compute the potential (SI).
                
            R (float):
                Mayor axis of the oblate spheroid (SI). 
                
            Z (float):
                Minor axis of the oblate spheroid (SI). 
                
            x (float):
                Integration variable (SI). 
    """
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
    """ Indefinite integral, analytic solution of the optential
        of an oblate spheroid evaluated at x with r = 0.
    
        Args:
            
            z (float):
                Cylindrical z coordinate where to compute the potential (SI).
                
            R (float):
                Mayor axis of the oblate spheroid (SI). 
                
            Z (float):
                Minor axis of the oblate spheroid (SI). 
                
            x (float):
                Integration variable (SI). 
    """
    
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
    """ Gravitational potential due to an oblate spheroid with constant density
        at r, theta = 0, z = 0.
        
        Args:
            
            r (float):
                Cylindrical r coordinate where to compute the optential (SI).
                
            R (float):
                Mayor axis of the oblate spheroid (SI).
                
            Z (float):
                Minor axis of the oblate spheroid (SI). 
                
            rho (float):
                Density of the spheroid (SI).
                
        Returns:
            V (float):
                Gravitational potential (SI).
    """
    
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
    """ Gravitational potential due to an oblate spheroid with constant density
        at r = 0, theta = 0, z.
        
        Args:
            
            z (float):
                Cylindrical z coordinate where to compute the optential (SI).
                
            R (float):
                Mayor axis of the oblate spheroid (SI).
                
            Z (float):
                Minor axis of the oblate spheroid (SI). 
                
            rho (float):
                Density of the spheroid (SI).
                
        Returns:
            V (float):
                Gravitational potential (SI).
    """    
    
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
def _el_eq(r, z, R, Z):
    return r*r/R/R + z*z/Z/Z

@jit(nopython=False)
def rho_rz(r, z, r_array, rho_e, z_array, rho_p):
    """ Computes the density at any point r, z given a spining profile.
    
        Args:
            
            r (float):
                Cylindrical r coordinte where to compute the density (SI).
            
            z (float):
                Cylindrical z coordinte where to compute the density (SI).
                
            r_array ([float]):
                Points at equatorial profile where the solution is defined (SI).
                
            rho_e ([float]):
                Equatorial profile of densities (SI).
                
            z_array ([float]):
                Points at equatorial profile where the solution is defined (SI).
                
            rho_p ([float]):
                Polar profile of densities (SI).
                
        Returns:
            
            rho_2 (float):
                Density at r, z (SI).
    
    """
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
    
############################## 1 layer ########################################
    
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
            V_e[j] += _Vgr(r_array[j], R_array[i], 
                           Z_array[i], delta_rho)                      
            
        for j in range(V_p.shape[0]):
            V_p[j] += _Vgz(z_array[j], R_array[i], 
                           Z_array[i], delta_rho)
            
    for i in range(V_e.shape[0]):
        V_e[i] += -(1/2)*(W*r_array[i])**2
        
    return V_e, V_p

@jit(nopython=True)
def _fillrho1(r_array, V_e, z_array, V_p, P_c, P_s, rho_c, rho_s,
             mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core):
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
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
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
        #print(i)
            
        if P_e[i + 1] >= P_s:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                         rho_s - 10, rho_e[i], ucold_array_core) 
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_p[i], ucold_array_core)
        else:
            rho_p[i + 1] = 0.
            break
        
    return rho_e, rho_p

def spin1layer(iterations, r_array, z_array, radii, densities, Tw,
               P_c, P_s, rho_c, rho_s,
               mat_id_core, T_rho_id_core, T_rho_args_core,
               ucold_array_core):
    """ Compute spining profile of densities for a 1 layer planet.
    
        Args:
            
            iterations (int):
                Number of iterations to run.
                
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
            
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
        Returns:
            
            profile_e ([[float]]):
                List of the iterations of the equatorial density profile (SI).
                
            profile_p ([[float]]):
                List of the iterations of the polar density profile (SI).
    
    """
        
    spherical_model = interp1d(radii, densities, bounds_error=False, fill_value=0)

    rho_e = spherical_model(r_array)
    rho_p = spherical_model(z_array)
    
    profile_e = []
    profile_p = []
    
    profile_e.append(rho_e)
    profile_p.append(rho_p)
    
    for i in tqdm(range(iterations), desc="Solving spining profile"):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho1(r_array, V_e, z_array, V_p, P_c, P_s, rho_c, rho_s,
                                mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core)
        profile_e.append(rho_e)
        profile_p.append(rho_p)
    
    return profile_e, profile_p

@jit(nopython=True)
def cubic_spline_kernel(rij, h):
    
    gamma = 1.825742
    H     = gamma*h
    C     = 16/np.pi
    u     = rij/H
    
    fu = np.zeros(u.shape)
    
    mask_1     = u < 1/2
    fu[mask_1] = (3*np.power(u,3) - 3*np.power(u,2) + 0.5)[mask_1]
    
    mask_2     = np.logical_and(u > 1/2, u < 1)
    fu[mask_2] = (-np.power(u,3) + 3*np.power(u,2) - 3*u + 1)[mask_2]
        
    return C*fu/np.power(H,3)
      
@jit(nopython=True)  
def N_neig_cubic_spline_kernel(eta):
        
    gamma = 1.825742
        
    return 4/3*np.pi*(gamma*eta)**3
    
@jit(nopython=True)    
def eta_cubic_spline_kernel(N_neig):
    
    gamma = 1.825742
    
    return np.cbrt(3*N_neig/4/np.pi)/gamma

@jit(nopython=True)
def SPH_density(M, R, H):
    
    rho_sph = np.zeros(H.shape[0])
    
    for i in range(H.shape[0]):
        
        rho_sph[i] = np.sum(M[i,:]*cubic_spline_kernel(R[i,:], H[i]))
        
    return rho_sph

@jit(nopython=True)
def _generate_M(indices, m):
    
    M = np.zeros(indices.shape)
    
    for i in range(M.shape[0]):
        M[i,:] = m[indices[i]]
        
    return M
 

def picle_placement_1layer(r_array, rho_e, z_array, rho_p, Tw, N,
                           mat_id_core, T_rho_id_core, T_rho_args_core,
                           ucold_array_core, N_neig=48, iterations=10):
    
    """ Particle placement for a 1 layer spining profile.
    
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
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
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
    
    c_core = weos._spec_c(mat_id_core)
    
    P = np.zeros((mP.shape[0],))
    
    for k in range(mP.shape[0]):
        u[k] = weos._ucold_tab(rho[k], mat_id_core, ucold_array_core)
        u[k] = u[k] + c_core*weos.T_rho(rho[k], T_rho_id_core, T_rho_args_core)
        P[k] = weos.P_EoS(u[k], rho[k], mat_id_core)
    
    #print("Internal energy u computed\n")
    ## Smoothing lengths, crudely estimated from the densities
    w_edge  = 2     # r/h at which the kernel goes to zero
    h       = np.cbrt(N_neig * mP / (4/3*np.pi * rho)) / w_edge 
    
    A1_id     = np.arange(mP.shape[0])
    A1_mat_id = np.ones((mP.shape[0],))*mat_id_core
    
    ############
    mP = particles.m*f 
    unique_R = np.unique(R)
    
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
        
        for _ in tqdm(range(iterations), desc="Tweeking mass of every particle"):
        
            M = _generate_M(indices, mP)
        
            rho_sph = SPH_density(M, distances, h)
            
            diff = (rho_sph - rho)/rho
            mP_next = (1 - diff)*mP
            mP_next[R == unique_R[-1]] = mP[R == unique_R[-1]] # do not change mass of boundary layers
            
            mP = mP_next
        
    else:
        
        k    = particles.N_picle // N_mem
        
        for _ in tqdm(range(iterations), desc="Tweeking mass of every particle"):
            
            mP_prev = mP.copy()
            
            for i in range(int(k)):
                
                distances_i, indices_i = nbrs.kneighbors(X[i*N_mem:(i + 1)*N_mem,:])
                
                M_i  = _generate_M(indices_i, mP_prev)
        
                rho_sph_i = SPH_density(M_i, distances_i, h[i*N_mem:(i + 1)*N_mem])
                
                diff_i = (rho_sph_i - rho[i*N_mem:(i + 1)*N_mem])/rho[i*N_mem:(i + 1)*N_mem]
                mP_next_i = (1 - diff_i)*mP[i*N_mem:(i + 1)*N_mem]
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R[-1]] = \
                        mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R[-1]] # do not change mass of boundary layers
            
                mP[i*N_mem:(i + 1)*N_mem] = mP_next_i
                
            distances_k, indices_k = nbrs.kneighbors(X[k*N_mem:,:])
                
            M_k  = _generate_M(indices_k, mP_prev)
        
            rho_sph_k = SPH_density(M_k, distances_k, h[k*N_mem:])
                
            diff_k = (rho_sph_k - rho[k*N_mem:])/rho[k*N_mem:]
            mP_next_k = (1 - diff_k)*mP[k*N_mem:]
            mP_next_k[R[k*N_mem:] == unique_R[-1]] = \
                    mP[k*N_mem:][R[k*N_mem:] == unique_R[-1]] # do not change mass of boundary layers
            
            mP[k*N_mem:] = mP_next_k    
    
# =============================================================================
#     ######
#     import matplotlib.pyplot as plt
#     
#     M = _generate_M(indices, mP) 
#     rho_sph = SPH_density(M, distances, h)
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

############################## 2 layers #######################################
    
@jit(nopython=True)
def _fillrho2(r_array, V_e, z_array, V_p, P_c, P_i, P_s, rho_c, rho_s,
             mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core,
             mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle, ucold_array_mantle):
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
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
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
        #print(i)
            
        if P_e[i + 1] >= P_s and P_e[i + 1] >= P_i:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_e[i], ucold_array_core) 
            
        elif P_e[i + 1] >= P_s and P_e[i + 1] < P_i:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                     rho_s - 10, rho_e[i], ucold_array_mantle) 
            
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s and P_p[i + 1] >= P_i:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_p[i], ucold_array_core)
            
        elif P_p[i + 1] >= P_s and P_p[i + 1] < P_i:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                     rho_s - 10, rho_p[i], ucold_array_mantle)
            
        else:
            rho_p[i + 1] = 0.
            break
        
    return rho_e, rho_p

def spin2layer(iterations, r_array, z_array, radii, densities, Tw,
               P_c, P_i, P_s, rho_c, rho_s,
               mat_id_core, T_rho_id_core, T_rho_args_core,
               mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
               ucold_array_core, ucold_array_mantle):
    """ Compute spining profile of densities for a 2 layer planet.
    
        Args:
            
            iterations (int):
                Number of iterations to run.
                
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
            
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
        Returns:
            
            profile_e ([[float]]):
                List of the iterations of the equatorial density profile (SI).
                
            profile_p ([[float]]):
                List of the iterations of the polar density profile (SI).
    
    """    
    
    spherical_model = interp1d(radii, densities, bounds_error=False, fill_value=0)

    rho_e = spherical_model(r_array)
    rho_p = spherical_model(z_array)
    
    profile_e = []
    profile_p = []
    
    profile_e.append(rho_e)
    profile_p.append(rho_p)
    
    for i in tqdm(range(iterations), desc="Solving spining profile"):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho2(r_array, V_e, z_array, V_p, P_c, P_i, P_s, rho_c, rho_s,
                                mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core,
                                mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle, ucold_array_mantle)
        profile_e.append(rho_e)
        profile_p.append(rho_p)
    
    return profile_e, profile_p

def picle_placement_2layer(r_array, rho_e, z_array, rho_p, Tw, N, rho_i,
                           mat_id_core, T_rho_id_core, T_rho_args_core,
                           mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                           ucold_array_core, ucold_array_mantle, N_neig=48,
                           iterations=10):
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
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
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
    
    c_core = weos._spec_c(mat_id_core)
    c_mantle = weos._spec_c(mat_id_mantle)
    
    P = np.zeros((mP.shape[0],))
    
    for k in range(mP.shape[0]):
        if rho[k] > rho_i:
            u[k] = weos._ucold_tab(rho[k], mat_id_core, ucold_array_core)
            u[k] = u[k] + c_core*weos.T_rho(rho[k], T_rho_id_core, T_rho_args_core)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_core)
        else:
            u[k] = weos._ucold_tab(rho[k], mat_id_mantle, ucold_array_mantle)
            u[k] = u[k] + c_mantle*weos.T_rho(rho[k], T_rho_id_mantle, T_rho_args_mantle)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_mantle)
    
    #print("Internal energy u computed\n")
    
    ## Smoothing lengths, crudely estimated from the densities
    num_ngb = N_neig    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = (rho > rho_i)*mat_id_core + (rho <= rho_i)*mat_id_mantle
    
    ############
    unique_R_core   = np.unique(R[A1_mat_id == mat_id_core])
    unique_R_mantle = np.unique(R[A1_mat_id == mat_id_mantle])
    
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
        
        for _ in tqdm(range(iterations), desc="Tweeking mass of every particle"):
        
            M = _generate_M(indices, mP)
        
            rho_sph = SPH_density(M, distances, h)
            
            diff = (rho_sph - rho)/rho
            mP_next = (1 - diff)*mP
            # do not change values of inter-boundary layers
            mP_next[R == unique_R_core[-1]]   = mP[R == unique_R_core[-1]]   # outer core layer
            mP_next[R == unique_R_mantle[0]]  = mP[R == unique_R_mantle[0]]  # inner mantle layer
            mP_next[R == unique_R_mantle[-1]] = mP[R == unique_R_mantle[-1]] # outer mantle layer
            
            mP = mP_next
        
    else:
        
        k    = particles.N_picle // N_mem
        
        for _ in tqdm(range(iterations), desc="Tweeking mass of every particle"):
            
            mP_prev = mP.copy()
            
            for i in range(int(k)):
                
                distances_i, indices_i = nbrs.kneighbors(X[i*N_mem:(i + 1)*N_mem,:])
                
                M_i  = _generate_M(indices_i, mP_prev)
        
                rho_sph_i = SPH_density(M_i, distances_i, h[i*N_mem:(i + 1)*N_mem])
                
                diff_i = (rho_sph_i - rho[i*N_mem:(i + 1)*N_mem])/rho[i*N_mem:(i + 1)*N_mem]
                mP_next_i = (1 - diff_i)*mP[i*N_mem:(i + 1)*N_mem]
                # do not change values of inter-boundary layers
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_core[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_core[-1]]   # outer core layer
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_mantle[0]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_mantle[0]]  # inner mantle layer
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_mantle[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_mantle[-1]] # outer mantle layer
            
                mP[i*N_mem:(i + 1)*N_mem] = mP_next_i
                
            distances_k, indices_k = nbrs.kneighbors(X[k*N_mem:,:])
                
            M_k  = _generate_M(indices_k, mP_prev)
        
            rho_sph_k = SPH_density(M_k, distances_k, h[k*N_mem:])
                
            diff_k = (rho_sph_k - rho[k*N_mem:])/rho[k*N_mem:]
            mP_next_k = (1 - diff_k)*mP[k*N_mem:]
            # do not change values of inter-boundary layers
            mP_next_k[R[k*N_mem:] == unique_R_core[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_core[-1]]   # outer core layer
            mP_next_k[R[k*N_mem:] == unique_R_mantle[0]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_mantle[0]]  # inner mantle layer
            mP_next_k[R[k*N_mem:] == unique_R_mantle[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_mantle[-1]] # outer mantle layer
            
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

############################## 3 layers #######################################
    
@jit(nopython=True)
def _fillrho3(r_array, V_e, z_array, V_p, P_c, P_cm, P_ma, P_s, rho_c, rho_s,
             mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core,
             mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle, ucold_array_mantle,
             mat_id_atm, T_rho_id_atm, T_rho_args_atm, ucold_array_atm):
    """ Compute densities of equatorial and polar profiles given the potential
        for a 3 layer planet.
        
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
                
            P_cm (float):
                Pressure at the boundary core-mantle of the planet (SI).
                
            P_ma (float):
                Pressure at the boundary mantle-atmosphere of the planet (SI).
                
            P_s (float):
                Pressure at the surface of the planet (SI).
                
            rho_c (float):
                Density at the center of the planet (SI).
                
            rho_s (float):
                Density at the surface of the planet (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
            mat_id_atm (int):
                Material id for the atmosphere.
                
            T_rho_id_atm (int)
                Relation between T and rho to be used at the atmosphere.
                
            T_rho_args_atm (list):
                Extra arguments to determine the relation at the atmosphere.
                
            ucold_array_atm ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the atmosphere (SI).
                
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
            
        if P_e[i + 1] >= P_s and P_e[i + 1] >= P_cm:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_e[i], ucold_array_core) 
            
        elif P_e[i + 1] >= P_s and P_e[i + 1] >= P_ma:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                     rho_s - 10, rho_e[i], ucold_array_mantle) 
            
        elif P_e[i + 1] >= P_s:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                                     rho_s - 10, rho_e[i], ucold_array_atm)
            
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s and P_p[i + 1] >= P_cm:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_p[i], ucold_array_core)
            
        elif P_p[i + 1] >= P_s and P_p[i + 1] >= P_ma:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                     rho_s - 10, rho_p[i], ucold_array_mantle)
            
        elif P_p[i + 1] >= P_s:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                                     rho_s - 10, rho_p[i], ucold_array_atm)
            
        else:
            rho_p[i + 1] = 0.
            break
        
    return rho_e, rho_p

def spin3layer(iterations, r_array, z_array, radii, densities, Tw,
               P_c, P_cm, P_ma, P_s, rho_c, rho_s,
               mat_id_core, T_rho_id_core, T_rho_args_core,
               mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
               mat_id_atm, T_rho_id_atm, T_rho_args_atm,
               ucold_array_core, ucold_array_mantle, ucold_array_atm):
    """ Compute spining profile of densities for a 3 layer planet.
    
        Args:
            
            iterations (int):
                Number of iterations to run.
                
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
                
            P_cm (float):
                Pressure at the boundary core-mantle of the planet (SI).
                
            P_ma (float):
                Pressure at the boundary mantle-atmosphere of the planet (SI).
                
            P_s (float):
                Pressure at the surface of the planet (SI).
                
            rho_c (float):
                Density at the center of the planet (SI).
                
            rho_s (float):
                Density at the surface of the planet (SI).
            
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            mat_id_atm (int):
                Material id for the atmosphere.
                
            T_rho_id_atm (int)
                Relation between T and rho to be used at the atmosphere.
                
            T_rho_args_atm (list):
                Extra arguments to determine the relation at the atmosphere.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
            ucold_array_atm ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the atmosphere (SI).
                
        Returns:
            
            profile_e ([[float]]):
                List of the iterations of the equatorial density profile (SI).
                
            profile_p ([[float]]):
                List of the iterations of the polar density profile (SI).
    
    """    
    spherical_model = interp1d(radii, densities, bounds_error=False, fill_value=0)

    rho_e = spherical_model(r_array)
    rho_p = spherical_model(z_array)
    
    profile_e = []
    profile_p = []
    
    profile_e.append(rho_e)
    profile_p.append(rho_p)
    
    for i in tqdm(range(iterations), desc="Solving spining profile"):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho3(r_array, V_e, z_array, V_p, P_c, P_cm, P_ma, P_s, rho_c, rho_s,
                                mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core,
                                mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle, ucold_array_mantle,
                                mat_id_atm, T_rho_id_atm, T_rho_args_atm, ucold_array_atm)
        profile_e.append(rho_e)
        profile_p.append(rho_p)
    
    return profile_e, profile_p


def picle_placement_3layer(r_array, rho_e, z_array, rho_p, Tw, N, rho_cm, rho_ma,
                           mat_id_core, T_rho_id_core, T_rho_args_core,
                           mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                           mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                           ucold_array_core, ucold_array_mantle, ucold_array_atm,
                           N_neig=48, iterations=10):
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
                
            rho_cm (float):
                Density at the boundary core-mantle (SI).
                
            rho_ma (float):
                Density at the boundary mantle-atmosphere (SI).
                
            mat_id_core (int):
                Material id for the core.
                
            T_rho_id_core (int)
                Relation between T and rho to be used at the core.
                
            T_rho_args_core (list):
                Extra arguments to determine the relation at the core.
                
            mat_id_mantle (int):
                Material id for the mantle.
                
            T_rho_id_mantle (int)
                Relation between T and rho to be used at the mantle.
                
            T_rho_args_mantle (list):
                Extra arguments to determine the relation at the mantle.
                
            mat_id_atm (int):
                Material id for the atmosphere.
                
            T_rho_id_atm (int)
                Relation between T and rho to be used at the atmosphere.
                
            T_rho_args_atm (list):
                Extra arguments to determine the relation at the atmosphere.
                
            ucold_array_core ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the core (SI).
                
            ucold_array_mantle ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the mantle (SI).
                
            ucold_array_atm ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() for the atmosphere (SI).
                
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
    
    c_core   = weos._spec_c(mat_id_core)
    c_mantle = weos._spec_c(mat_id_mantle)
    c_atm    = weos._spec_c(mat_id_atm)
    
    P = np.zeros((mP.shape[0],))
    
    for k in range(mP.shape[0]):
        if rho[k] > rho_cm:
            u[k] = weos._ucold_tab(rho[k], mat_id_core, ucold_array_core)
            u[k] = u[k] + c_core*weos.T_rho(rho[k], T_rho_id_core, T_rho_args_core)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_core)
            
        elif rho[k] > rho_ma:
            u[k] = weos._ucold_tab(rho[k], mat_id_mantle, ucold_array_mantle)
            u[k] = u[k] + c_mantle*weos.T_rho(rho[k], T_rho_id_mantle, T_rho_args_mantle)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_mantle)
            
        else:
            u[k] = weos._ucold_tab(rho[k], mat_id_atm, ucold_array_atm)
            u[k] = u[k] + c_atm*weos.T_rho(rho[k], T_rho_id_atm, T_rho_args_atm)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_atm)
    
    #print("Internal energy u computed\n")
    ## Smoothing lengths, crudely estimated from the densities
    num_ngb = N_neig    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = (rho > rho_cm)*mat_id_core                       \
                + np.logical_and(rho <= rho_cm, rho > rho_ma)*mat_id_mantle \
                + (rho < rho_ma)*mat_id_atm
    
    ############
    unique_R_core   = np.unique(R[A1_mat_id == mat_id_core])
    unique_R_mantle = np.unique(R[A1_mat_id == mat_id_mantle])
    unique_R_atm    = np.unique(R[A1_mat_id == mat_id_atm])
    
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
        
        for _ in tqdm(range(iterations), desc="Tweeking mass of every particle"):
        
            M = _generate_M(indices, mP)
        
            rho_sph = SPH_density(M, distances, h)
            
            diff = (rho_sph - rho)/rho
            mP_next = (1 - diff)*mP
            # do not change values of inter-boundary layers
            mP_next[R == unique_R_core[-1]]   = mP[R == unique_R_core[-1]]   # outer core layer
            mP_next[R == unique_R_mantle[0]]  = mP[R == unique_R_mantle[0]]  # inner mantle layer
            mP_next[R == unique_R_mantle[-1]] = mP[R == unique_R_mantle[-1]] # outer mantle layer
            mP_next[R == unique_R_atm[0]]  = mP[R == unique_R_atm[0]]        # inner atm layer
            mP_next[R == unique_R_atm[-1]] = mP[R == unique_R_atm[-1]]       # outer atm layer
            
            mP = mP_next
        
    else:
        
        k    = particles.N_picle // N_mem
        
        for _ in tqdm(range(iterations), desc="Tweeking mass of every particle"):
            
            mP_prev = mP.copy()
            
            for i in range(int(k)):
                
                distances_i, indices_i = nbrs.kneighbors(X[i*N_mem:(i + 1)*N_mem,:])
                
                M_i  = _generate_M(indices_i, mP_prev)
        
                rho_sph_i = SPH_density(M_i, distances_i, h[i*N_mem:(i + 1)*N_mem])
                
                diff_i = (rho_sph_i - rho[i*N_mem:(i + 1)*N_mem])/rho[i*N_mem:(i + 1)*N_mem]
                mP_next_i = (1 - diff_i)*mP[i*N_mem:(i + 1)*N_mem]
                # do not change values of inter-boundary layers
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_core[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_core[-1]]   # outer core layer
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_mantle[0]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_mantle[0]]  # inner mantle layer
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_mantle[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_mantle[-1]] # outer mantle layer
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_atm[0]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_atm[0]]  # inner atm layer
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_atm[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_atm[-1]] # outer atm layer
            
                mP[i*N_mem:(i + 1)*N_mem] = mP_next_i
                
            distances_k, indices_k = nbrs.kneighbors(X[k*N_mem:,:])
                
            M_k  = _generate_M(indices_k, mP_prev)
        
            rho_sph_k = SPH_density(M_k, distances_k, h[k*N_mem:])
                
            diff_k = (rho_sph_k - rho[k*N_mem:])/rho[k*N_mem:]
            mP_next_k = (1 - diff_k)*mP[k*N_mem:]
            # do not change values of inter-boundary layers
            mP_next_k[R[k*N_mem:] == unique_R_core[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_core[-1]]   # outer core layer
            mP_next_k[R[k*N_mem:] == unique_R_mantle[0]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_mantle[0]]  # inner mantle layer
            mP_next_k[R[k*N_mem:] == unique_R_mantle[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_mantle[-1]] # outer mantle layer
            mP_next_k[R[k*N_mem:] == unique_R_atm[0]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_atm[0]]  # inner mantle layer
            mP_next_k[R[k*N_mem:] == unique_R_atm[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_atm[-1]] # outer mantle layer
            
            mP[k*N_mem:] = mP_next_k    
    
    ######
# =============================================================================
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


###############################################################################
####################### Spining profile classes ###############################
###############################################################################
    
class _spin():
    
    N_steps_spin = 1000
    iterations   = 40
    
    P_surface   = np.nan
    T_surface   = np.nan
    rho_surface = np.nan
    
    P_center   = np.nan
    T_center   = np.nan
    rho_center = np.nan
    
    Re = np.nan
    Rp = np.nan
    Tw = np.nan
    
    A1_equator     = np.nan
    A1_pole        = np.nan
    A1_rho_equator = np.nan
    A1_rho_pole    = np.nan
    
    def __init__(self):
        return None
    
class _1l_spin(_spin):
    
    def __init__(self, planet):
        
        self.N_layers        = 1
        self.mat_id_core     = planet.mat_id_core
        self.T_rho_id_core   = planet.T_rho_id_core
        self.T_rho_args_core = planet.T_rho_args_core
        
        self.P_surface   = planet.P_surface
        self.T_surface   = planet.T_surface
        self.rho_surface = planet.rho_surface
        
        self.P_center    = planet.A1_P[-1]
        self.T_center    = planet.A1_T[-1]
        self.rho_center  = planet.A1_rho[-1]
        
        self.A1_R   = planet.A1_R
        self.A1_rho = planet.A1_rho
        self.A1_P   = planet.A1_P
        
    def solve(self, Tw, Re, Rp):
        
        r_array     = np.linspace(0, Re, self.N_steps_spin)
        z_array     = np.linspace(0, Rp, self.N_steps_spin) 
        
        ucold_array = weos.load_ucold_array(self.mat_id_core) 
        
        profile_e, profile_p = \
            spin1layer(self.iterations, r_array, z_array,
                       self.A1_R, self.A1_rho, Tw,
                       self.P_center, self.P_surface,
                       self.rho_center, self.rho_surface,
                       self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                       ucold_array)
            
        print("\nDone!")
        
        self.Tw = Tw
        self.Re = Re
        self.Rp = Rp
        
        self.A1_equator     = r_array
        self.A1_pole        = z_array
        self.A1_rho_equator = profile_e[-1]
        self.A1_rho_pole    = profile_p[-1]
        
class _2l_spin(_spin):
    
    def __init__(self, planet):
        
        self.N_layers          = 2
        self.mat_id_core       = planet.mat_id_core
        self.T_rho_id_core     = planet.T_rho_id_core
        self.T_rho_args_core   = planet.T_rho_args_core
        self.mat_id_mantle     = planet.mat_id_mantle
        self.T_rho_id_mantle   = planet.T_rho_id_mantle
        self.T_rho_args_mantle = planet.T_rho_args_mantle
        
        self.P_surface   = planet.P_surface
        self.T_surface   = planet.T_surface
        self.rho_surface = planet.rho_surface
        
        self.P_center    = planet.A1_P[-1]
        self.T_center    = planet.A1_T[-1]
        self.rho_center  = planet.A1_rho[-1]
        
        k               = np.sum(planet.A1_material == planet.mat_id_core)
        self.P_boundary = planet.A1_P[planet.N_integ_steps - k] + \
                          planet.A1_P[planet.N_integ_steps - k - 1]
        self.P_boundary = self.P_boundary/2.
        
        self.A1_R   = planet.A1_R
        self.A1_rho = planet.A1_rho
        self.A1_P   = planet.A1_P
        
    def solve(self, Tw, Re, Rp):
        
        r_array     = np.linspace(0, Re, self.N_steps_spin)
        z_array     = np.linspace(0, Rp, self.N_steps_spin)
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core) 
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle) 
        
        profile_e, profile_p = \
            spin2layer(self.iterations, r_array, z_array,
                       self.A1_R, self.A1_rho, Tw,
                       self.P_center, self.P_boundary, self.P_surface,
                       self.rho_center, self.rho_surface,
                       self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                       self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                       ucold_array_core, ucold_array_mantle)
            
        print("\nDone!")
            
        self.Tw = Tw
        self.Re = Re
        self.Rp = Rp
        
        self.A1_equator     = r_array
        self.A1_pole        = z_array
        self.A1_rho_equator = profile_e[-1]
        self.A1_rho_pole    = profile_p[-1]
        
class _3l_spin(_spin):
    
    def __init__(self, planet):
        
        self.N_layers          = 3
        self.mat_id_core       = planet.mat_id_core
        self.T_rho_id_core     = planet.T_rho_id_core
        self.T_rho_args_core   = planet.T_rho_args_core
        self.mat_id_mantle     = planet.mat_id_mantle
        self.T_rho_id_mantle   = planet.T_rho_id_mantle
        self.T_rho_args_mantle = planet.T_rho_args_mantle
        self.mat_id_atm        = planet.mat_id_atm
        self.T_rho_id_atm      = planet.T_rho_id_atm
        self.T_rho_args_atm    = planet.T_rho_args_atm
        
        self.P_surface   = planet.P_surface
        self.T_surface   = planet.T_surface
        self.rho_surface = planet.rho_surface
        
        k                  = np.sum(planet.A1_material == planet.mat_id_core)
        self.P_boundary_cm = planet.A1_P[planet.N_integ_steps - k] + \
                             planet.A1_P[planet.N_integ_steps - k - 1]
        self.P_boundary_cm = self.P_boundary_cm/2.
        
        k                  = np.sum(planet.A1_material == planet.mat_id_atm)
        self.P_boundary_ma = (planet.A1_P[k] + planet.A1_P[k - 1])/2.
        
        self.P_center    = planet.A1_P[-1]
        self.T_center    = planet.A1_T[-1]
        self.rho_center  = planet.A1_rho[-1]
        
        self.A1_R   = planet.A1_R
        self.A1_rho = planet.A1_rho
        self.A1_P   = planet.A1_P
        
    def solve(self, Tw, Re, Rp):
        
        r_array     = np.linspace(0, Re, self.N_steps_spin)
        z_array     = np.linspace(0, Rp, self.N_steps_spin)
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core) 
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        ucold_array_atm = weos.load_ucold_array(self.mat_id_atm) 
        
        profile_e, profile_p = \
            spin3layer(self.iterations, r_array, z_array,
                       self.A1_R, self.A1_rho, Tw,
                       self.P_center, self.P_boundary_cm, self.P_boundary_ma, self.P_surface,
                       self.rho_center, self.rho_surface,
                       self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                       self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                       self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                       ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        print("\nDone!")
            
        self.Tw = Tw
        self.Re = Re
        self.Rp = Rp
        
        self.A1_equator     = r_array
        self.A1_pole        = z_array
        self.A1_rho_equator = profile_e[-1]
        self.A1_rho_pole    = profile_p[-1]
    
def Spin(planet):
        
    _print_banner()
    
    if planet.N_layers not in [1, 2, 3]:
        
        print(f"Can't build a planet with {planet.N_layers} layers!")
        return None
        
    if planet.N_layers == 1:
        
        spin_planet = _1l_spin(planet)
        return spin_planet
            
    elif planet.N_layers == 2:
        
        spin_planet = _2l_spin(planet)
        return spin_planet
        
    elif planet.N_layers == 3:
        
        spin_planet = _3l_spin(planet)
        return spin_planet

class _1l_genspheroid():
    
    def __init__(self, spin_planet, N_particles, N_neig=48, iterations=10):
        
        self.N_layers    = 1
        self.iterations  = iterations
        
        self.A1_equator     = spin_planet.A1_equator
        self.A1_rho_equator = spin_planet.A1_rho_equator
        self.A1_pole        = spin_planet.A1_pole
        self.A1_rho_pole    = spin_planet.A1_rho_pole
        self.Tw             = spin_planet.Tw
        
        self.mat_id_core     = spin_planet.mat_id_core
        self.T_rho_id_core   = spin_planet.T_rho_id_core
        self.T_rho_args_core = spin_planet.T_rho_args_core
        
        ucold_array_core = weos.load_ucold_array(self.mat_id_core)
        
        x, y, z, vx, vy, vz, m, h, rho, P, u, mat_id, picle_id = \
            picle_placement_1layer(self.A1_equator, self.A1_rho_equator,
                                   self.A1_pole, self.A1_rho_pole, self.Tw, N_particles,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   ucold_array_core, N_neig, iterations)
            
        self.A1_x   = x
        self.A1_y   = y
        self.A1_z   = z
        self.A1_vx  = vx
        self.A1_vy  = vy
        self.A1_vz  = vz
        self.A1_m   = m
        self.A1_h   = h
        self.A1_rho = rho
        self.A1_P   = P
        self.A1_u   = u
        self.A1_mat_id   = mat_id
        self.A1_picle_id = picle_id
        self.N_particles = x.shape[0]
        
class _2l_genspheroid():
    
    def __init__(self, spin_planet, N_particles, N_neig=48, iterations=10):
        
        self.N_layers    = 2
        
        
        self.A1_equator     = spin_planet.A1_equator
        self.A1_rho_equator = spin_planet.A1_rho_equator
        self.A1_pole        = spin_planet.A1_pole
        self.A1_rho_pole    = spin_planet.A1_rho_pole
        self.Tw             = spin_planet.Tw
        
        self.mat_id_core       = spin_planet.mat_id_core
        self.T_rho_id_core     = spin_planet.T_rho_id_core
        self.T_rho_args_core   = spin_planet.T_rho_args_core
        self.mat_id_mantle     = spin_planet.mat_id_mantle
        self.T_rho_id_mantle   = spin_planet.T_rho_id_mantle
        self.T_rho_args_mantle = spin_planet.T_rho_args_mantle
        
        self.P_boundary   = spin_planet.P_boundary
        rho_P_model       = interp1d(spin_planet.A1_P, spin_planet.A1_rho)
        self.rho_boundary = rho_P_model(self.P_boundary)
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        
        x, y, z, vx, vy, vz, m, h, rho, P, u, mat_id, picle_id = \
            picle_placement_2layer(self.A1_equator, self.A1_rho_equator,
                                   self.A1_pole, self.A1_rho_pole,
                                   self.Tw, N_particles, self.rho_boundary,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   ucold_array_core, ucold_array_mantle,
                                   N_neig, iterations)
            
        self.A1_x   = x
        self.A1_y   = y
        self.A1_z   = z
        self.A1_vx  = vx
        self.A1_vy  = vy
        self.A1_vz  = vz
        self.A1_m   = m
        self.A1_h   = h
        self.A1_rho = rho
        self.A1_P   = P
        self.A1_u   = u
        self.A1_mat_id   = mat_id
        self.A1_picle_id = picle_id
        self.N_particles = x.shape[0]
    
class _3l_genspheroid():
    
    def __init__(self, spin_planet, N_particles, N_neig=48, iterations=10):
        
        self.N_layers    = 3
        
        self.A1_equator     = spin_planet.A1_equator
        self.A1_rho_equator = spin_planet.A1_rho_equator
        self.A1_pole        = spin_planet.A1_pole
        self.A1_rho_pole    = spin_planet.A1_rho_pole
        self.Tw             = spin_planet.Tw
        
        self.mat_id_core       = spin_planet.mat_id_core
        self.T_rho_id_core     = spin_planet.T_rho_id_core
        self.T_rho_args_core   = spin_planet.T_rho_args_core
        self.mat_id_mantle     = spin_planet.mat_id_mantle
        self.T_rho_id_mantle   = spin_planet.T_rho_id_mantle
        self.T_rho_args_mantle = spin_planet.T_rho_args_mantle
        self.mat_id_atm        = spin_planet.mat_id_atm
        self.T_rho_id_atm      = spin_planet.T_rho_id_atm
        self.T_rho_args_atm    = spin_planet.T_rho_args_atm
        
        self.P_boundary_cm    = spin_planet.P_boundary_cm
        self.P_boundary_ma    = spin_planet.P_boundary_ma
        rho_P_model           = interp1d(spin_planet.A1_P, spin_planet.A1_rho)
        self.rho_boundary_cm  = rho_P_model(self.P_boundary_cm)
        self.rho_boundary_ma  = rho_P_model(self.P_boundary_ma)
        
        ucold_array_core   = weos.load_ucold_array(self.mat_id_core)
        ucold_array_mantle = weos.load_ucold_array(self.mat_id_mantle)
        ucold_array_atm    = eos.load_ucold_array(self.mat_id_atm)
        
        x, y, z, vx, vy, vz, m, h, rho, P, u, mat_id, picle_id = \
            picle_placement_3layer(self.A1_equator, self.A1_rho_equator,
                                   self.A1_pole, self.A1_rho_pole,
                                   self.Tw, N_particles, self.rho_boundary_cm, self.rho_boundary_ma,
                                   self.mat_id_core, self.T_rho_id_core, self.T_rho_args_core,
                                   self.mat_id_mantle, self.T_rho_id_mantle, self.T_rho_args_mantle,
                                   self.mat_id_atm, self.T_rho_id_atm, self.T_rho_args_atm,
                                   ucold_array_core, ucold_array_mantle, ucold_array_atm,
                                   N_neig, iterations)
            
        self.A1_x   = x
        self.A1_y   = y
        self.A1_z   = z
        self.A1_vx  = vx
        self.A1_vy  = vy
        self.A1_vz  = vz
        self.A1_m   = m
        self.A1_h   = h
        self.A1_rho = rho
        self.A1_P   = P
        self.A1_u   = u
        self.A1_mat_id   = mat_id
        self.A1_picle_id = picle_id
        self.N_particles = x.shape[0]

def GenSpheroid(spin_planet, N_particles, N_neig=48, iterations=10):
    
    _print_banner()
    
    sp = spin_planet
    
    if sp.N_layers not in [1, 2, 3]:
        
        print(f"Can't build a planet with {sp.N_layers} layers!")
        return None
        
    if sp.N_layers == 1:
        
        spheroid = _1l_genspheroid(sp, N_particles, N_neig, iterations)
        return spheroid
            
    elif sp.N_layers == 2:
        
        spheroid = _2l_genspheroid(sp, N_particles, N_neig, iterations)
        return spheroid
        
    elif sp.N_layers == 3:
        
        spheroid = _3l_genspheroid(sp, N_particles, N_neig, iterations)
        return spheroid
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        