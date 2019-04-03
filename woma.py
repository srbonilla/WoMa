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
import eos

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
    
    ucold_array_100 = eos._create_ucold_array(100)
    ucold_array_101 = eos._create_ucold_array(101)
    ucold_array_102 = eos._create_ucold_array(102)
    
    np.save("data/ucold_array_100", ucold_array_100)
    np.save("data/ucold_array_101", ucold_array_101)
    np.save("data/ucold_array_102", ucold_array_102)
    
def load_ucold_array(mat_id):
    """ Load precomputed values of cold internal energy for a given material.
    
        Returns:
            ucold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() (SI).
    """
    
    if mat_id == 100:
        ucold_array = np.load('data/ucold_array_100.npy')
    elif mat_id == 101:
        ucold_array = np.load('data/ucold_array_101.npy')
    elif mat_id == 102:
        ucold_array = np.load('data/ucold_array_102.npy')
        
    return ucold_array


   
############################## 1 layer ########################################
    
@jit(nopython=True)
def integrate_1layer(N, R, M, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                     rhos_min, rhos_max, ucold_array):
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
                
            mat_id (int):
                Material id.
                
            T_rho_id (int)
                Relation between T and rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
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
        
    rhos = eos._find_rho_fixed_T(Ps, mat_id, Ts, rhos_min, rhos_max, ucold_array)
    
    c = eos._spec_c(mat_id)
    
    if rhos == rhos_min or rhos == rhos_max or rhos == (rhos_min + rhos_max)/2.:
        print("Could not find rho surface in that interval")
        return r, m, P, T, rho, u, mat
    
    else:
        us = eos.ucold(rhos, mat_id, 10000) + c*Ts
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
        rho[i] = eos._find_rho(P[i], mat_id, T_rho_id, T_rho_args,
                               rho[i - 1], 1.1*rho[i - 1], ucold_array)
        T[i]   = eos.T_rho(rho[i], T_rho_id, T_rho_args)
        u[i]   = eos._ucold_tab(rho[i], ucold_array) + c*T[i]
        
        if m[i] < 0:
            
            return r, m, P, T, rho, u, mat
        
    return r, m, P, T, rho, u, mat

@jit(nopython=True)
def find_mass_1layer(N, R, M_max, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                     rhos_min, rhos_max, ucold_array):
    
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
                
            mat_id (int):
                Material id.
                
            T_rho_id (int)
                Relation between T and rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
            ucold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_ucold_array() (SI).
                
        Returns:
            
            M_max (float):
                Mass of the planet (SI).
            
    """
    
    M_min = 0.
    
    r, m, P, T, rho, u, mat = integrate_1layer(N, R, M_max, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                                               rhos_min, rhos_max, ucold_array)
    
    if m[-1] > 0.:
        
        for i in range(30):
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_1layer(N, R, M_try, Ps, Ts, mat_id,
                                   T_rho_id, T_rho_args,
                                   rhos_min, rhos_max, ucold_array)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max
     
############################## 2 layers #######################################
    
@jit(nopython=True)
def integrate_2layer(N, R, M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
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
                
            b_cm (float):
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
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
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
        
    rhos = eos._find_rho_fixed_T(Ps, mat_id_mantle, Ts,
                                 rhos_min, rhos_max, ucold_array_mantle)
    
    c_core   = eos._spec_c(mat_id_core)
    c_mantle = eos._spec_c(mat_id_mantle)
    
    if rhos == rhos_min or rhos == rhos_max or rhos == (rhos_min + rhos_max)/2.:
        print("Could not find rho surface in that interval")
        return r, m, P, T, rho, u, mat
    
    else:
        us = eos.ucold(rhos, mat_id_mantle, 10000) + c_mantle*Ts
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
        if r[i] > b_cm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_mantle)
            T[i]   = eos.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_mantle) + c_mantle*T[i]
            mat[i] = mat_id_mantle
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
        # boundary core mantle
        elif r[i] <= b_cm and r[i - 1] > b_cm:
            
            rho_transition = eos._find_rho_fixed_T(P[i - 1], mat_id_core, T[i - 1],
                                                   rho[i - 1], 5*rho[i - 1], ucold_array_core)
            
            if T_rho_id_core == 1:
                T_rho_args_core[0] = T[i - 1]*rho_transition**(-T_rho_args_core[1])
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho_transition*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho_transition/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho_transition, ucold_array_core)
            T[i]   = eos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
        # core  
        elif r[i] <= b_cm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_core)
            T[i]   = eos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
    return r, m, P, T, rho, u, mat

@jit(nopython=True)
def find_mass_2layer(N, R, M_max, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
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
                
            b_cm (float):
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
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
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
        integrate_2layer(N, R, M_max, Ps, Ts, b_cm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle)
    
    if m[-1] > 0.:
        
        for i in range(30):
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_2layer(N, R, M_try, Ps, Ts, b_cm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

@jit(nopython=True)
def find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle):
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
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
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
    
    b_min = 0.
    b_max = R 
    
    r, m, P, T, rho, u, mat = \
        integrate_2layer(N, R, M, Ps, Ts, b_max,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle)
    
    if m[-1] == 0.:
        
        for i in range(40):
            
            b_try = (b_min + b_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_2layer(N, R, M, Ps, Ts, b_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle)
            
            if m[-1] > 0.:
                b_min = b_try
            else:
                b_max = b_try
                
    else:
        print("R is too low, ran out of mass in first iteration")
        return 0.
        
    return b_min

############################## 3 layers #######################################
    
@jit(nopython=True)
def integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
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
                
            b_cm (float):
                Boundary core-mantle (SI).
                
            b_ma (float):
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
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
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
        
    rhos = eos._find_rho_fixed_T(Ps, mat_id_atm, Ts,
                                 rhos_min, rhos_max, ucold_array_atm)
    
    c_core   = eos._spec_c(mat_id_core)
    c_mantle = eos._spec_c(mat_id_mantle)
    c_atm    = eos._spec_c(mat_id_atm)
    
    if rhos == rhos_min or rhos == rhos_max or rhos == (rhos_min + rhos_max)/2.:
        print("Could not find rho surface in that interval")
        return r, m, P, T, rho, u, mat
    
    else:
        us = eos.ucold(rhos, mat_id_atm, 10000) + c_atm*Ts
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
        if r[i] > b_ma:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_atm)
            T[i]   = eos.T_rho(rho[i], T_rho_id_atm, T_rho_args_atm)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_atm) + c_atm*T[i]
            mat[i] = mat_id_atm
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
        # boundary mantle atm
        elif r[i] <= b_ma and r[i - 1] > b_ma:
            
            rho_transition = eos._find_rho_fixed_T(P[i - 1], mat_id_mantle, T[i - 1],
                                                   rho[i - 1], 5*rho[i - 1], ucold_array_mantle)
            
            if T_rho_id_mantle == 1:
                T_rho_args_mantle[0] = T[i - 1]*rho_transition**(-T_rho_args_mantle[1])
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho_transition*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho_transition/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                   rho[i - 1], 1.1*rho_transition, ucold_array_mantle)
            T[i]   = eos.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_mantle) + c_mantle*T[i]
            mat[i] = mat_id_mantle
            
        # mantle
        elif r[i] > b_cm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_mantle)
            T[i]   = eos.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_mantle) + c_mantle*T[i]
            mat[i] = mat_id_mantle
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
        # boundary core mantle
        elif r[i] <= b_cm and r[i - 1] > b_cm:
            
            rho_transition = eos._find_rho_fixed_T(P[i - 1], mat_id_core, T[i - 1],
                                                   rho[i - 1], 5*rho[i - 1], ucold_array_core)
            
            if T_rho_id_core == 1:
                T_rho_args_core[0] = T[i - 1]*rho_transition**(-T_rho_args_core[1])
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho_transition*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho_transition/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho_transition, ucold_array_core)
            T[i]   = eos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
        # core  
        elif r[i] <= b_cm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_core)
            T[i]   = eos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
    return r, m, P, T, rho, u, mat


@jit(nopython=True)
def find_mass_3layer(N, R, M_max, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
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
                
            b_cm (float):
                Boundary core-mantle (SI).
                
            b_ma (float):
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
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
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
        integrate_3layer(N, R, M_max, Ps, Ts, b_cm, b_ma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
    
    if m[-1] > 0.:
        
        for i in range(40):
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M_try, Ps, Ts, b_cm, b_ma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

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

@jit(nopython=True)
def find_b_ma_3layer(N, R, M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
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
                
            b_cm (float):
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
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
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
    
    b_cm_min = find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)
    
    b_cm_max = find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_atm)
    
    if b_cm > b_cm_max:
        print("value of b_cm is too high,")
        print("maximum value available for this configuration is:")
        print(b_cm_max/R_earth, "R_earth")
        return -1
        
    elif b_cm < b_cm_min:
        print("value of b_cm is too low,")
        print("minimum value available for this configuration is:")
        print(b_cm_min/R_earth, "R_earth")
        return -1
        
    b_ma_min = R
    b_ma_max = b_cm
    
    r, m_min, P, T, rho, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_min,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
    r, m_max, P, T, rho, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_max,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
    
    if m_max[-1] > 0. and m_min[-1] == 0:
        
        for i in range(30):
            
            b_ma_try = (b_ma_min + b_ma_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if m[-1] > 0.:
                b_ma_max = b_ma_try
            else:
                b_ma_min = b_ma_try
                
    else:
        print("Something went wrong")
        return 0.
        
    return b_ma_max


@jit(nopython=True)
def _find_b_ma_3layer_nocheck(N, R, M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
    """ Fast finder of the boundary mantle-atmosphere of the planet for
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
                
            b_cm (float):
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
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
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
    
    b_ma_min = R
    b_ma_max = b_cm
    
    r, m_min, P, T, rho, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_min,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
    r, m_max, P, T, rho, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_max,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
    

        
    for i in range(30):
            
        b_ma_try = (b_ma_min + b_ma_max)/2.
            
        r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        if m[-1] > 0.:
            b_ma_max = b_ma_try
        else:
            b_ma_min = b_ma_try
                

    return b_ma_max



@jit(nopython=True)
def find_boundaries_3layer(N, R, M, Ps, Ts, MoI,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
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
                
            rhos_min (float):
                Lower bound for the density at the surface.
                
            rhos_min (float):
                Upper bound for the density at the surface.
                
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
    
    b_cm_max = find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)
    
    b_cm_min = find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_atm)
        
    b_ma_max = R
    b_ma_min = b_cm_min
    
    r_min, m, P, T, rho_min, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm_min, b_ma_min,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
    r_max, m, P, T, rho_max, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm_max, b_ma_max,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
    moi_min = moi(r_min, rho_min)
    moi_max = moi(r_max, rho_max)
    
    if MoI > moi_min and  MoI < moi_max:
        
        for i in range(10):
            
            b_cm_try = (b_cm_min + b_cm_max)/2.
            
            b_ma_try = _find_b_ma_3layer_nocheck(N, R, M, Ps, Ts, b_cm_try,
                             mat_id_core, T_rho_id_core, T_rho_args_core,
                             mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                             mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                             rhos_min, rhos_max,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm)
                    
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M, Ps, Ts, b_cm_try, b_ma_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if moi(r,rho) < MoI:
                b_cm_min = b_cm_try
            else:
                b_cm_max = b_cm_try
                
    elif MoI > moi_max:
        print("Moment of interia is too high,")
        print("maximum value is:")
        print(moi_max/M_earth/R_earth/R_earth,"[M_earth R_earth^2]")
        b_cm_try = 0.
        b_ma_try = 0.
    
    elif MoI < moi_min:
        print("Moment of interia is too low,")
        print("minimum value is:")
        print(moi_min/M_earth/R_earth/R_earth,"[M_earth R_earth^2]")
        b_cm_try = 0.
        b_ma_try = 0.
        
    else:
        print("Something went wrong")
        b_cm_try = 0.
        b_ma_try = 0.
        
    return b_cm_try, b_ma_try

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
            rho_e[i + 1] = eos._find_rho(P_e[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                         rho_s - 10, rho_e[i], ucold_array_core) 
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s:
            rho_p[i + 1] = eos._find_rho(P_p[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
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
    
    for i in range(iterations):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho1(r_array, V_e, z_array, V_p, P_c, P_s, rho_c, rho_s,
                                mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core)
        profile_e.append(rho_e)
        profile_p.append(rho_p)
    
    return profile_e, profile_p

def picle_placement_1layer(r_array, rho_e, z_array, rho_p, Tw, N,
                           mat_id_core, T_rho_id_core, T_rho_args_core,
                           ucold_array_core, N_neig):
    
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
    
    #ucold_array_core = spipgen_v2._create_ucold_array(mat_id_core)
    c_core = eos._spec_c(mat_id_core)
    
    P = np.zeros((mP.shape[0],))
    
    for k in range(mP.shape[0]):
        u[k] = eos._ucold_tab(rho[k], ucold_array_core)
        u[k] = u[k] + c_core*eos.T_rho(rho[k], T_rho_id_core, T_rho_args_core)
        P[k] = eos.P_EoS(u[k], rho[k], mat_id_core)
    
    print("Internal energy u computed\n")
    ## Smoothing lengths, crudely estimated from the densities
    num_ngb = N_neig    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_id     = np.arange(mP.shape[0])
    A1_mat_id = np.ones((mP.shape[0],))*mat_id_core
    
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
            rho_e[i + 1] = eos._find_rho(P_e[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_e[i], ucold_array_core) 
            
        elif P_e[i + 1] >= P_s and P_e[i + 1] < P_i:
            rho_e[i + 1] = eos._find_rho(P_e[i + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                     rho_s - 10, rho_e[i], ucold_array_mantle) 
            
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s and P_p[i + 1] >= P_i:
            rho_p[i + 1] = eos._find_rho(P_p[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_p[i], ucold_array_core)
            
        elif P_p[i + 1] >= P_s and P_p[i + 1] < P_i:
            rho_p[i + 1] = eos._find_rho(P_p[i + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
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
    
    for i in range(iterations):
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
                           ucold_array_core, ucold_array_mantle, N_neig=48):
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
    
    c_core = eos._spec_c(mat_id_core)
    c_mantle = eos._spec_c(mat_id_mantle)
    
    P = np.zeros((mP.shape[0],))
    
    for k in range(mP.shape[0]):
        if particles_rho[k] > rho_i:
            u[k] = eos._ucold_tab(rho[k], ucold_array_core)
            u[k] = u[k] + c_core*eos.T_rho(rho[k], T_rho_id_core, T_rho_args_core)
            P[k] = eos.P_EoS(u[k], rho[k], mat_id_core)
        else:
            u[k] = eos._ucold_tab(rho[k], ucold_array_mantle)
            u[k] = u[k] + c_mantle*eos.T_rho(rho[k], T_rho_id_mantle, T_rho_args_mantle)
            P[k] = eos.P_EoS(u[k], rho[k], mat_id_mantle)
    
    print("Internal energy u computed\n")
    
    ## Smoothing lengths, crudely estimated from the densities
    num_ngb = N_neig    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = (rho > rho_i)*mat_id_core + (rho <= rho_i)*mat_id_mantle
    
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
            rho_e[i + 1] = eos._find_rho(P_e[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_e[i], ucold_array_core) 
            
        elif P_e[i + 1] >= P_s and P_e[i + 1] >= P_ma:
            rho_e[i + 1] = eos._find_rho(P_e[i + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                     rho_s - 10, rho_e[i], ucold_array_mantle) 
            
        elif P_e[i + 1] >= P_s:
            rho_e[i + 1] = eos._find_rho(P_e[i + 1], mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                                     rho_s - 10, rho_e[i], ucold_array_atm)
            
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s and P_p[i + 1] >= P_cm:
            rho_p[i + 1] = eos._find_rho(P_p[i + 1], mat_id_core, T_rho_id_core, T_rho_args_core,
                                     rho_s - 10, rho_p[i], ucold_array_core)
            
        elif P_p[i + 1] >= P_s and P_p[i + 1] >= P_ma:
            rho_p[i + 1] = eos._find_rho(P_p[i + 1], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                     rho_s - 10, rho_p[i], ucold_array_mantle)
            
        elif P_p[i + 1] >= P_s:
            rho_p[i + 1] = eos._find_rho(P_p[i + 1], mat_id_atm, T_rho_id_atm, T_rho_args_atm,
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
    
    for i in range(iterations):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho3(r_array, V_e, z_array, V_p, P_c, P_cm, P_ma, P_s, rho_c, rho_s,
                                mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core,
                                mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle, ucold_array_mantle,
                                mat_id_atm, T_rho_id_atm, T_rho_args_atm, ucold_array_atm)
        profile_e.append(rho_e)
        profile_p.append(rho_p)
    
    return profile_e, profile_p


def _picle_placement_3layer(r_array, rho_e, z_array, rho_p, Tw, N, rho_cm, rho_ma,
                            mat_id_core, T_rho_id_core, T_rho_args_core,
                            mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                            mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                            ucold_array_core, ucold_array_mantle, ucold_array_atm,
                            N_neig=48):
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
    
    c_core    = eos._spec_c(mat_id_core)
    c_mantle  = eos._spec_c(mat_id_mantle)
    c_atm     = eos._spec_c(mat_id_atm)
    
    P = np.zeros((mP.shape[0],))
    
    for k in range(mP.shape[0]):
        if particles_rho[k] > rho_cm:
            u[k] = eos._ucold_tab(rho[k], ucold_array_core)
            u[k] = u[k] + c_core*eos.T_rho(rho[k], T_rho_id_core, T_rho_args_core)
            P[k] = eos.P_EoS(u[k], rho[k], mat_id_core)
            
        elif particles_rho[k] > rho_ma:
            u[k] = eos._ucold_tab(rho[k], ucold_array_mantle)
            u[k] = u[k] + c_mantle*eos.T_rho(rho[k], T_rho_id_mantle, T_rho_args_mantle)
            P[k] = eos.P_EoS(u[k], rho[k], mat_id_mantle)
            
        else:
            u[k] = eos._ucold_tab(rho[k], ucold_array_atm)
            u[k] = u[k] + c_atm*eos.T_rho(rho[k], T_rho_id_atm, T_rho_args_atm)
            P[k] = eos.P_EoS(u[k], rho[k], mat_id_atm)
    
    print("Internal energy u computed\n")
    ## Smoothing lengths, crudely estimated from the densities
    num_ngb = N_neig    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = (rho > rho_cm)*mat_id_core                       \
                + np.logical_and(rho <= rho_cm, rho > rho_ma)*mat_id_mantle \
                + (rho < rho_ma)*mat_id_atm
    
    return x, y, zP, vx, vy, vz, mP, h, rho, P, u, A1_mat_id, A1_id 