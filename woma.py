#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:18:01 2019

@author: Sergio Ruiz-Bonilla
"""

# ============================================================================ #
# ===================== Libraries and constants ============================== #
# ============================================================================ #

import numpy as np
from numba import jit
from scipy.interpolate import interp1d
import seagen
import weos
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os
import sys
import h5py

# Go to the WoMa directory
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
sys.path.append(dir)

# Global constants
G = 6.67408E-11
R_earth = 6371000
M_earth = 5.972E24

def _print_banner():
    print("\n")
    print("#  WoMa - World Maker")
    print("#  sergio.ruiz-bonilla@durham.ac.uk")
    print("\n")
    
# Misc utilities (move to separate file...)
def check_end(string, end):
    """ Check that a string ends with the required characters and append them
        if not.
    """
    if string[-len(end):] != end:
        string  += end

    return string
    
def format_array_string(A1_x, format):
    """ Return a print-ready string of a 1D array's contents in a given format.

        Args:
            A1_x ([])
                An array of values that can be printed with the given format.

            format (str)
                A printing format, e.g. "%d", "%.5g".

        Returns:
            string (str)
                The formatted string.
    """
    string  = ""

    # Append each element
    for x in A1_x:
        string += "%s, " % (format % x)

    # Add brackets and remove the trailing comma
    return "[%s]" % string[:-2]
    
def add_whitespace(string, space):
    """ Return a string for aligned printing with adjusted spaces to account for
        the length of the input.

        e.g.
            >>> asdf = 123
            >>> qwerty = 456
            >>> print("%s = %d \n""%s = %d" %
                      (add_whitespace("asdf", 12), asdf,
                       add_whitespace("qwerty", 12), qwerty))
            asdf         = 123
            qwerty       = 456
    """
    return "%s" % string + " " * (space - len("%s" % string))
    
@jit(nopython=True)
def moi(A1_r, A1_rho):
    """ Computes moment of inertia for a planet with spherical symmetry.
        
        Args:
            A1_r ([float]):
                Radii of the planet (SI).
                
            A1_rho ([float]):
                Densities asociated with the radii (SI)
                
        Returns:
            MoI (float):
                Moment of inertia (SI).
    """    
    dr  = np.abs(A1_r[0] - A1_r[1])
    r4  = np.power(A1_r, 4)
    MoI = 2*np.pi*(4/3)*np.sum(r4*A1_rho)*dr
    
    return MoI

# Output
Di_hdf5_planet_label  = {
    "num_layer"     : "Number of Layers",
    "mat_layer"     : "Layer Materials",
    "mat_id_layer"  : "Layer Material IDs",
    "T_rho_type"    : "Layer T-rho Type",
    "T_rho_args"    : "Layer T-rho Args",
    "R_layer"       : "Layer Boundary Radii",
    "M_layer"       : "Mass in each Layer",
    "M"             : "Total Mass",
    "R"             : "Total Radius",
    "idx_layer"     : "Outer Index of each Layer",
    "P_s"           : "Surface Pressure",
    "T_s"           : "Surface Temperature",
    "rho_s"         : "Surface Density",
    "r"             : "Profile Radii",
    "m_enc"         : "Profile Enclosed Masses",
    "rho"           : "Profile Densities",
    "T"             : "Profile Temperatures",
    "u"             : "Profile Specific Internal Energies",
    "P"             : "Profile Pressures",
    "mat_id"        : "Profile Material IDs",
    }

def get_planet_data(f, param):
    """ Load a planet attribute or array.

        Args:
            f (h5py File)
                The opened hdf5 data file (with "r").

            param (str)
                The array or attribute to get. See Di_hdf5_planet_label for
                details.

        Returns:
            data (?)
                The array or attribute (std units).
    """
    # Attributes
    try:
        return f["planet"].attrs[Di_hdf5_planet_label[param]]
    # Datasets
    except KeyError:
        return f["planet/" + Di_hdf5_planet_label[param]][()]

def multi_get_planet_data(f, A1_param):
    """ Load multiple planet attributes or arrays.

        Args:
            f (h5py File)
                The opened hdf5 data file (with "r").

            A1_param ([str])
                List of the arrays or attributes to get. See Di_hdf5_planet_label
                for details.

        Returns:
            A1_data ([?])
                The list of the arrays or attributes (std units).
    """
    A1_data = []
    # Load each requested array
    for param in A1_param:
        A1_data.append(get_planet_data(f, param))

    return A1_data
    
# ============================================================================ #
# ===================== Spherical profile functions ========================== #
# ============================================================================ #

def set_up():
    """ Create tabulated values of cold internal energy if they don't exist, 
        and save the results in the data/ folder.    
    """
    # Make the directory if it doesn't already exist
    if not os.path.isdir("data"):
        os.mkdir("data")
    
    # Make the files if they don't already exist    
    if not os.path.isfile(weos.Fp_u_cold_Til_iron):
        print('Creating u cold curve for material Til_iron... ', end='')
        sys.stdout.flush()
        u_cold_array = weos._create_u_cold_array(weos.id_Til_iron)
        np.save(weos.Fp_u_cold_Til_iron, u_cold_array)
        del u_cold_array
        print("Done")
    
    if not os.path.isfile(weos.Fp_u_cold_Til_granite):
        print('Creating u cold curve for material Til_granite... ', end='')
        sys.stdout.flush()
        u_cold_array = weos._create_u_cold_array(weos.id_Til_granite)
        np.save(weos.Fp_u_cold_Til_granite, u_cold_array)
        del u_cold_array
        print("Done")
    
    if not os.path.isfile(weos.Fp_u_cold_Til_water):
        print('Creating u cold curve for material Til_water... ', end='')
        sys.stdout.flush()
        u_cold_array = weos._create_u_cold_array(weos.id_Til_water)
        np.save(weos.Fp_u_cold_Til_water, u_cold_array)
        del u_cold_array
        print("Done")
      
# ===================== 1 layer ============================================== #

@jit(nopython=True)
def L1_integrate(num_prof, R, M, P_s, T_s, rho_s, mat_id, T_rho_type, 
                 T_rho_args, u_cold_array):
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
                
            T_rho_type (int)
                Relation between A1_T and A1_rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
            u_cold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() (SI).
                
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
    
    u_s = weos._find_u(rho_s, mat_id, T_s, u_cold_array)
    if T_rho_type == 1:
        T_rho_args[0] = T_s*rho_s**(-T_rho_args[1])
    
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
        A1_P[i]     = A1_P[i - 1] + G*A1_m_enc[i - 1]*A1_rho[i - 1]/(A1_r[i - 1]**2)*dr
        A1_rho[i]   = weos._find_rho(A1_P[i], mat_id, T_rho_type, T_rho_args,
                                     A1_rho[i - 1], 1.1*A1_rho[i - 1], u_cold_array)
        A1_T[i]     = weos.T_rho(A1_rho[i], T_rho_type, T_rho_args)
        A1_u[i]     = weos._find_u(A1_rho[i], mat_id, A1_T[i], u_cold_array)
        
        if A1_m_enc[i] < 0:
            return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id
        
    return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id

@jit(nopython=True)
def L1_find_mass(num_prof, R, M_max, P_s, T_s, rho_s, mat_id, T_rho_type, 
                 T_rho_args, u_cold_array):
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
                
            T_rho_type (int)
                Relation between A1_T and A1_rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
            u_cold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() (SI).
                
        Returns:
            M_max (float):
                Mass of the planet (SI).            
    """    
    M_min = 0.
    
    # Try integrating the profile with the maximum mass
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
        num_prof, R, M_max, P_s, T_s, rho_s, mat_id, T_rho_type, T_rho_args, 
        u_cold_array
        )
        
    if A1_m_enc[-1] < 0: 
        print("M_max is too low, ran out of mass in first iteration")
    
    # Iterate the mass     
    while np.abs(M_min - M_max) > 1e-8*M_min:
        M_try = (M_min + M_max) * 0.5
        
        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
            num_prof, R, M_try, P_s, T_s, rho_s, mat_id,
            T_rho_type, T_rho_args, u_cold_array)
        
        if A1_m_enc[-1] > 0.:
            M_max = M_try
        else:
            M_min = M_try
    
    return M_max

#@jit(nopython=True)
def L1_find_radius(num_prof, R_max, M, P_s, T_s, rho_s, mat_id, T_rho_type, 
                   T_rho_args, u_cold_array, num_attempt=40):
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
                
            T_rho_type (int)
                Relation between A1_T and A1_rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
            u_cold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() (SI).
                
        Returns:
            M_max (float):
                Mass of the planet (SI).            
    """    
    R_min = 0.
    
    # Try integrating the profile with the minimum radius 
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
        num_prof, R_max, M, P_s, T_s, rho_s, mat_id, T_rho_type, T_rho_args, 
        u_cold_array
        )
        
    if A1_m_enc[-1] != 0: 
        print("R_max is too low, did not ran out of mass in first iteration")
        return 0.
            
    # Iterate the radius
    for i in tqdm(range(num_attempt), desc="Finding R given M"):
        R_try = (R_min + R_max) * 0.5
        
        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
            num_prof, R_try, M, P_s, T_s, rho_s, mat_id, T_rho_type, T_rho_args, 
            u_cold_array
            )
        
        if A1_m_enc[-1] > 0.:
            R_min = R_try
        else:
            R_max = R_try
        
    return R_min
     
# ===================== 2 layers ============================================= #
    
@jit(nopython=True)
def L2_integrate(
    num_prof, R, M, P_s, T_s, rho_s, R1, mat_id_L1, T_rho_type_L1, 
    T_rho_args_L1,mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, 
    u_cold_array_L2
    ):
    """ Integration of a 2 layer spherical planet.
    
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
                
            R1 (float):
                Boundary between layers 1 and 2 (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.                
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
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
    A1_r        = np.linspace(R, 0, int(num_prof))
    A1_m_enc    = np.zeros(A1_r.shape)
    A1_P        = np.zeros(A1_r.shape)
    A1_T        = np.zeros(A1_r.shape)
    A1_rho      = np.zeros(A1_r.shape)
    A1_u        = np.zeros(A1_r.shape)
    A1_mat_id   = np.zeros(A1_r.shape)
    
    u_s = weos._find_u(rho_s, mat_id_L2, T_s, u_cold_array_L2)
    if T_rho_type_L2 == 1:
        T_rho_args_L2[0] = T_s*rho_s**(-T_rho_args_L2[1])
    
    dr = A1_r[0] - A1_r[1]
    
    A1_m_enc[0]     = M
    A1_P[0]         = P_s
    A1_T[0]         = T_s
    A1_rho[0]       = rho_s
    A1_u[0]         = u_s
    A1_mat_id[0]    = mat_id_L2 
    
    for i in range(1, A1_r.shape[0]):
        # Layer 2
        if A1_r[i] > R1:
            rho             = A1_rho[i - 1]
            mat_id          = mat_id_L2
            T_rho_type      = T_rho_type_L2
            T_rho_args      = T_rho_args_L2
            rho0            = rho
            u_cold_array    = u_cold_array_L2       
        # Layer 1, 2 boundary
        elif A1_r[i] <= R1 and A1_r[i - 1] > R1:
            rho = weos._find_rho_fixed_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L1, 
                                           u_cold_array_L1)            
            if T_rho_type_L1 == 1:
                T_rho_args_L1[0] = A1_T[i - 1] * rho**(-T_rho_args_L1[1])           
            mat_id          = mat_id_L1
            T_rho_type      = T_rho_type_L1
            T_rho_args      = T_rho_args_L1
            rho0            = A1_rho[i - 1]
            u_cold_array    = u_cold_array_L1          
        # Layer 1  
        elif A1_r[i] <= R1:
            rho             = A1_rho[i - 1]
            mat_id          = mat_id_L1
            T_rho_type      = T_rho_type_L1
            T_rho_args      = T_rho_args_L1
            rho0            = A1_rho[i - 1]
            u_cold_array    = u_cold_array_L1      
        
        A1_m_enc[i] = A1_m_enc[i - 1] - 4*np.pi*A1_r[i - 1]**2*rho*dr
        A1_P[i]     = A1_P[i - 1] + G*A1_m_enc[i - 1]*rho/(A1_r[i - 1]**2)*dr
        A1_rho[i]   = weos._find_rho(A1_P[i], mat_id, T_rho_type, 
                                     T_rho_args, rho0, 1.1*rho, u_cold_array)
        A1_T[i]     = weos.T_rho(A1_rho[i], T_rho_type, T_rho_args)
        A1_u[i]     = weos._find_u(A1_rho[i], mat_id, A1_T[i], u_cold_array)
        A1_mat_id[i] = mat_id
        
        if A1_m_enc[i] < 0:
            break
        
    return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id

@jit(nopython=True)
def L2_find_mass(
    num_prof, R, M_max, P_s, T_s, rho_s, R1, mat_id_L1, T_rho_type_L1, 
    T_rho_args_L1,mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, 
    u_cold_array_L2
    ):
    """ Finder of the total mass of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet. 
    
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
                
            R1 (float):
                Boundary between layers 1 and 2 (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
        Returns:
            M_max ([float]):
                Mass of the planet (SI).            
    """
    M_min = 0.
    
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
        num_prof, R, M_max, P_s, T_s, rho_s, R1, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, 
        u_cold_array_L2
        )
    
    if A1_m_enc[-1] > 0.:
        while np.abs(M_min - M_max) > 1e-10*M_min:
            M_try = (M_min + M_max) * 0.5
            
            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
                num_prof, R, M_try, P_s, T_s, rho_s, R1, mat_id_L1, 
                T_rho_type_L1, T_rho_args_L1, mat_id_L2, T_rho_type_L2, 
                T_rho_args_L2, u_cold_array_L1, u_cold_array_L2
                )
            
            if A1_m_enc[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

#@jit(nopython=True)
def L2_find_radius(
    num_prof, R_max, M, P_s, T_s, rho_s, R1, mat_id_L1, T_rho_type_L1, 
    T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, 
    u_cold_array_L2, num_attempt=40
    ):
    """ Finder of the total radius of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet. 
    
        Args:
            num_prof (int):
                Number of profile integration steps.
            
            R_max (float):
                Maximum radius of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            P_s (float):
                Pressure at the surface (SI).
                
            T_s (float):
                Temperature at the surface (SI).
                
            rho_s (float):
                Density at the surface (SI).
                
            R1 (float):
                Boundary between layers 1 and 2 (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
        Returns:
            M_max ([float]):
                Mass of the planet (SI).            
    """
    R_min = R1
    
    A1_r, A1_m_enc_1, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
        num_prof, R_max, M, P_s, T_s, rho_s, R1, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, 
        u_cold_array_L2
        )
    
    rho_s_L1 = weos._find_rho_fixed_P_T(P_s, T_s, mat_id_L1, u_cold_array_L1)
    
    A1_r, A1_m_enc_2, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
        num_prof, R1, M, P_s, T_s, rho_s_L1, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, u_cold_array_L1
        )
        
    if A1_m_enc_1[-1] > 0:
        print("R_max too low, excess of mass for R = R_max")
        return R_max
    
    if A1_m_enc_2[-1] == 0:
        print("R = R1 yields a planet which already lacks mass.")
        print("Try increase M or reduce R1.")
        return -1
    
    if A1_m_enc_1[-1] == 0.:
        for i in tqdm(range(num_attempt), desc="Finding R given M, R1"):
            R_try = (R_min + R_max) * 0.5
            
            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
                num_prof, R_try, M, P_s, T_s, rho_s, R1, mat_id_L1, 
                T_rho_type_L1, T_rho_args_L1, mat_id_L2, T_rho_type_L2, 
                T_rho_args_L2, u_cold_array_L1, u_cold_array_L2
                )
            
            if A1_m_enc[-1] > 0.:
                R_min = R_try
            else:
                R_max = R_try
        
    return R_min

#@jit(nopython=True)
def L2_find_R1(
    num_prof, R, M, P_s, T_s, rho_s, mat_id_L1, T_rho_type_L1, T_rho_args_L1,
    mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, u_cold_array_L2, 
    num_attempt=40
    ):
    """ Finder of the boundary of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet. 
    
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
                Temperature at the surface (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
        Returns:
            R1_min ([float]):
                Boundary of the planet (SI).            
    """    
    R1_min = 0.
    R1_max = R 
    
    rho_s_L2 = weos._find_rho_fixed_P_T(P_s, T_s, mat_id_L2, u_cold_array_L2)
    
    A1_r, A1_m_enc_1, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
        num_prof, R, M, P_s, T_s, rho_s_L2, mat_id_L2, T_rho_type_L2, 
        T_rho_args_L2, u_cold_array_L2
        )
        
    rho_s_L1 = weos._find_rho_fixed_P_T(P_s, T_s, mat_id_L1, u_cold_array_L1)
    
    A1_r, A1_m_enc_2, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L1_integrate(
        num_prof, R, M, P_s, T_s, rho_s_L1, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, u_cold_array_L1
        )
        
    if A1_m_enc_1[-1] == 0:
        print("Ran out of mass for a planet made of layer 2 material")
        print("Try increasing the mass or decreasing the radius")
        return R1_min
    
    elif A1_m_enc_2[-1] > 0:
        print("Excess of mass for a planet made of layer 1 material")
        print("Try decreasing the mass or increasing the radius")
        return R1_max
         
    for i in tqdm(range(num_attempt), desc="Finding R1 given R, M"):
        R1_try = (R1_min + R1_max) * 0.5
        
        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
            num_prof, R, M, P_s, T_s, rho_s, R1_try, mat_id_L1, T_rho_type_L1, 
            T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, 
            u_cold_array_L1, u_cold_array_L2
            )
        
        if A1_m_enc[-1] > 0.:
            R1_min = R1_try
        else:
            R1_max = R1_try
        
    return R1_min

# ===================== 3 layers ============================================= #
    
@jit(nopython=True)
def L3_integrate(
    num_prof, R, M, P_s, T_s, rho_s, R1, R2, mat_id_L1, T_rho_type_L1, 
    T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, mat_id_L3, 
    T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, u_cold_array_L2, 
    u_cold_array_L3
    ):
    """ Integration of a 2 layer spherical planet.
    
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
                
            R1 (float):
                Boundary between layers 1 and 2 (SI).
                
            R2 (float):
                Boundary between layers 2 and 3 (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.
                
            T_rho_type_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.
                
            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
            u_cold_array_L3 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 3 (SI).
                
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
    A1_r        = np.linspace(R, 0, int(num_prof))
    A1_m_enc    = np.zeros(A1_r.shape)
    A1_P        = np.zeros(A1_r.shape)
    A1_T        = np.zeros(A1_r.shape)
    A1_rho      = np.zeros(A1_r.shape)
    A1_u        = np.zeros(A1_r.shape)
    A1_mat_id   = np.zeros(A1_r.shape)
    
    u_s = weos._find_u(rho_s, mat_id_L3, T_s, u_cold_array_L3)
    if T_rho_type_L3 == 1:
        T_rho_args_L3[0] = T_s*rho_s**(-T_rho_args_L3[1])        
    
    dr = A1_r[0] - A1_r[1]
    
    A1_m_enc[0]     = M
    A1_P[0]         = P_s
    A1_T[0]         = T_s
    A1_rho[0]       = rho_s
    A1_u[0]         = u_s
    A1_mat_id[0]    = mat_id_L3
    
    for i in range(1, A1_r.shape[0]):
        # Layer 3
        if A1_r[i] > R2:
            rho             = A1_rho[i - 1]
            mat_id          = mat_id_L3
            T_rho_type      = T_rho_type_L3
            T_rho_args      = T_rho_args_L3
            rho0            = rho
            u_cold_array    = u_cold_array_L3        
        # Layer 2, 3 boundary
        elif A1_r[i] <= R2 and A1_r[i - 1] > R2:
            rho = weos._find_rho_fixed_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L2, 
                                           u_cold_array_L2)            
            if T_rho_type_L2 == 1:
                T_rho_args_L2[0] = A1_T[i - 1] * rho**(-T_rho_args_L2[1])           
            mat_id          = mat_id_L2
            T_rho_type      = T_rho_type_L2
            T_rho_args      = T_rho_args_L2
            rho0            = A1_rho[i - 1]
            u_cold_array    = u_cold_array_L2    
        # Layer 2
        elif A1_r[i] > R1:
            rho             = A1_rho[i - 1]
            mat_id          = mat_id_L2
            T_rho_type      = T_rho_type_L2
            T_rho_args      = T_rho_args_L2
            rho0            = rho
            u_cold_array    = u_cold_array_L2       
        # Layer 1, 2 boundary
        elif A1_r[i] <= R1 and A1_r[i - 1] > R1:
            rho = weos._find_rho_fixed_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L1, 
                                           u_cold_array_L1)            
            if T_rho_type_L1 == 1:
                T_rho_args_L1[0] = A1_T[i - 1] * rho**(-T_rho_args_L1[1])           
            mat_id          = mat_id_L1
            T_rho_type      = T_rho_type_L1
            T_rho_args      = T_rho_args_L1
            rho0            = A1_rho[i - 1]
            u_cold_array    = u_cold_array_L1          
        # Layer 1  
        elif A1_r[i] <= R1:
            rho             = A1_rho[i - 1]
            mat_id          = mat_id_L1
            T_rho_type      = T_rho_type_L1
            T_rho_args      = T_rho_args_L1
            rho0            = A1_rho[i - 1]
            u_cold_array    = u_cold_array_L1      
        
        A1_m_enc[i] = A1_m_enc[i - 1] - 4*np.pi*A1_r[i - 1]**2*rho*dr
        A1_P[i]     = A1_P[i - 1] + G*A1_m_enc[i - 1]*rho/(A1_r[i - 1]**2)*dr
        A1_rho[i]   = weos._find_rho(A1_P[i], mat_id, T_rho_type, 
                                     T_rho_args, rho0, 1.1*rho, u_cold_array)
        A1_T[i]     = weos.T_rho(A1_rho[i], T_rho_type, T_rho_args)
        A1_u[i]     = weos._find_u(A1_rho[i], mat_id, A1_T[i], u_cold_array)
        A1_mat_id[i] = mat_id
        
        if A1_m_enc[i] < 0:
            break
        
    return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id

@jit(nopython=True)
def L3_find_mass(
    num_prof, R, M_max, P_s, T_s, rho_s, R1, R2, mat_id_L1, T_rho_type_L1, 
    T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, mat_id_L3, 
    T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, u_cold_array_L2, 
    u_cold_array_L3
    ):
    """ Finder of the total mass of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet. 
    
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
                
            R1 (float):
                Boundary between layers 1 and 2 (SI).
                
            R2 (float):
                Boundary between layers 2 and 3 (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.
                
            T_rho_type_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.
                
            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
            u_cold_array_L3 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 3 (SI).
                
        Returns:
            
            M_max ([float]):
                Mass of the planet (SI).
            
    """    
    M_min = 0.
    
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof, R, M_max, P_s, T_s, rho_s, R1, R2, mat_id_L1, 
            T_rho_type_L1, T_rho_args_L1, mat_id_L2, T_rho_type_L2, 
            T_rho_args_L2, mat_id_L3, T_rho_type_L3, T_rho_args_L3,
            u_cold_array_L1, u_cold_array_L2, u_cold_array_L3
            )
    
    if A1_m_enc[-1] > 0.:
        while np.abs(M_min - M_max) > 1e-10*M_min:
            M_try = (M_min + M_max) * 0.5
            
            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
                num_prof, R, M_try, P_s, T_s, rho_s, R1, R2, mat_id_L1, 
                T_rho_type_L1, T_rho_args_L1, mat_id_L2, T_rho_type_L2, 
                T_rho_args_L2, mat_id_L3, T_rho_type_L3, T_rho_args_L3,
                u_cold_array_L1, u_cold_array_L2, u_cold_array_L3
                )
            
            if A1_m_enc[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

#@jit(nopython=True)
def L3_find_radius(
    num_prof, R_max, M, P_s, T_s, rho_s, R1, R2, mat_id_L1, T_rho_type_L1, 
    T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, mat_id_L3, 
    T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, u_cold_array_L2, 
    u_cold_array_L3, num_attempt=40
    ):
    """ Finder of the total mass of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet. 
    
        Args:
            num_prof (int):
                Number of profile integration steps.
            
            R_max (float):
                Maximum radius of the planet (SI).
                
            M (float):
                Mass of the planet (SI).
                
            P_s (float):
                Pressure at the surface (SI).
                
            T_s (float):
                Temperature at the surface (SI).
                
            rho_s (float):
                Density at the surface (SI).
                
            R1 (float):
                Boundary between layers 1 and 2 (SI).
                
            R2 (float):
                Boundary between layers 2 and 3 (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.
                
            T_rho_type_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.
                
            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
            u_cold_array_L3 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 3 (SI).
                
        Returns:
            
            M_max ([float]):
                Mass of the planet (SI).
            
    """
    if R1 > R2:
        print("R1 should not be greater than R2")
        return -1
    
    R_min = R2
    
    rho_s_L2 = weos._find_rho_fixed_P_T(P_s, T_s, mat_id_L2, u_cold_array_L2)
    
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
        num_prof, R2, M, P_s, T_s, rho_s_L2, R1, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, 
        u_cold_array_L2
        )
        
    if A1_m_enc[-1] == 0:
        print("Ran out of mass for a 2 layer planet.")
        print("Try increase the mass or reduce R1")
        return R_min
        
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
        num_prof, R_max, M, P_s, T_s, rho_s, R1, R2, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, mat_id_L3, 
        T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, u_cold_array_L2, 
        u_cold_array_L3
        )
        
    if A1_m_enc[-1] > 0:
        print("Excess of mass for a 3 layer planet with R = R_max.")
        print("Try reduce the mass or increase R_max")
        return R_max
        
    for i in tqdm(range(num_attempt), desc="Finding R given M, R1, R2"):
            R_try = (R_min + R_max) * 0.5
            
            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
                num_prof, R_try, M, P_s, T_s, rho_s, R1, R2, mat_id_L1, 
                T_rho_type_L1, T_rho_args_L1, mat_id_L2, T_rho_type_L2, 
                T_rho_args_L2, mat_id_L3, T_rho_type_L3, T_rho_args_L3,
                u_cold_array_L1, u_cold_array_L2, u_cold_array_L3
                )
            
            if A1_m_enc[-1] > 0.:
                R_min = R_try
            else:
                R_max = R_try
        
    return R_min

#@jit(nopython=True)
def L3_find_R2(
    num_prof, R, M, P_s, T_s, rho_s, R1, mat_id_L1, T_rho_type_L1, 
    T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, mat_id_L3, 
    T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, u_cold_array_L2, 
    u_cold_array_L3, num_attempt=40
    ):
    """ Finder of the boundary between layers 2 and 3 of the planet for
        fixed boundary between layers 1 and 2.
        The correct value yields A1_m_enc -> 0 at the center of the planet. 
    
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
                
            R1 (float):
                Boundary between layers 1 and 2 (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.
                
            T_rho_type_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.
                
            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                             
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
            u_cold_array_L3 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 3 (SI).
                
        Returns:
            R2_max ([float]):
                Boundary between layers 2 and 3 of the planet (SI).            
    """
    R2_min = R1
    R2_max = R
    
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
        num_prof, R, M, P_s, T_s, rho_s, R1, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L3, T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, 
        u_cold_array_L3
        )
        
    if A1_m_enc[-1] == 0:
        print("A planet made of layer 1 and layer 2 materials excess mass.")  
        print("Try decreasing the mass, decreasing R1 or increasing R")
        return R2_min
        
    rho_s_L2 = weos._find_rho_fixed_P_T(P_s, T_s, mat_id_L2, u_cold_array_L2)
    
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
        num_prof, R, M, P_s, T_s, rho_s_L2, R1, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, 
        u_cold_array_L2
        )
        
    if A1_m_enc[-1] > 0:
        print("A planet made of layer 1 and layer 3 materials lacks mass.")
        print("Try increasing the mass, increasing R1 or decreasing R.") 
        
        return R2_max
    
    for i in tqdm(range(num_attempt), desc="Finding R2 given M, R, R1"):
            
        R2_try = (R2_min + R2_max) * 0.5
        
        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof, R, M, P_s, T_s, rho_s, R1, R2_try, mat_id_L1, 
            T_rho_type_L1, T_rho_args_L1, mat_id_L2, T_rho_type_L2, 
            T_rho_args_L2, mat_id_L3, T_rho_type_L3, T_rho_args_L3,
            u_cold_array_L1, u_cold_array_L2, u_cold_array_L3
            )
            
        if A1_m_enc[-1] > 0.:
            R2_min = R2_try
        else:
            R2_max = R2_try
        
    return R2_max

#@jit(nopython=True)
def L3_find_R1(
    num_prof, R, M, P_s, T_s, rho_s, R2, mat_id_L1, T_rho_type_L1, 
    T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, mat_id_L3, 
    T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, u_cold_array_L2, 
    u_cold_array_L3, num_attempt=40
    ):
    """ Finder of the boundary between layers 2 and 3 of the planet for
        fixed boundary between layers 1 and 2.
        The correct value yields A1_m_enc -> 0 at the center of the planet. 
    
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
                
            R2 (float):
                Boundary between layers 2 and 3 (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.
                
            T_rho_type_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.
                
            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                             
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
            u_cold_array_L3 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 3 (SI).
                
        Returns:
            R2_max ([float]):
                Boundary between layers 2 and 3 of the planet (SI).            
    """
    R1_min = 0.
    R1_max = R2
    
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
        num_prof, R, M, P_s, T_s, rho_s, R2, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L3, T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, 
        u_cold_array_L3
        )
        
    if A1_m_enc[-1] > 0:
        print("A planet made of layer 1 and layer 3 materials excess mass.")
        print("Try decreasing the mass, increasing R2 or increasing R")
        return R1_min
    
    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_integrate(
        num_prof, R, M, P_s, T_s, rho_s, R2, mat_id_L2, T_rho_type_L2, 
        T_rho_args_L2, mat_id_L3, T_rho_type_L3, T_rho_args_L3, u_cold_array_L2, 
        u_cold_array_L3
        )
        
    if A1_m_enc[-1] == 0:
        print("A planet made of layer 2 and layer 3 materials lacks mass.")  
        print("Try increasing the mass, increasing R2 or decreasing R")
        return R1_max
    
    for i in tqdm(range(num_attempt), desc="Finding R1 given R, M, R2"):
            
        R1_try = (R1_min + R1_max) * 0.5
        
        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof, R, M, P_s, T_s, rho_s, R1_try, R2, mat_id_L1, 
            T_rho_type_L1, T_rho_args_L1, mat_id_L2, T_rho_type_L2, 
            T_rho_args_L2, mat_id_L3, T_rho_type_L3, T_rho_args_L3,
            u_cold_array_L1, u_cold_array_L2, u_cold_array_L3
            )
            
        if A1_m_enc[-1] > 0.:
            R1_min = R1_try
        else:
            R1_max = R1_try
        
    return R1_max

#@jit(nopython=True)
def L3_find_R1_R2(
    num_prof, R, M, P_s, T_s, rho_s, MoI, mat_id_L1, T_rho_type_L1, 
    T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, mat_id_L3, 
    T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, u_cold_array_L2, 
    u_cold_array_L3, num_attempt=20, num_attempt_2=10
    ):
    """ Finder of the boundaries of the planet for a
        fixed moment of inertia.
        The correct value yields A1_m_enc -> 0 at the center of the planet. 
    
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
                
            MoI (float):
                moment of inertia (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.
                
            T_rho_type_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.
                
            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
            u_cold_array_L3 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 3 (SI).
                
        Returns:
            R1, R2 ([float]):
                Boundaries between layers 1 and 2 and between layers 2 and 3 of 
                the planet (SI).            
    """
    rho_s_L2 = weos._find_rho_fixed_P_T(P_s, T_s, mat_id_L2, u_cold_array_L2)
    
    R1_I_max = L2_find_R1(
        num_prof, R, M, P_s, T_s, rho_s_L2, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, 
        u_cold_array_L2
        )
    
    R1_I_min = L2_find_R1(
        num_prof, R, M, P_s, T_s, rho_s, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L3, T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, 
        u_cold_array_L3
        )
    
    r_max, A1_m_enc, A1_P, A1_T, rho_23x, A1_u, A1_mat_id = L2_integrate(
        num_prof, R, M, P_s, T_s, rho_s_L2, R1_I_max, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L1, 
        u_cold_array_L2
        )
        
    r_min, A1_m_enc, A1_P, A1_T, rho_min, A1_u, A1_mat_id = L2_integrate(
        num_prof, R, M, P_s, T_s, rho_s, R1_I_min, mat_id_L1, T_rho_type_L1, 
        T_rho_args_L1, mat_id_L3, T_rho_type_L3, T_rho_args_L3, u_cold_array_L1, 
        u_cold_array_L3
        )
        
    moi_min = moi(r_min, rho_min)
    moi_max = moi(r_max, rho_23x)
    
    R1_min = R1_I_max
    R1_max = R1_I_min
    
    if MoI > moi_min and  MoI < moi_max:
        for i in tqdm(range(num_attempt), 
                      desc="Finding R1, R2 given R, M, I_MR2"):
            R1_try = (R1_min + R1_max) * 0.5
            
            R2_try = L3_find_R2(
                num_prof, R, M, P_s, T_s, rho_s, R1_try, mat_id_L1, 
                T_rho_type_L1, T_rho_args_L1, mat_id_L2, T_rho_type_L2, 
                T_rho_args_L2, mat_id_L3, T_rho_type_L3, T_rho_args_L3,
                u_cold_array_L1, u_cold_array_L2, u_cold_array_L3, 
                num_attempt_2
                )
                    
            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
                num_prof, R, M, P_s, T_s, rho_s, R1_try, R2_try, mat_id_L1, 
                T_rho_type_L1, T_rho_args_L1, mat_id_L2, T_rho_type_L2, 
                T_rho_args_L2, mat_id_L3, T_rho_type_L3, T_rho_args_L3,
                u_cold_array_L1, u_cold_array_L2, u_cold_array_L3
                )
            
            if moi(A1_r,A1_rho) < MoI:
                R1_max = R1_try
            else:
                R1_min = R1_try
                
    elif MoI > moi_max:
        print("Moment of interia is too high,")
        print("maximum value is:")
        print(moi_max/M_earth/R_earth/R_earth,"[M_earth R_earth^2]")
        R1_try = 0.
        R2_try = 0.
    
    elif MoI < moi_min:
        print("Moment of interia is too low,")
        print("minimum value is:")
        print(moi_min/M_earth/R_earth/R_earth,"[M_earth R_earth^2]")
        R1_try = 0.
        R2_try = 0.
        
    else:
        print("Something went wrong")
        R1_try = 0.
        R2_try = 0.
        
    return R1_try, R2_try

# ============================================================================ #
# ===================== Spherical profile class ============================== #
# ============================================================================ #

class Planet():
    """ Planet class ...
    
        Args:
            name (str)
                The name of the planet object.
                
            Fp_planet (str)
                The object data file path. Default to "data/<name>.hdf5".

            A1_mat_layer ([str])
                The name of the material in each layer, from the central layer
                outwards. See Di_mat_id in weos.py.

            A1_T_rho_type ([str])
                The type of temperature-density relation in each layer, from the
                central layer outwards.

                Options...

            A1_T_rho_args ([?])
                ...

            A1_R_layer ([float])
                The outer radii of each layer, from the central layer outwards
                (SI).

            A1_M_layer ([float])
                The mass within each layer, starting from the from the central
                layer outwards (SI).

            M (float)
                The total mass.

            P_s, T_s, rho_s (float)
                The pressure, temperature, and density at the surface. Only two
                of the three must be provided (Pa, K, kg m^-3).

            I_MR2 (float)
                The reduced moment of inertia. (SI)
            
            M_max (float)
                ...
            
            R_max (float)
                ...
                
            rho_min (float)
                The minimum density for the outer edge of the profile.
                
            num_prof (int)
                The number of profile integration steps.
                
            num_attempt, num_attempt_2 (int)
                The maximum number of iteration attempts.
                
        Attrs (in addition to the args):
            A1_mat_id_layer ([int])
                The ID of the material in each layer, from the central layer
                outwards.
            
            num_layer (int)
                The number of planetary layers.
        
            A1_r ([float])
                The profile radii, in increasing order.
                
            ...
    """    
    def __init__(
        self, name, Fp_planet=None, A1_mat_layer=None, A1_T_rho_type=None, 
        A1_T_rho_args=None, A1_R_layer=None, A1_M_layer=None, M=None, P_s=None, 
        T_s=None, rho_s=None, I_MR2=None, M_max=None, R_max=None, rho_min=None, 
        num_prof=10000, num_attempt=40, num_attempt_2=40
        ):
        self.name               = name
        self.Fp_planet          = Fp_planet
        self.A1_mat_layer       = A1_mat_layer
        self.A1_T_rho_type      = A1_T_rho_type
        self.A1_T_rho_args      = A1_T_rho_args
        self.A1_R_layer         = A1_R_layer
        self.A1_M_layer         = A1_M_layer
        self.M                  = M
        self.P_s                = P_s
        self.T_s                = T_s
        self.rho_s              = rho_s
        self.I_MR2              = I_MR2
        self.M_max              = M_max
        self.R_max              = R_max
        self.rho_min            = rho_min
        self.num_prof           = num_prof
        self.num_attempt        = num_attempt
        self.num_attempt_2      = num_attempt_2
        
        # Derived or default attributes
        if self.A1_mat_layer is not None:
            self.num_layer          = len(self.A1_mat_layer)
            self.A1_mat_id_layer    = [weos.Di_mat_id[mat] 
                                       for mat in self.A1_mat_layer]
        else:
            # Placeholder
            self.num_layer  = 1
        if self.Fp_planet is None:
            self.Fp_planet  = "data/%s.hdf5" % self.name
        if self.A1_R_layer is None:
            self.A1_R_layer = [None] * self.num_layer
        if self.A1_M_layer is None:
            self.A1_M_layer = [None] * self.num_layer
        self.R  = self.A1_R_layer[-1]
        
        # Force types for numba
        if self.A1_R_layer is not None:
            self.A1_R_layer = np.array(self.A1_R_layer, dtype="float")
        if self.A1_T_rho_args is not None:
            self.A1_T_rho_args = np.array(self.A1_T_rho_args, dtype="float")
        
        # Two of P, T, rho must be provided at the surface to calculate the 
        # third. If all three are provided then rho is overwritten.
        if self.P_s is not None and self.T_s is not None:
            self.rho_s  = weos.find_rho_fixed_P_T(self.P_s, self.T_s, 
                                                  self.A1_mat_id_layer[-1])
        ###todo:
        # elif self.P_s is not None and self.rho_s is not None:
        #     self.T_s    = weos.find_T_fixed_P_rho(self.P_s, self.rho_s, 
        #                                           self.A1_mat_id_layer[-1])
        # elif self.rho_s is not None and self.T_s is not None:
        #     self.P_s    = weos.find_P_fixed_rho_T(self.rho_s, self.T_s, 
        #                                           self.A1_mat_id_layer[-1])
        
        ### default M_max and R_max?
        
        # Help info ###todo
        if not True:
            if self.num_layer == 1:
                print("For a 1 layer planet, please specify:")
                print("pressure, temperature and density at the surface of the planet,")
                print("material, relation between temperature and density with any desired aditional parameters,")
                print("for layer 1 of the planet.")
            elif self.num_layer == 2:
                print("For a 2 layer planet, please specify:")
                print("pressure, temperature and density at the surface of the planet,")
                print("materials, relations between temperature and density with any desired aditional parameters,")
                print("for layer 1 and layer 2 of the planet.")
            elif self.num_layer == 3:
                print("For a 3 layer planet, please specify:")
                print("pressure, temperature and density at the surface of the planet,")
                print("materials, relations between temperature and density with any desired aditional parameters,")
                print("for layer 1, layer 2, and layer 3 of the planet.")
                        
    # ========
    # General
    # ========
    def update_attributes(self):
        """ Set all planet information after making the profiles.
        """
        self.num_prof   = len(self.A1_r)
        
        # Reverse profile arrays to be ordered by increasing radius
        if self.A1_r[-1] < self.A1_r[0]:
            self.A1_r       = self.A1_r[::-1]
            self.A1_m_enc   = self.A1_m_enc[::-1]
            self.A1_P       = self.A1_P[::-1]
            self.A1_T       = self.A1_T[::-1]
            self.A1_rho     = self.A1_rho[::-1]
            self.A1_u       = self.A1_u[::-1]
            self.A1_mat_id  = self.A1_mat_id[::-1]
        
        # Index of the outer edge of each layer
        self.A1_idx_layer   = np.append(
            np.where(np.diff(self.A1_mat_id) != 0)[0], self.num_prof - 1)
            
        # Boundary radii
        self.A1_R_layer = self.A1_r[self.A1_idx_layer]
        self.R          = self.A1_R_layer[-1]
        
        # Layer masses
        self.A1_M_layer = self.A1_m_enc[self.A1_idx_layer]
        if self.num_layer > 1:
            self.A1_M_layer[1:] -= self.A1_M_layer[:-1]
        self.M  = np.sum(self.A1_M_layer)
        
        # Moment of inertia
        self.I_MR2  = moi(self.A1_r, self.A1_rho)
        
        # Update P_s, T_s and rho_s for the case create L3 profile from L2 profile
        self.P_s = self.A1_P[-1]
        self.T_s = self.A1_T[-1]
        self.rho_s = self.A1_rho[-1]
        
    def print_info(self):
        """ Print the Planet objects's main properties. """
        space   = 12
        print("Planet \"%s\": " % self.name)
        print("    %s = %.5g kg / %.5g M_earth" % 
              (add_whitespace("M", space), self.M, self.M/M_earth))
        print("    %s = %.5g m / %.5g R_earth" %
              (add_whitespace("R", space), self.R, self.R/R_earth))
        print("    %s = %s " % (add_whitespace("mat", space), 
              format_array_string(self.A1_mat_layer, "%s")))
        print("    %s = %s " % (add_whitespace("mat_id", space), 
              format_array_string(self.A1_mat_id_layer, "%d")))
        print("    %s = %s R_earth" % (add_whitespace("R_layer", space), 
              format_array_string(self.A1_R_layer / R_earth, "%.5g")))
        print("    %s = %s M_earth" % (add_whitespace("M_layer", space), 
              format_array_string(self.A1_M_layer / M_earth, "%.5g")))
        print("    %s = %s M_total" % (add_whitespace("M_frac_layer", space), 
              format_array_string(self.A1_M_layer / self.M, "%.5g")))
        print("    %s = %s " % (add_whitespace("idx_layer", space), 
              format_array_string(self.A1_idx_layer, "%d")))
        print("    %s = %.5g Pa" % (add_whitespace("P_s", space), self.P_s))
        print("    %s = %.5g K" % (add_whitespace("T_s", space), self.T_s))
        print("    %s = %.5g kg/m^3" % (add_whitespace("rho_s", space), self.rho_s))
        print("    %s = %.5g kg*m^2 / %.5g M_earth*R_earth^2" %
              (add_whitespace("I_MR2", space), self.I_MR2, self.I_MR2/M_earth/R_earth/R_earth))
    
    def save_planet(self):
        Fp_planet = check_end(self.Fp_planet, ".hdf5")
        
        print("Saving \"%s\"... " % Fp_planet[-60:], end='')
        sys.stdout.flush()
        
        with h5py.File(Fp_planet, "w") as f:
            # Group
            grp = f.create_group("/planet")

            # Attributes
            grp.attrs[Di_hdf5_planet_label["num_layer"]]    = self.num_layer
            grp.attrs[Di_hdf5_planet_label["mat_layer"]]    = self.A1_mat_layer
            grp.attrs[Di_hdf5_planet_label["mat_id_layer"]] = self.A1_mat_id_layer
            grp.attrs[Di_hdf5_planet_label["T_rho_type"]]   = self.A1_T_rho_type
            grp.attrs[Di_hdf5_planet_label["T_rho_args"]]   = self.A1_T_rho_args
            grp.attrs[Di_hdf5_planet_label["R_layer"]]      = self.A1_R_layer
            grp.attrs[Di_hdf5_planet_label["M_layer"]]      = self.A1_M_layer
            grp.attrs[Di_hdf5_planet_label["M"]]            = self.M
            grp.attrs[Di_hdf5_planet_label["R"]]            = self.R
            grp.attrs[Di_hdf5_planet_label["idx_layer"]]    = self.A1_idx_layer
            grp.attrs[Di_hdf5_planet_label["P_s"]]          = self.P_s
            grp.attrs[Di_hdf5_planet_label["T_s"]]          = self.T_s
            grp.attrs[Di_hdf5_planet_label["rho_s"]]        = self.rho_s

            # Arrays
            grp.create_dataset(
                Di_hdf5_planet_label["r"], data=self.A1_r, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["m_enc"], data=self.A1_m_enc, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["rho"], data=self.A1_rho, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["T"], data=self.A1_T, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["P"], data=self.A1_P, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["u"], data=self.A1_u, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["mat_id"], data=self.A1_mat_id, dtype="i")
                        
        print("Done")
        
    def load_planet_profiles(self):
        """ Load the profiles arrays for an existing Planet object from a file.
        """
        Fp_planet = check_end(self.Fp_planet, ".hdf5")
        
        print("Loading \"%s\"... " % Fp_planet[-60:], end='')
        sys.stdout.flush()
        
        with h5py.File(Fp_planet, "r") as f:
            (self.A1_r, self.A1_m_enc, self.A1_rho, self.A1_T, self.A1_P, 
             self.A1_u, self.A1_mat_id) = multi_get_planet_data(
                f, ["r", "m_enc", "rho", "T", "P", "u", "mat_id"])
                
        print("Done")
        
    # ========
    # 1 Layer
    # ========
    def gen_prof_L1_fix_R_given_M(self):
        """ Compute the profile of a planet with 1 layer by finding the correct 
            radius for a given mass.
        """
        # Check for necessary input
        assert(self.num_layer == 1)
        assert(self.R_max is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        
        u_cold_array = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        
        self.R = L1_find_radius(
            self.num_prof, self.R_max, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], u_cold_array, self.num_attempt
            )
        self.A1_R_layer[-1] = self.R
        
        print("Tweaking M to avoid peaks at the center of the planet...")
        
        self.M = L1_find_mass(
            self.num_prof, self.R, 2*self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], u_cold_array
            )
        
        print("Done!")
              
        # Integrate the profiles  
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L1_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], u_cold_array
            )

        self.update_attributes()
        self.print_info()
    
    def gen_prof_L1_fix_M_given_R(self):
        # Check for necessary input
        assert(self.num_layer == 1)
        assert(self.R is not None)
        assert(self.M_max is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
    
        u_cold_array = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        
        print("Finding M given R...")
        
        self.M = L1_find_mass(
            self.num_prof, self.R, self.M_max, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], u_cold_array
            )
            
        print("Done!")
                  
        # Integrate the profiles           
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L1_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], u_cold_array
            )

        self.update_attributes()
        self.print_info()
    
    def gen_prof_L1_given_R_M(self):
        # Check for necessary input
        assert(self.num_layer == 1)
        assert(self.R is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        
        u_cold_array = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        
        # Integrate the profiles
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L1_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], u_cold_array
            )

        self.update_attributes()
        self.print_info()
    
    # ========
    # 2 Layers
    # ========
    def gen_prof_L2_fix_R1_given_R_M(self):
        # Check for necessary input
        assert(self.num_layer == 2)
        assert(self.R is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        
        self.A1_R_layer[0] = L2_find_R1(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2, self.num_attempt
            )
        
        print("Tweaking M to avoid peaks at the center of the planet...")
        
        self.M = L2_find_mass(
            self.num_prof, self.R, 2 * self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )
        
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L2_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )
            
        print("Done!")

        self.update_attributes()
        self.print_info()
            
    def gen_prof_L2_fix_M_given_R1_R(self):
        # Check for necessary input
        assert(self.num_layer == 2)
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.M_max is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        
        print("Finding M given R1 and R...")
        
        self.M = L2_find_mass(
            self.num_prof, self.R, self.M_max, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )
            
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L2_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )
            
        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L2_fix_R_given_M_R1(self):
        # Check for necessary input
        assert(self.num_layer == 2)
        assert(self.A1_R_layer[0] is not None)
        assert(self.R_max is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        
        self.R = L2_find_radius(
            self.num_prof, self.R_max, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2, self.num_attempt
            )
        self.A1_R_layer[-1] = self.R
        
        print("Tweaking M to avoid peaks at the center of the planet...")
        
        self.M = L2_find_mass(
            self.num_prof, self.R, 2 * self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )
        
        print("Done!")
        
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L2_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )

        self.update_attributes()
        self.print_info()
        
    def gen_prof_L2_fix_R1_R_given_M1_M2(self, M1=None, M2=None):
        # Check for necessary input
        if M1 is not None:
            self.A1_M_layer[0]  = M1
        if M2 is not None:
            self.A1_M_layer[1]  = M2
        assert(self.num_layer == 2)
        assert(self.A1_M_layer[0] is not None)
        assert(self.A1_M_layer[1] is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        self.M  = np.sum(self.A1_M_layer)
                
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        
        ###WIP make inputs/attrs
        R_min   = 0.95 * R_earth
        R_max   = 0.98 * R_earth
        M_tol   = 0.01 * self.M
        
        ###Tidy this! Replace with other function, verb=0.
        # Check the maximum radius yields a too small layer 1 mass
        print("R_max = %.5g m = %.5g R_E " % (R_min, R_min / R_earth))
        self.A1_R_layer[0] = L2_find_R1(
            self.num_prof, R_max, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2, self.num_attempt
            )        
        self.M = L2_find_mass(
            self.num_prof, R_max, 2 * self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L2_integrate(
            self.num_prof, R_max, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )
        M1  = self.A1_m_enc[self.A1_mat_id == self.A1_mat_id_layer[0]][0]
        print("    --> M1 = %.5g kg = %.5g M_E = %.4f M" 
              % (M1, M1 / M_earth, M1 / self.M))
        assert(M1 < self.A1_M_layer[0])
               
        # Check the minimum radius yields a too large layer 1 mass
        print("R_min = %.5g m = %.5g R_E " % (R_min, R_min / R_earth))
        self.A1_R_layer[0] = L2_find_R1(
            self.num_prof, R_min, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2, self.num_attempt
            )        
        self.M = L2_find_mass(
            self.num_prof, R_min, 2 * self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L2_integrate(
            self.num_prof, R_min, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )
        M1  = self.A1_m_enc[self.A1_mat_id == self.A1_mat_id_layer[0]][0]
        print("    --> M1 = %.5g kg = %.5g M_E = %.4f M" 
              % (M1, M1 / M_earth, M1 / self.M))
        assert(M1 > self.A1_M_layer[0])        
        
        # Iterate to obtain desired layer masses
        iter    = 0
        M1      = 0
        while M_tol < abs(M1 - self.A1_M_layer[0]):
            R_try = (R_min + R_max) * 0.5
            print("iter %d: R = %.5g m = %.5g R_E" 
                  % (iter, R_try, R_try / R_earth))
            
            self.A1_R_layer[0] = L2_find_R1(
                self.num_prof, R_try, self.M, self.P_s, self.T_s, self.rho_s,
                self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
                self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
                self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
                u_cold_array_L2, self.num_attempt
                )            
            self.M = L2_find_mass(
                self.num_prof, R_try, 2 * self.M, self.P_s, self.T_s, self.rho_s, 
                self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
                self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
                self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
                u_cold_array_L2
                )            
            (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
             self.A1_mat_id) = L2_integrate(
                self.num_prof, R_try, self.M, self.P_s, self.T_s, self.rho_s, 
                self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
                self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
                self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
                u_cold_array_L2
                )
        
            M1  = self.A1_m_enc[self.A1_mat_id == self.A1_mat_id_layer[0]][0]
            print("    --> M1 = %.5g kg = %.5g M_E = %.4f M" 
                  % (M1, M1 / M_earth, M1 / self.M))
            
            if M1 < self.A1_M_layer[0]:
                R_max = R_try
            else:
                R_min = R_try
            
            iter    += 1

        self.update_attributes()
        self.print_info()

    def gen_prof_L2_given_R_M_R1(self):
        # Check for necessary input
        assert(self.num_layer == 2)
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L2_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], u_cold_array_L1, 
            u_cold_array_L2
            )

        self.update_attributes()
        self.print_info()
        
    # ========
    # 3 Layers
    # ========
    def gen_prof_L3_fix_R1_R2_given_R_M_I(self):
        # Check for necessary input
        assert(self.num_layer == 3)
        assert(self.R is not None)
        assert(self.M is not None)
        assert(self.I_MR2 is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        assert(self.A1_mat_id_layer[2] is not None)
        assert(self.A1_T_rho_type[2] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        u_cold_array_L3 = weos.load_u_cold_array(self.A1_mat_id_layer[2])
        
        self.A1_R_layer[0], self.A1_R_layer[1] = L3_find_R1_R2(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.I_MR2, self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], 
            self.A1_mat_id_layer[2], self.A1_T_rho_type[2], 
            self.A1_T_rho_args[2], u_cold_array_L1, u_cold_array_L2, 
            u_cold_array_L3, self.num_attempt, self.num_attempt_2
            )

        print("Tweaking M to avoid peaks at the center of the planet...")
        
        self.M = L3_find_mass(
            self.num_prof, self.R, 2 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0], 
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )
        
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0], 
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )
            
        print("Done!")

        self.update_attributes()
        self.print_info()
        
    def gen_prof_L3_fix_R2_given_R_M_R1(self):
        # Check for necessary input
        assert(self.num_layer == 3)
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        assert(self.A1_mat_id_layer[2] is not None)
        assert(self.A1_T_rho_type[2] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        u_cold_array_L3 = weos.load_u_cold_array(self.A1_mat_id_layer[2])
        
        self.A1_R_layer[1] = L3_find_R2(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], 
            self.A1_mat_id_layer[2], self.A1_T_rho_type[2], 
            self.A1_T_rho_args[2], u_cold_array_L1, u_cold_array_L2, 
            u_cold_array_L3, self.num_attempt
            )
        
        print("Tweaking M to avoid peaks at the center of the planet...")
        
        self.M = L3_find_mass(
            self.num_prof, self.R, 2 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0], 
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2],
            u_cold_array_L1, u_cold_array_L2, u_cold_array_L3
            )
            
        print("Done!")

        self.update_attributes()
        self.print_info()
    
    def gen_prof_L3_fix_R1_given_R_M_R2(self):
        # Check for necessary input
        assert(self.num_layer == 3)
        assert(self.R is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        assert(self.A1_mat_id_layer[2] is not None)
        assert(self.A1_T_rho_type[2] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        u_cold_array_L3 = weos.load_u_cold_array(self.A1_mat_id_layer[2])
        
        self.A1_R_layer[0] = L3_find_R1(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s, 
            self.A1_R_layer[1], self.A1_mat_id_layer[0], self.A1_T_rho_type[0], 
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1], 
            self.A1_T_rho_type[1], self.A1_T_rho_args[1], 
            self.A1_mat_id_layer[2], self.A1_T_rho_type[2], 
            self.A1_T_rho_args[2], u_cold_array_L1, u_cold_array_L2, 
            u_cold_array_L3, self.num_attempt
            )
        
        print("Tweaking M to avoid peaks at the center of the planet...")
        
        self.M = L3_find_mass(
            self.num_prof, self.R, 2 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0], 
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0], 
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )
            
        print("Done!")
        
        self.update_attributes()
        self.print_info()
    
    def gen_prof_L3_fix_M_given_R_R1_R2(self):
        # Check for necessary input
        assert(self.num_layer == 3)
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M_max is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        assert(self.A1_mat_id_layer[2] is not None)
        assert(self.A1_T_rho_type[2] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        u_cold_array_L3 = weos.load_u_cold_array(self.A1_mat_id_layer[2])
        
        print("Finding M given R1, R2 and R...")
        
        self.M = L3_find_mass(
            self.num_prof, self.R, self.M_max, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0], 
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )
            
        print("Done!")
        
        self.update_attributes()
        self.print_info()
    
    def gen_prof_L3_fix_R_given_M_R1_R2(self):
        # Check for necessary input
        assert(self.num_layer == 3)
        assert(self.R_max is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        assert(self.A1_mat_id_layer[2] is not None)
        assert(self.A1_T_rho_type[2] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        u_cold_array_L3 = weos.load_u_cold_array(self.A1_mat_id_layer[2])
        
        self.R = L3_find_radius(
            self.num_prof, self.R_max, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3, self.num_attempt
            )
        self.A1_R_layer[-1] = self.R
        
        print("Tweaking M to avoid peaks at the center of the planet...")
        
        self.M = L3_find_mass(
            self.num_prof, self.R, 2 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0], 
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )
            
        print("Done!")
        
        self.update_attributes()
        self.print_info()
    
    def gen_prof_L3_given_R_M_R1_R2(self):
        # Check for necessary input
        assert(self.num_layer == 3)
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        assert(self.A1_mat_id_layer[2] is not None)
        assert(self.A1_T_rho_type[2] is not None)
        
        u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
        u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
        u_cold_array_L3 = weos.load_u_cold_array(self.A1_mat_id_layer[2])
        
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u, 
         self.A1_mat_id) = L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0], 
            self.A1_T_rho_type[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type[1], 
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2], 
            self.A1_T_rho_type[2], self.A1_T_rho_args[2], u_cold_array_L1, 
            u_cold_array_L2, u_cold_array_L3
            )
            
        self.update_attributes()
        self.print_info()
    
    def gen_prof_L3_given_prof_L2(self, mat=None, T_rho_type=None, 
                                  T_rho_args=None, rho_min=None):
        """ Add a third layer (atmosphere) on top of existing 2 layer profiles.
        
            Args or set attributes:
                ...
                
            Sets:
                ...
        """   
        # Check for necessary input
        assert(self.num_layer == 2)
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type[1] is not None)
        
        self.num_layer   = 3
        if mat is not None:
            self.A1_mat_layer       = np.append(self.A1_mat_layer, mat)
            self.A1_mat_id_layer    = [weos.Di_mat_id[mat] 
                                       for mat in self.A1_mat_layer]
        if T_rho_type is not None:
            self.A1_T_rho_type = np.append(self.A1_T_rho_type, T_rho_type)
        if T_rho_args is not None:
            A1_T_rho_args_aux = np.zeros((3,2))
            A1_T_rho_args_aux[0:2] = self.A1_T_rho_args
            A1_T_rho_args_aux[2] = np.array(T_rho_args, dtype='float')
            self.A1_T_rho_args = A1_T_rho_args_aux
        if rho_min is not None:
            self.rho_min    = rho_min
            
        dr              = self.A1_r[1]
        mat_id_L3       = self.A1_mat_id_layer[2]        
        u_cold_array_L3 = weos.load_u_cold_array(mat_id_L3)
        
        # Reverse profile arrays to be ordered by increasing radius
        if self.A1_r[-1] < self.A1_r[0]:
            self.A1_r       = self.A1_r[::-1]
            self.A1_m_enc   = self.A1_m_enc[::-1]
            self.A1_P       = self.A1_P[::-1]
            self.A1_T       = self.A1_T[::-1]
            self.A1_rho     = self.A1_rho[::-1]
            self.A1_u       = self.A1_u[::-1]
            self.A1_mat_id  = self.A1_mat_id[::-1]
        
        # Initialise the new profiles
        A1_r        = [self.A1_r[-1]]
        A1_m_enc    = [self.A1_m_enc[-1]]
        A1_P        = [self.A1_P[-1]]
        A1_T        = [self.A1_T[-1]]
        A1_u        = [self.A1_u[-1]]
        A1_mat_id   = [mat_id_L3]
        A1_rho      = [weos._find_rho_fixed_P_T(A1_P[0], A1_T[0], mat_id_L3, 
                                                u_cold_array_L3)]
        
        if self.A1_T_rho_type[2] == 1:
            self.A1_T_rho_args[2][0] = (
                self.A1_T[0] * A1_rho[0]**(-self.A1_T_rho_args[2][1]))            
        self.A1_T_rho_args[2] = np.array(self.A1_T_rho_args[2])

        # Integrate outwards until the minimum density
        step = 1            
        while A1_rho[-1] > self.rho_min:
            A1_r.append(A1_r[-1] + dr)
            A1_m_enc.append(A1_m_enc[-1] + 4*np.pi*A1_r[-1]*A1_r[-1]*A1_rho[-1]*dr)
            A1_P.append(A1_P[-1] - G*A1_m_enc[-1]*A1_rho[-1]/(A1_r[-1]**2)*dr)
            rho = weos._find_rho(A1_P[-1], mat_id_L3, self.A1_T_rho_type[2], self.A1_T_rho_args[2],
                                 0.9*A1_rho[-1], A1_rho[-1], u_cold_array_L3)
            A1_rho.append(rho)
            A1_T.append(weos.T_rho(rho, self.A1_T_rho_type[2], self.A1_T_rho_args[2]))
            A1_u.append(weos._find_u(rho, mat_id_L3, A1_T[-1], u_cold_array_L3))
            A1_mat_id.append(mat_id_L3)   

            step += 1
            if step >= self.num_prof:
                print("Layer 3 goes out too far!")
                break
        
        # Apppend the new layer to the profiles 
        self.A1_r       = np.append(self.A1_r, A1_r[1:])
        self.A1_m_enc   = np.append(self.A1_m_enc, A1_m_enc[1:])
        self.A1_P       = np.append(self.A1_P, A1_P[1:])
        self.A1_T       = np.append(self.A1_T, A1_T[1:])
        self.A1_rho     = np.append(self.A1_rho, A1_rho[1:])
        self.A1_u       = np.append(self.A1_u, A1_u[1:])
        self.A1_mat_id  = np.append(self.A1_mat_id, A1_mat_id[1:])

        self.update_attributes()
        self.print_info()

def load_planet(name, Fp_planet):
    """ Return a new Planet object loaded from a file.
    
        Args:
            name (str)
                The name of the planet object.
                
            Fp_planet (str)
                The object data file path.
    """
    p = Planet(name=name, Fp_planet=Fp_planet)
    
    Fp_planet = check_end(p.Fp_planet, ".hdf5")
    
    print("Loading \"%s\"... " % Fp_planet[-60:], end='')
    sys.stdout.flush()
    
    with h5py.File(Fp_planet, "r") as f:
        (p.num_layer, p.A1_mat_layer, p.A1_mat_id_layer, p.A1_T_rho_type, 
         p.A1_T_rho_args, p.A1_R_layer, p.A1_M_layer, p.M, p.R, p.A1_idx_layer, 
         p.P_s, p.T_s, p.rho_s, p.A1_r, p.A1_m_enc, p.A1_rho, p.A1_T, p.A1_P, 
         p.A1_u, p.A1_mat_id
         ) = multi_get_planet_data(
            f, ["num_layer", "mat_layer", "mat_id_layer", "T_rho_type", 
                "T_rho_args", "R_layer", "M_layer", "M", "R", "idx_layer", 
                "P_s", "T_s", "rho_s", "r", "m_enc", "rho", "T", "P", "u", 
                "mat_id"]
            )
            
    print("Done")
    
    p.update_attributes()
    p.print_info()
    
    return p

# ============================================================================ #
# ===================== Spining profile functions ============================ #
# ============================================================================ #

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
        r_2 = (r_0 + r_1) * 0.5
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
                
            r_2 = (r_0 + r_1) * 0.5
            rho_2 = rho_e_model(r_2)
            R_2 = r_2
            Z_2 = rho_p_model_inv(rho_2)
            
        return rho_2
    
    return -1
    
# ===================== 1 layer ============================================== #
    
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
             mat_id_L1, T_rho_type_L1, T_rho_args_L1, u_cold_array_L1):
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
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
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
            
        if P_e[i + 1] >= P_s:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                                         rho_s - 10, rho_e[i], u_cold_array_L1) 
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                                     rho_s - 10, rho_p[i], u_cold_array_L1)
        else:
            rho_p[i + 1] = 0.
            break
        
    return rho_e, rho_p

def spin1layer(num_attempt, r_array, z_array, radii, densities, Tw,
               P_c, P_s, rho_c, rho_s,
               mat_id_L1, T_rho_type_L1, T_rho_args_L1,
               u_cold_array_L1):
    """ Compute spining profile of densities for a 1 layer planet.
    
        Args:
            
            num_attempt (int):
                Number of num_attempt to run.
                
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
            
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
        Returns:
            
            profile_e ([[float]]):
                List of the num_attempt of the equatorial density profile (SI).
                
            profile_p ([[float]]):
                List of the num_attempt of the polar density profile (SI).
    
    """
        
    spherical_model = interp1d(radii, densities, bounds_error=False, fill_value=0)

    rho_e = spherical_model(r_array)
    rho_p = spherical_model(z_array)
    
    profile_e = []
    profile_p = []
    
    profile_e.append(rho_e)
    profile_p.append(rho_p)
    
    for i in tqdm(range(num_attempt), desc="Solving spining profile"):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho1(r_array, V_e, z_array, V_p, P_c, P_s, rho_c, rho_s,
                                mat_id_L1, T_rho_type_L1, T_rho_args_L1, u_cold_array_L1)
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
def _generate_M(indices, m_enc):
    
    M = np.zeros(indices.shape)
    
    for i in range(M.shape[0]):
        M[i,:] = m_enc[indices[i]]
        
    return M

def picle_placement_L1(r_array, rho_e, z_array, rho_p, Tw, N,
                       mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                       u_cold_array_L1, N_neig=48, num_attempt=10):
    
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
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
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
                
            m_enc ([float]):
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
    
    P = np.zeros((mP.shape[0],))
    
    for k in range(mP.shape[0]):
        T = weos.T_rho(rho[k], T_rho_type_L1, T_rho_args_L1)
        u[k] = weos._find_u(rho[k], mat_id_L1, T, u_cold_array_L1)
        P[k] = weos.P_EoS(u[k], rho[k], mat_id_L1)
    
    #print("Internal energy u computed\n")
    # Smoothing lengths, crudely estimated from the densities
    w_edge  = 2     # r/h at which the kernel goes to zero
    h       = np.cbrt(N_neig * mP / (4/3*np.pi * rho)) / w_edge 
    
    A1_id     = np.arange(mP.shape[0])
    A1_mat_id = np.ones((mP.shape[0],))*mat_id_L1
    
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
        
        for _ in tqdm(range(num_attempt), desc="Tweaking mass of every particle"):
        
            M = _generate_M(indices, mP)
        
            rho_sph = SPH_density(M, distances, h)
            
            diff = (rho_sph - rho)/rho
            mP_next = (1 - diff)*mP
            mP_next[R == unique_R[-1]] = mP[R == unique_R[-1]] # do not change mass of boundary layers
            
            mP = mP_next
        
    else:
        
        k    = particles.N_picle // N_mem
        
        for _ in tqdm(range(num_attempt), desc="Tweaking mass of every particle"):
            
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

# ===================== 2 layers ============================================= #
    
@jit(nopython=True)
def _fillrho2(r_array, V_e, z_array, V_p, P_c, P_i, P_s, rho_c, rho_s,
             mat_id_L1, T_rho_type_L1, T_rho_args_L1, u_cold_array_L1,
             mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L2):
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
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between T and rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
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
            
        if P_e[i + 1] >= P_s and P_e[i + 1] >= P_i:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                                     rho_s - 10, rho_e[i], u_cold_array_L1) 
            
        elif P_e[i + 1] >= P_s and P_e[i + 1] < P_i:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_L2, T_rho_type_L2, T_rho_args_L2,
                                     rho_s - 10, rho_e[i], u_cold_array_L2) 
            
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s and P_p[i + 1] >= P_i:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                                     rho_s - 10, rho_p[i], u_cold_array_L1)
            
        elif P_p[i + 1] >= P_s and P_p[i + 1] < P_i:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_L2, T_rho_type_L2, T_rho_args_L2,
                                     rho_s - 10, rho_p[i], u_cold_array_L2)
            
        else:
            rho_p[i + 1] = 0.
            break
        
    return rho_e, rho_p

def spin2layer(num_attempt, r_array, z_array, radii, densities, Tw,
               P_c, P_i, P_s, rho_c, rho_s,
               mat_id_L1, T_rho_type_L1, T_rho_args_L1,
               mat_id_L2, T_rho_type_L2, T_rho_args_L2,
               u_cold_array_L1, u_cold_array_L2):
    """ Compute spining profile of densities for a 2 layer planet.
    
        Args:
            
            num_attempt (int):
                Number of num_attempt to run.
                
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
            
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between T and rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
        Returns:
            
            profile_e ([[float]]):
                List of the num_attempt of the equatorial density profile (SI).
                
            profile_p ([[float]]):
                List of the num_attempt of the polar density profile (SI).
    
    """    
    
    spherical_model = interp1d(radii, densities, bounds_error=False, fill_value=0)

    rho_e = spherical_model(r_array)
    rho_p = spherical_model(z_array)
    
    profile_e = []
    profile_p = []
    
    profile_e.append(rho_e)
    profile_p.append(rho_p)
    
    for i in tqdm(range(num_attempt), desc="Solving spining profile"):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho2(r_array, V_e, z_array, V_p, P_c, P_i, P_s, rho_c, rho_s,
                                mat_id_L1, T_rho_type_L1, T_rho_args_L1, u_cold_array_L1,
                                mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L2)
        profile_e.append(rho_e)
        profile_p.append(rho_p)
    
    return profile_e, profile_p

def picle_placement_L2(r_array, rho_e, z_array, rho_p, Tw, N, rho_i,
                           mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                           mat_id_L2, T_rho_type_L2, T_rho_args_L2,
                           u_cold_array_L1, u_cold_array_L2, N_neig=48,
                           num_attempt=10):
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
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between T and rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
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
    
    P = np.zeros((mP.shape[0],))
    
    for k in range(mP.shape[0]):
        if rho[k] > rho_i:
            T = weos.T_rho(rho[k], T_rho_type_L1, T_rho_args_L1)
            u[k] = weos._find_u(rho[k], mat_id_L1, T, u_cold_array_L1)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_L1)
        else:
            T = weos.T_rho(rho[k], T_rho_type_L2, T_rho_args_L2)
            u[k] = weos._find_u(rho[k], mat_id_L2, T, u_cold_array_L2)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_L2)
    
    #print("Internal energy u computed\n")
    
    # Smoothing lengths, crudely estimated from the densities
    num_ngb = N_neig    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = (rho > rho_i)*mat_id_L1 + (rho <= rho_i)*mat_id_L2
    
    unique_R_L1   = np.unique(R[A1_mat_id == mat_id_L1])
    unique_R_L2 = np.unique(R[A1_mat_id == mat_id_L2])
    
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
        
        for _ in tqdm(range(num_attempt), desc="Tweaking mass of every particle"):
        
            M = _generate_M(indices, mP)
        
            rho_sph = SPH_density(M, distances, h)
            
            diff = (rho_sph - rho)/rho
            mP_next = (1 - diff)*mP
            # do not change values of inter-boundary layers
            mP_next[R == unique_R_L1[-1]]   = mP[R == unique_R_L1[-1]]  # outer layer 1
            mP_next[R == unique_R_L2[0]]    = mP[R == unique_R_L2[0]]   # inner layer 2
            mP_next[R == unique_R_L2[-1]]   = mP[R == unique_R_L2[-1]]  # outer layer 2
            
            mP = mP_next
        
    else:
        
        k    = particles.N_picle // N_mem
        
        for _ in tqdm(range(num_attempt), desc="Tweaking mass of every particle"):
            
            mP_prev = mP.copy()
            
            for i in range(int(k)):
                
                distances_i, indices_i = nbrs.kneighbors(X[i*N_mem:(i + 1)*N_mem,:])
                
                M_i  = _generate_M(indices_i, mP_prev)
        
                rho_sph_i = SPH_density(M_i, distances_i, h[i*N_mem:(i + 1)*N_mem])
                
                diff_i = (rho_sph_i - rho[i*N_mem:(i + 1)*N_mem])/rho[i*N_mem:(i + 1)*N_mem]
                mP_next_i = (1 - diff_i)*mP[i*N_mem:(i + 1)*N_mem]
                # do not change values of inter-boundary layers
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L1[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L1[-1]]  # outer layer 1
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[0]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[0]]   # inner layer 2
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[-1]]  # outer layer 2
            
                mP[i*N_mem:(i + 1)*N_mem] = mP_next_i
                
            distances_k, indices_k = nbrs.kneighbors(X[k*N_mem:,:])
                
            M_k  = _generate_M(indices_k, mP_prev)
        
            rho_sph_k = SPH_density(M_k, distances_k, h[k*N_mem:])
                
            diff_k = (rho_sph_k - rho[k*N_mem:])/rho[k*N_mem:]
            mP_next_k = (1 - diff_k)*mP[k*N_mem:]
            # do not change values of inter-boundary layers
            mP_next_k[R[k*N_mem:] == unique_R_L1[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L1[-1]]    # outer layer 1
            mP_next_k[R[k*N_mem:] == unique_R_L2[0]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L2[0]]     # inner layer 2
            mP_next_k[R[k*N_mem:] == unique_R_L2[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L2[-1]]    # outer layer 2
            
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

# ===================== 3 layers ============================================= #
    
@jit(nopython=True)
def _fillrho3(r_array, V_e, z_array, V_p, P_c, P_12, P_23, P_s, rho_c, rho_s,
             mat_id_L1, T_rho_type_L1, T_rho_args_L1, u_cold_array_L1,
             mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L2,
             mat_id_L3, T_rho_type_L3, T_rho_args_L3, u_cold_array_L3):
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
                
            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between T and rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
            mat_id_L3 (int):
                Material id for layer 3.
                
            T_rho_type_L3 (int)
                Relation between T and rho to be used in layer 3.
                
            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            u_cold_array_L3 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 3 (SI).
                
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
            
        if P_e[i + 1] >= P_s and P_e[i + 1] >= P_12:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                                     rho_s - 10, rho_e[i], u_cold_array_L1) 
            
        elif P_e[i + 1] >= P_s and P_e[i + 1] >= P_23:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_L2, T_rho_type_L2, T_rho_args_L2,
                                     rho_s - 10, rho_e[i], u_cold_array_L2) 
            
        elif P_e[i + 1] >= P_s:
            rho_e[i + 1] = weos._find_rho(P_e[i + 1], mat_id_L3, T_rho_type_L3, T_rho_args_L3,
                                     rho_s - 10, rho_e[i], u_cold_array_L3)
            
        else:
            rho_e[i + 1] = 0.
            break
        
    for i in range(z_array.shape[0] - 1):
        gradV = V_p[i + 1] - V_p[i]
        gradP = -rho_p[i]*gradV
        P_p[i + 1] = P_p[i] + gradP
        
        if P_p[i + 1] >= P_s and P_p[i + 1] >= P_12:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                                     rho_s - 10, rho_p[i], u_cold_array_L1)
            
        elif P_p[i + 1] >= P_s and P_p[i + 1] >= P_23:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_L2, T_rho_type_L2, T_rho_args_L2,
                                     rho_s - 10, rho_p[i], u_cold_array_L2)
            
        elif P_p[i + 1] >= P_s:
            rho_p[i + 1] = weos._find_rho(P_p[i + 1], mat_id_L3, T_rho_type_L3, T_rho_args_L3,
                                     rho_s - 10, rho_p[i], u_cold_array_L3)
            
        else:
            rho_p[i + 1] = 0.
            break
        
    return rho_e, rho_p

def spin3layer(num_attempt, r_array, z_array, radii, densities, Tw,
               P_c, P_12, P_23, P_s, rho_c, rho_s,
               mat_id_L1, T_rho_type_L1, T_rho_args_L1,
               mat_id_L2, T_rho_type_L2, T_rho_args_L2,
               mat_id_L3, T_rho_type_L3, T_rho_args_L3,
               u_cold_array_L1, u_cold_array_L2, u_cold_array_L3):
    """ Compute spining profile of densities for a 3 layer planet.
    
        Args:
            
            num_attempt (int):
                Number of num_attempt to run.
                
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
                
            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between T and rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.
                
            T_rho_type_L3 (int)
                Relation between T and rho to be used in layer 3.
                
            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
            u_cold_array_L3 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 3 (SI).
                
        Returns:
            
            profile_e ([[float]]):
                List of the num_attempt of the equatorial density profile (SI).
                
            profile_p ([[float]]):
                List of the num_attempt of the polar density profile (SI).
    
    """    
    spherical_model = interp1d(radii, densities, bounds_error=False, fill_value=0)

    rho_e = spherical_model(r_array)
    rho_p = spherical_model(z_array)
    
    profile_e = []
    profile_p = []
    
    profile_e.append(rho_e)
    profile_p.append(rho_p)
    
    for i in tqdm(range(num_attempt), desc="Solving spining profile"):
        V_e, V_p = _fillV(r_array, rho_e, z_array, rho_p, Tw)
        rho_e, rho_p = _fillrho3(r_array, V_e, z_array, V_p, P_c, P_12, P_23, P_s, rho_c, rho_s,
                                mat_id_L1, T_rho_type_L1, T_rho_args_L1, u_cold_array_L1,
                                mat_id_L2, T_rho_type_L2, T_rho_args_L2, u_cold_array_L2,
                                mat_id_L3, T_rho_type_L3, T_rho_args_L3, u_cold_array_L3)
        profile_e.append(rho_e)
        profile_p.append(rho_p)
    
    return profile_e, profile_p

def picle_placement_L3(r_array, rho_e, z_array, rho_p, Tw, N, rho_12, rho_23,
                           mat_id_L1, T_rho_type_L1, T_rho_args_L1,
                           mat_id_L2, T_rho_type_L2, T_rho_args_L2,
                           mat_id_L3, T_rho_type_L3, T_rho_args_L3,
                           u_cold_array_L1, u_cold_array_L2, u_cold_array_L3,
                           N_neig=48, num_attempt=10):
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
                
            rho_12 (float):
                Density at the boundary between layers 1 and 2 (SI).
                
            rho_23 (float):
                Density at the boundary between layers 2 and 3 (SI).
                
            mat_id_L1 (int):
                Material id for layer 1.
                
            T_rho_type_L1 (int)
                Relation between T and rho to be used in layer 1.
                
            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.
                
            mat_id_L2 (int):
                Material id for layer 2.
                
            T_rho_type_L2 (int)
                Relation between T and rho to be used in layer 2.
                
            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.
                
            mat_id_L3 (int):
                Material id for layer 3.
                
            T_rho_type_L3 (int)
                Relation between T and rho to be used in layer 3.
                
            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            u_cold_array_L1 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 1 (SI).
                
            u_cold_array_L2 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 2 (SI).
                
            u_cold_array_L3 ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() for layer 3 (SI).
                
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
    
    P = np.zeros((mP.shape[0],))
    
    for k in range(mP.shape[0]):
        if rho[k] > rho_12:
            T = weos.T_rho(rho[k], T_rho_type_L1, T_rho_args_L1)
            u[k] = weos._find_u(rho[k], mat_id_L1, T, u_cold_array_L1)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_L1)
            
        elif rho[k] > rho_23:
            T = weos.T_rho(rho[k], T_rho_type_L2, T_rho_args_L2)
            u[k] = weos._find_u(rho[k], mat_id_L2, T, u_cold_array_L2)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_L2)
            
        else:
            T = weos.T_rho(rho[k], T_rho_type_L3, T_rho_args_L3)
            u[k] = weos._find_u(rho[k], mat_id_L3, T, u_cold_array_L3)
            P[k] = weos.P_EoS(u[k], rho[k], mat_id_L3)
    
    # Smoothing lengths, crudely estimated from the densities
    num_ngb = N_neig    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = (rho > rho_12)*mat_id_L1                       \
                + np.logical_and(rho <= rho_12, rho > rho_23)*mat_id_L2 \
                + (rho < rho_23)*mat_id_L3
    
    unique_R_L1   = np.unique(R[A1_mat_id == mat_id_L1])
    unique_R_L2 = np.unique(R[A1_mat_id == mat_id_L2])
    unique_R_L3    = np.unique(R[A1_mat_id == mat_id_L3])
    
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
        
        for _ in tqdm(range(num_attempt), desc="Tweaking mass of every particle"):
        
            M = _generate_M(indices, mP)
        
            rho_sph = SPH_density(M, distances, h)
            
            diff = (rho_sph - rho)/rho
            mP_next = (1 - diff)*mP
            # do not change values of inter-boundary layers
            mP_next[R == unique_R_L1[-1]]   = mP[R == unique_R_L1[-1]]  # outer layer 1
            mP_next[R == unique_R_L2[0]]  = mP[R == unique_R_L2[0]]     # inner layer 2
            mP_next[R == unique_R_L2[-1]] = mP[R == unique_R_L2[-1]]    # outer layer 2
            mP_next[R == unique_R_L3[0]]  = mP[R == unique_R_L3[0]]     # inner layer 3
            mP_next[R == unique_R_L3[-1]] = mP[R == unique_R_L3[-1]]    # outer layer 3
            
            mP = mP_next
        
    else:
        
        k    = particles.N_picle // N_mem
        
        for _ in tqdm(range(num_attempt), desc="Tweaking mass of every particle"):
            
            mP_prev = mP.copy()
            
            for i in range(int(k)):
                
                distances_i, indices_i = nbrs.kneighbors(X[i*N_mem:(i + 1)*N_mem,:])
                
                M_i  = _generate_M(indices_i, mP_prev)
        
                rho_sph_i = SPH_density(M_i, distances_i, h[i*N_mem:(i + 1)*N_mem])
                
                diff_i = (rho_sph_i - rho[i*N_mem:(i + 1)*N_mem])/rho[i*N_mem:(i + 1)*N_mem]
                mP_next_i = (1 - diff_i)*mP[i*N_mem:(i + 1)*N_mem]
                # do not change values of inter-boundary layers
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L1[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L1[-1]]  # outer layer 1
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[0]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[0]]   # inner layer 2
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L2[-1]]  # outer layer 2
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L3[0]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L3[0]]   # inner layer 3
                mP_next_i[R[i*N_mem:(i + 1)*N_mem] == unique_R_L3[-1]] = \
                    mP[i*N_mem:(i + 1)*N_mem][R[i*N_mem:(i + 1)*N_mem] == unique_R_L3[-1]]  # outer layer 3
            
                mP[i*N_mem:(i + 1)*N_mem] = mP_next_i
                
            distances_k, indices_k = nbrs.kneighbors(X[k*N_mem:,:])
                
            M_k  = _generate_M(indices_k, mP_prev)
        
            rho_sph_k = SPH_density(M_k, distances_k, h[k*N_mem:])
                
            diff_k = (rho_sph_k - rho[k*N_mem:])/rho[k*N_mem:]
            mP_next_k = (1 - diff_k)*mP[k*N_mem:]
            # do not change values of inter-boundary layers
            mP_next_k[R[k*N_mem:] == unique_R_L1[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L1[-1]]    # outer layer 1
            mP_next_k[R[k*N_mem:] == unique_R_L2[0]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L2[0]]     # inner layer 2
            mP_next_k[R[k*N_mem:] == unique_R_L2[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L2[-1]]    # outer layer 2
            mP_next_k[R[k*N_mem:] == unique_R_L3[0]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L3[0]]     # inner layer 3
            mP_next_k[R[k*N_mem:] == unique_R_L3[-1]] = \
                mP[k*N_mem:][R[k*N_mem:] == unique_R_L3[-1]]    # outer layer 3
            
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

# ============================================================================ #
# ===================== Spining profile classes ============================== #
# ============================================================================ #
class SpinPlanet():
    
    def __init__(
    self, name=None, planet=None, Fp_planet=None, Tw=None,
    A1_mat_layer=None, A1_R_layer=None,
    A1_T_rho_type=None, A1_T_rho_args=None, A1_r=None, A1_P=None, A1_T=None,
    A1_rho=None, num_prof=1000, num_attempt=15, R_e=None, R_p=None
    ):
        
        self.name               = name
        self.Fp_planet          = Fp_planet
        self.num_prof           = num_prof
        self.num_attempt        = num_attempt
        self.R_e                = R_e
        self.R_p                = R_p
        self.Tw                 = Tw
        self.P_R1               = None
        self.P_R2               = None
        
        if planet is not None:
            self.num_layer       = planet.num_layer
            self.A1_mat_layer    = planet.A1_mat_layer
            self.A1_R_layer      = planet.A1_R_layer
            self.A1_mat_id_layer = planet.A1_mat_id_layer
            self.A1_T_rho_type   = planet.A1_T_rho_type
            self.A1_T_rho_args   = planet.A1_T_rho_args
            self.A1_r            = planet.A1_r
            self.A1_P            = planet.A1_P
            self.A1_T            = planet.A1_T
            self.A1_rho          = planet.A1_rho
            
        else:
            self.A1_R_layer      = A1_R_layer
            self.A1_mat_layer    = A1_mat_layer
            self.A1_T_rho_type   = A1_T_rho_type
            self.A1_T_rho_args   = A1_T_rho_args
            self.A1_r            = A1_r
            self.A1_P            = A1_P
            self.A1_T            = A1_T
            self.A1_rho          = A1_rho
            
            # Derived or default attributes
            if self.A1_mat_layer is not None:
                self.num_layer          = len(self.A1_mat_layer)
                self.A1_mat_id_layer    = [weos.Di_mat_id[mat] 
                                           for mat in self.A1_mat_layer]
            
        assert(self.num_layer in [1, 2, 3])
        
    def spin(self):
        # Check for necessary input
        assert(self.R_e is not None)
        assert(self.R_p is not None)
        assert(self.Tw is not None)
        
        P_c   = np.max(self.A1_P)
        P_s   = np.min(self.A1_P)
        rho_c = np.max(self.A1_rho)
        rho_s = np.min(self.A1_rho)
        
        r_array     = np.linspace(0, self.R_e, self.num_prof)
        z_array     = np.linspace(0, self.R_p, self.num_prof)
        
        if self.num_layer == 1:
            # Check for necessary input
            assert(self.A1_mat_id_layer[0] is not None)
            assert(self.A1_T_rho_type[0] is not None)
            
            u_cold_array = weos.load_u_cold_array(self.A1_mat_id_layer[0]) 
        
            profile_e, profile_p = \
                spin1layer(self.num_attempt, r_array, z_array,
                           self.A1_r, self.A1_rho, self.Tw,
                           P_c, P_s,
                           rho_c, rho_s,
                           self.A1_mat_id_layer[0], self.A1_T_rho_type[0], self.A1_T_rho_args[0],
                           u_cold_array)
            
            print("\nDone!")
        
            self.A1_r_equator   = r_array
            self.A1_r_pole      = z_array
            self.A1_rho_equator = profile_e[-1]
            self.A1_rho_pole    = profile_p[-1]
            
        elif self.num_layer == 2:
            # Check for necessary input
            assert(self.A1_mat_id_layer[0] is not None)
            assert(self.A1_T_rho_type[0] is not None)
            assert(self.A1_mat_id_layer[1] is not None)
            assert(self.A1_T_rho_type[1] is not None)
            
            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[0]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[0]])
            P_boundary = 0.5*(a + b)
            
            self.P_R1 = P_boundary
            
            u_cold_array_L1   = weos.load_u_cold_array(self.A1_mat_id_layer[0]) 
            u_cold_array_L2   = weos.load_u_cold_array(self.A1_mat_id_layer[1]) 
            
            profile_e, profile_p = \
                spin2layer(self.num_attempt, r_array, z_array,
                           self.A1_r, self.A1_rho, self.Tw,
                           P_c, P_boundary, P_s,
                           rho_c, rho_s,
                           self.A1_mat_id_layer[0], self.A1_T_rho_type[0], self.A1_T_rho_args[0],
                           self.A1_mat_id_layer[1], self.A1_T_rho_type[1], self.A1_T_rho_args[1],
                           u_cold_array_L1, u_cold_array_L2)
                
            print("\nDone!")
            
            self.A1_r_equator     = r_array
            self.A1_r_pole        = z_array
            self.A1_rho_equator   = profile_e[-1]
            self.A1_rho_pole      = profile_p[-1]
            
            
        elif self.num_layer == 3:
            # Check for necessary input
            assert(self.A1_mat_id_layer[0] is not None)
            assert(self.A1_T_rho_type[0] is not None)
            assert(self.A1_mat_id_layer[1] is not None)
            assert(self.A1_T_rho_type[1] is not None)
            assert(self.A1_mat_id_layer[2] is not None)
            assert(self.A1_T_rho_type[2] is not None)
            
            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[0]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[0]])
            P_boundary_12 = 0.5*(a + b)
            
            self.P_R1 = P_boundary_12
            
            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[1]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[1]])
            P_boundary_ma = 0.5*(a + b)
            
            self.P_R2 = P_boundary_ma
        
            u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0]) 
            u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
            u_cold_array_L3 = weos.load_u_cold_array(self.A1_mat_id_layer[2]) 
            
            profile_e, profile_p = \
                spin3layer(self.num_attempt, r_array, z_array,
                           self.A1_r, self.A1_rho, self.Tw,
                           P_c, P_boundary_12, P_boundary_ma, P_s,
                           rho_c, rho_s,
                           self.A1_mat_id_layer[0], self.A1_T_rho_type[0], self.A1_T_rho_args[0],
                           self.A1_mat_id_layer[1], self.A1_T_rho_type[1], self.A1_T_rho_args[1],
                           self.A1_mat_id_layer[2], self.A1_T_rho_type[2], self.A1_T_rho_args[2],
                           u_cold_array_L1, u_cold_array_L2, u_cold_array_L3)
                
            print("\nDone!")
            
            self.A1_r_equator     = r_array
            self.A1_r_pole        = z_array
            self.A1_rho_equator   = profile_e[-1]
            self.A1_rho_pole      = profile_p[-1]
            
class GenSpheroid():
    
    def __init__(
    self, name=None, spin_planet=None, Fp_planet=None, Tw=None,
    A1_mat_layer=None, A1_T_rho_args=None, A1_T_rho_type=None, 
    A1_r_equator=None, A1_rho_equator=None, 
    A1_r_pole=None, A1_rho_pole=None, P_R1=None, P_R2=None,
    N_particles = None, N_neig=48, num_attempt=10,
    A1_r=None, A1_rho=None, A1_P=None
    ):
        self.name               = name
        self.Fp_planet          = Fp_planet
        self.N_particles        = N_particles
        self.num_attempt        = num_attempt
        
        if spin_planet is not None:
            self.num_layer       = spin_planet.num_layer
            self.A1_mat_layer    = spin_planet.A1_mat_layer
            self.A1_mat_id_layer = spin_planet.A1_mat_id_layer
            self.A1_T_rho_type   = spin_planet.A1_T_rho_type
            self.A1_T_rho_args   = spin_planet.A1_T_rho_args
            self.A1_r_equator    = spin_planet.A1_r_equator
            self.A1_rho_equator  = spin_planet.A1_rho_equator
            self.A1_r_pole       = spin_planet.A1_r_pole
            self.A1_rho_pole     = spin_planet.A1_rho_pole
            self.Tw              = spin_planet.Tw
            self.P_R1            = spin_planet.P_R1
            self.P_R2            = spin_planet.P_R2
            self.A1_r            = spin_planet.A1_r
            self.A1_rho          = spin_planet.A1_rho
            self.A1_P            = spin_planet.A1_P
            
        else:
            self.A1_mat_layer    = A1_mat_layer
            self.A1_T_rho_type   = A1_T_rho_type
            self.A1_T_rho_args   = A1_T_rho_args
            self.A1_r_equator    = A1_r_equator
            self.A1_rho_equator  = A1_rho_equator
            self.A1_r_pole       = A1_r_pole
            self.A1_rho_pole     = A1_rho_pole
            self.Tw              = Tw
            self.P_R1            = P_R1
            self.P_R2            = P_R2
            self.A1_r            = A1_r
            self.A1_rho          = A1_rho
            self.A1_P            = A1_P
            
            # Derived or default attributes
            if self.A1_mat_layer is not None:
                self.num_layer          = len(self.A1_mat_layer)
                self.A1_mat_id_layer    = [weos.Di_mat_id[mat] 
                                           for mat in self.A1_mat_layer]
            
        assert(self.num_layer in [1, 2, 3])
        assert(self.N_particles is not None)
        
        if self.num_layer == 1:
            
            u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
            
            x, y, z, vx, vy, vz, m, h, rho, P, u, mat_id, picle_id = \
            picle_placement_L1(self.A1_r_equator, self.A1_rho_equator,
                                   self.A1_r_pole, self.A1_rho_pole, self.Tw, self.N_particles,
                                   self.A1_mat_id_layer[0], self.A1_T_rho_type[0], self.A1_T_rho_args[0],
                                   u_cold_array_L1, N_neig, num_attempt)
            
            self.A1_picle_x      = x
            self.A1_picle_y      = y
            self.A1_picle_z      = z
            self.A1_picle_vx     = vx
            self.A1_picle_vy     = vy
            self.A1_picle_vz     = vz
            self.A1_picle_m      = m
            self.A1_picle_h      = h
            self.A1_picle_rho    = rho
            self.A1_picle_P      = P
            self.A1_picle_u      = u
            self.A1_picle_mat_id = mat_id
            self.A1_picle_id     = picle_id
            self.N_particles     = x.shape[0]
            
        elif self.num_layer == 2:
            
            rho_P_model       = interp1d(self.A1_P, self.A1_rho)
            self.rho_R1       = rho_P_model(self.P_R1)
            
            u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
            u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
            
            x, y, z, vx, vy, vz, m, h, rho, P, u, mat_id, picle_id = \
                picle_placement_L2(self.A1_r_equator, self.A1_rho_equator,
                                       self.A1_r_pole, self.A1_rho_pole,
                                       self.Tw, self.N_particles, self.rho_R1,
                                       self.A1_mat_id_layer[0], self.A1_T_rho_type[0], self.A1_T_rho_args[0],
                                       self.A1_mat_id_layer[1], self.A1_T_rho_type[1], self.A1_T_rho_args[1],
                                       u_cold_array_L1, u_cold_array_L2,
                                       N_neig, num_attempt)
                
            self.A1_picle_x      = x
            self.A1_picle_y      = y
            self.A1_picle_z      = z
            self.A1_picle_vx     = vx
            self.A1_picle_vy     = vy
            self.A1_picle_vz     = vz
            self.A1_picle_m      = m
            self.A1_picle_h      = h
            self.A1_picle_rho    = rho
            self.A1_picle_P      = P
            self.A1_picle_u      = u
            self.A1_picle_mat_id = mat_id
            self.A1_picle_id     = picle_id
            self.N_particles     = x.shape[0]
            
        elif self.num_layer == 3:
            
            rho_P_model  = interp1d(self.A1_P, self.A1_rho)
            self.rho_R1  = rho_P_model(self.P_R1)
            self.rho_R2  = rho_P_model(self.P_R2)
            
            u_cold_array_L1 = weos.load_u_cold_array(self.A1_mat_id_layer[0])
            u_cold_array_L2 = weos.load_u_cold_array(self.A1_mat_id_layer[1])
            u_cold_array_L3 = weos.load_u_cold_array(self.A1_mat_id_layer[2])
            
            x, y, z, vx, vy, vz, m, h, rho, P, u, mat_id, picle_id = \
                picle_placement_L3(self.A1_r_equator, self.A1_rho_equator,
                                       self.A1_r_pole, self.A1_rho_pole,
                                       self.Tw, self.N_particles, self.rho_R1, self.rho_R2,
                                       self.A1_mat_id_layer[0], self.A1_T_rho_type[0], self.A1_T_rho_args[0],
                                       self.A1_mat_id_layer[1], self.A1_T_rho_type[1], self.A1_T_rho_args[1],
                                       self.A1_mat_id_layer[2], self.A1_T_rho_type[2], self.A1_T_rho_args[2],
                                       u_cold_array_L1, u_cold_array_L2, u_cold_array_L3,
                                       N_neig, num_attempt)
                
            self.A1_picle_x      = x
            self.A1_picle_y      = y
            self.A1_picle_z      = z
            self.A1_picle_vx     = vx
            self.A1_picle_vy     = vy
            self.A1_picle_vz     = vz
            self.A1_picle_m      = m
            self.A1_picle_h      = h
            self.A1_picle_rho    = rho
            self.A1_picle_P      = P
            self.A1_picle_u      = u
            self.A1_picle_mat_id = mat_id
            self.A1_picle_id     = picle_id
            self.N_particles     = x.shape[0]
        
# Set up equation of state data
set_up()
