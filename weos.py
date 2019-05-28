#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:38:18 2019

@author: Sergio Ruiz-Bonilla
"""
###############################################################################
####################### Libraries and constants ###############################
###############################################################################

import numpy as np
from numba import jit

G       = 6.67408E-11;
R_earth = 6371000;
M_earth = 5.972E24;  
R_gas   = 8.3145     # Gas constant (J K^-1 mol^-1)

# Material IDs, same as SWIFT ( = type_id * type_factor + unit_id)
type_factor = 100
Di_mat_type = {
    "idg"       : 0,
    "Til"       : 1,
    "HM80"      : 2,
    "SESAME"    : 3,
    }
Di_mat_id   = {
    # Ideal Gas
    "idg_HHe"       : Di_mat_type["idg"]*type_factor,
    "idg_N2"        : Di_mat_type["idg"]*type_factor + 1,
    # Tillotson
    "Til_iron"      : Di_mat_type["Til"]*type_factor,
    "Til_granite"   : Di_mat_type["Til"]*type_factor + 1,
    "Til_water"     : Di_mat_type["Til"]*type_factor + 2,
    # Hubbard & MacFarlane (1980) Uranus/Neptune
    "HM80_HHe"      : Di_mat_type["HM80"]*type_factor,      # Hydrogen-helium atmosphere
    "HM80_ice"      : Di_mat_type["HM80"]*type_factor + 1,  # H20-CH4-NH3 ice mix
    "HM80_rock"     : Di_mat_type["HM80"]*type_factor + 2,  # SiO2-MgO-FeS-FeO rock mix
    # SESAME
    "SESAME_iron"   : Di_mat_type["SESAME"]*type_factor,        # 2140
    "SESAME_basalt" : Di_mat_type["SESAME"]*type_factor + 1,    # 7530
    "SESAME_water"  : Di_mat_type["SESAME"]*type_factor + 2,    # 7154
    "SS08_water"    : Di_mat_type["SESAME"]*type_factor + 3,    # Senft & Stewart (2008)
    }
# Separate variables because numba can't handle dictionaries
id_idg_HHe          = Di_mat_id["idg_HHe"]
id_idg_N2           = Di_mat_id["idg_N2"]
id_Til_iron         = Di_mat_id["Til_iron"]
id_Til_granite      = Di_mat_id["Til_granite"]
id_Til_water        = Di_mat_id["Til_water"]
id_HM80_HHe         = Di_mat_id["HM80_HHe"]
id_HM80_ice         = Di_mat_id["HM80_ice"]
id_HM80_rock        = Di_mat_id["HM80_rock"]
id_SESAME_iron      = Di_mat_id["SESAME_iron"]
id_SESAME_basalt    = Di_mat_id["SESAME_basalt"]
id_SESAME_water     = Di_mat_id["SESAME_water"]
id_SS08_water       = Di_mat_id["SS08_water"]


###############################################################################
############################### Functions #####################################
###############################################################################

@jit(nopython=True)
def _P_EoS_Till(u, rho, mat_id):
    """ Computes pressure for Tillotson EoS.
    
        Args:
            u (double)
                Internal energy (SI).
                
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
    iron    = np.array([id_Til_iron, 7800, 0.5, 1.5, 1.28e11, 1.05e11, 9.5e9, 2.4e9, 8.67e9, 5, 5, 0, 0, 0])
    granite = np.array([id_Til_granite, 2680, 0.5, 1.3, 1.8e10, 1.8e10, 1.6e10, 3.5e9, 1.8e10, 5, 5, 0, 0, 0])
    water   = np.array([id_Til_water, 998, 0.7, 0.15, 2.18e9, 1.325e10, 7.0e9, 4.19e8, 2.69e9, 10, 5, 0.925, 0, 0.875])
    
    if (mat_id == id_Til_iron):
        material = iron
    elif (mat_id == id_Til_granite):
        material = granite
    elif (mat_id == id_Til_water):
        material = water
    else:
        print("Material not implemented")
        return None
        
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

@jit(nopython=True)
def _P_EoS_idg(u, rho, mat_id):
    """ Computes pressure for the ideal gas EoS.
    
        Args:
            u (double)
                Internal energy (SI).
                
            rho (double) 
                Density (SI).
            
            mat_id (int)
                Material id.
                
        Returns:
            P (double)
                Pressure (SI).
    """
    # Material constants for ideal gas EoS
    # mat_id, gamma
    HHe = np.array([id_idg_HHe, 1.4])
    N2  = np.array([id_idg_N2, 1.4])
    
    if (mat_id == id_idg_HHe):
        material = HHe
    elif (mat_id == id_idg_N2):
        material = N2
    else:
        print("Material not implemented")
        return None
        
    gamma    = material[1]

    P = (gamma - 1)*u*rho
        
    return P

@jit(nopython=True)
def P_EoS(u, rho, mat_id):
    """ Computes pressure for a given material.
    
        Args:
            u (double)
                Internal energy (SI).
                
            rho (double) 
                Density (SI).
            
            mat_id (int)
                Material id.
                
        Returns:
            P (double)
                Pressure (SI).
    """
    if (mat_id == id_idg_HHe):
        return _P_EoS_idg(u, rho, mat_id)
    elif (mat_id == id_idg_N2):
        return _P_EoS_idg(u, rho, mat_id)
    elif (mat_id == id_Til_iron):
        return _P_EoS_Till(u, rho, mat_id)
    elif (mat_id == id_Til_granite):
        return _P_EoS_Till(u, rho, mat_id)
    elif (mat_id == id_Til_water):
        return _P_EoS_Till(u, rho, mat_id)
    else:
        print("Material not implemented")
        return None

@jit(nopython=True)
def _rho_0_material(mat_id):
    """ Returns rho0 for a given material id. u_{cold}(rho0) = 0
    
        Args:          
            mat_id (int)
                Material id.
                
        Returns:
            rho0 (double)
                Density (SI).
    
    """
    if (mat_id in [id_idg_HHe, id_idg_N2]):
        return 0
    elif (mat_id == id_Til_iron):
        return 7800.
    elif (mat_id == id_Til_granite):
        return 2680.
    elif (mat_id == id_Til_water):
        return 998.
    elif (mat_id == id_HM80_HHe):
        return None
    elif (mat_id == id_HM80_ice):
        return None
    elif (mat_id == id_HM80_rock):
        return None
    elif (mat_id == id_SESAME_iron):
        return 7902.
    elif (mat_id == id_SESAME_basalt):
        return 2902.
    elif (mat_id == id_SESAME_water):
        return 1402.
    elif (mat_id == id_SS08_water):
        return 1002.
    else:
        print("Material not implemented")
        return None
    
@jit(nopython=True)
def _C_V(mat_id):
    """ Returns specific heat capacity for a given material id (SI)
    
        Args:          
            mat_id (int)
                Material id.
                
        Returns:
            C_V (double)
                Specific heat capacity (SI).
    
    """
    if mat_id == id_idg_HHe:
        return 9093.98
    elif mat_id == id_idg_N2:
        return 742.36
    elif (mat_id == id_Til_iron):
        return 449
    elif (mat_id == id_Til_granite):
        return 790
    elif (mat_id == id_Til_water):
        return 4186
    elif (mat_id == id_HM80_HHe):
        return None
    elif (mat_id == id_HM80_ice):
        return None
    elif (mat_id == id_HM80_rock):
        return None
    elif (mat_id == id_SESAME_iron):
        return 449
    elif (mat_id == id_SESAME_basalt):
        return 790
    elif (mat_id == id_SESAME_water):
        return 4186
    elif (mat_id == id_SS08_water):
        return 4186
    else:
        print("Material not implemented")
        return None
    
@jit(nopython=True)
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
                Cold internal energy (SI).
    """
    if mat_id in [id_idg_HHe, id_idg_N2]:
        return 0
    
    elif mat_id in [id_Til_iron, id_Til_granite, id_Til_water]:
        
        rho0 = _rho_0_material(mat_id)
        drho = (rho - rho0)/N
        x = rho0
        u_cold = 1e-9
    
        for j in range(N):
            x += drho
            u_cold += P_EoS(u_cold, x, mat_id)*drho/x**2

    return u_cold

@jit(nopython=True)
def T_rho(rho, T_rho_id, T_rho_args):
    """ Computes temperature given density (T = f(rho)).
    
        Args:
            rho (float)
                Density (SI).
                
            T_rho_id (int)
                Relation between T and rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
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
    """ Computes pressure using Tillotson EoS, and
        internal energy = internal energy cold + C_V*Temperature 
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
    C_V = _C_V(mat_id)
    u = u_cold(rho, mat_id, N) + C_V*T_rho(rho, T_rho_id, T_rho_args)
    P = P_EoS(u, rho, mat_id)

    return P

@jit(nopython=True)
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
    
    rho = rho_min
    for i in range(N_row):
        u_cold_array[i] = u_cold(rho, mat_id, N_u_cold)
        rho = rho + drho
    
    return u_cold_array

@jit(nopython=True)
def _u_cold_tab(rho, mat_id, u_cold_array):
    """ Fast computation of cold internal energy using the table previously
        computed.
    
        Args:
            rho (float):
                Density (SI).
                
            mat_id (int):
                Material id.
                
            u_cold_array ([float])
                Precomputed values of cold internal energy for a particular material
                with function _create_u_cold_array() (SI).
                
        Returns:
            interpolation (float):
                cold internal energy (SI).
    """
    mat_id = int(mat_id)
    
    if mat_id in [id_idg_HHe, id_idg_N2]:
        
        return 0.
    
    elif mat_id in [id_Til_iron, id_Til_granite, id_Til_water]:
        
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
    
    else:
        
        print("Material not implemented")

@jit(nopython=True)
def _find_rho(P_s, mat_id, T_rho_id, T_rho_args, rho0, rho1, u_cold_array):
    """ Root finder of the density for EoS using 
        tabulated values of cold internal energy
        
        Args:
            P_s (float):
                Pressure (SI).
                
            mat_id (int):
                Material id.
            
            T_rho_id (int)
                Relation between T and rho to be used.
                
            T_rho_args (list):
                Extra arguments to determine the relation.
                
            rho0 (float):
                Lower bound for where to look the root (SI).
                
            rho1 (float):
                Upper bound for where to look the root (SI).
            
            u_cold_array ([float])
                Precomputed values of cold internal energy
                with function _create_u_cold_array() (SI).
                
        Returns:
            rho2 (float):
                Value of the density which satisfies P(u(rho), rho) = 0 
                (SI).
    """

    C_V       = _C_V(mat_id)
    tolerance = 1E-5
    
    u0   = _u_cold_tab(rho0, mat_id, u_cold_array) + C_V*T_rho(rho0, T_rho_id, T_rho_args)
    P0   = P_EoS(u0, rho0, mat_id)
    u1   = _u_cold_tab(rho1, mat_id, u_cold_array) + C_V*T_rho(rho1, T_rho_id, T_rho_args)
    P1   = P_EoS(u1, rho1, mat_id)
    rho2 = (rho0 + rho1)/2.
    u2   = _u_cold_tab(rho2, mat_id, u_cold_array) + C_V*T_rho(rho2, T_rho_id, T_rho_args)
    P2   = P_EoS(u2, rho2, mat_id)
    
    rho_aux = rho0 + 1e-6
    u_aux   = _u_cold_tab(rho_aux, mat_id, u_cold_array) + C_V*T_rho(rho_aux, T_rho_id, T_rho_args)
    P_aux   = P_EoS(u_aux, rho_aux, mat_id)

    if ((P0 < P_s and P_s < P1) or (P0 > P_s and P_s > P1)):
        while np.abs(rho1 - rho0) > tolerance:
            u0 = _u_cold_tab(rho0, mat_id, u_cold_array) + C_V*T_rho(rho0, T_rho_id, T_rho_args)
            P0 = P_EoS(u0, rho0, mat_id)
            u1 = _u_cold_tab(rho1, mat_id, u_cold_array) + C_V*T_rho(rho1, T_rho_id, T_rho_args)
            P1 = P_EoS(u1, rho1, mat_id)
            u2 = _u_cold_tab(rho2, mat_id, u_cold_array) + C_V*T_rho(rho2, T_rho_id, T_rho_args)
            P2 = P_EoS(u2, rho2, mat_id)
            
            f0 = P_s - P0
            f2 = P_s - P2
            
            if f0*f2 > 0:
                rho0 = rho2
            else:
                rho1 = rho2
                
            rho2 = (rho0 + rho1)/2.
            
        return rho2
    
    elif (P0 == P_s and P_aux == P_s and P1 != P_s):
        while np.abs(rho1 - rho0) > tolerance:
            rho2 = (rho0 + rho1)/2.
            u0 = _u_cold_tab(rho0, mat_id, u_cold_array) + C_V*T_rho(rho0, T_rho_id, T_rho_args)
            P0 = P_EoS(u0, rho0, mat_id)
            u1 = _u_cold_tab(rho1, mat_id, u_cold_array) + C_V*T_rho(rho1, T_rho_id, T_rho_args)
            P1 = P_EoS(u1, rho1, mat_id)
            rho2 = (rho0 + rho1)/2.
            u2 = _u_cold_tab(rho2, mat_id, u_cold_array) + C_V*T_rho(rho2, T_rho_id, T_rho_args)
            P2 = P_EoS(u2, rho2, mat_id)
            
            if P2 == P_s:
                rho0 = rho2
            else:
                rho1 = rho2
            
            rho2 = rho2 = (rho0 + rho1)/2.
            
        return rho2
    
    elif P_s < P0 and P0 < P1:
        #print("Exception 1\n")
        #print("P0: %.2f P1 %.2f P_s %.2f" %(round(P0/1e9,2), round(P1/1e9,2), round(P_s/1e9,2)))
        return rho0
    elif P_s > P1 and P0 < P1:
        #print("Exception 2\n")
        #print("P0: %.2f P1 %.2f P_s %.2f" %(round(P0/1e9,2), round(P1/1e9,2), round(P_s/1e9,2)))
        return rho1
    else:
        #print("Exception 3\n")
        #print("P0: %.2f P1 %.2f P_s %.2f" %(round(P0/1e9,2), round(P1/1e9,2), round(P_s/1e9,2)))
        return rho2

    return rho2;

def load_u_cold_array(mat_id):
    """ Load precomputed values of cold internal energy for a given material.
    
        Returns:
            u_cold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() (SI).
    """
    if mat_id in [id_idg_HHe, id_idg_N2]:
        return np.array([0.])
    elif mat_id == id_Til_iron:
        u_cold_array = np.load('data/u_cold_array_100.npy')
    elif mat_id == id_Til_granite:
        u_cold_array = np.load('data/u_cold_array_101.npy')
    elif mat_id == id_Til_water:
        u_cold_array = np.load('data/u_cold_array_102.npy')
        
    return u_cold_array

def find_rho_fixed_P_T(P, T, mat_id):
    """ Root finder of the density for EoS using 
        tabulated values of cold internal energy
        
        Args:
            P (float):
                Pressure (SI).
                
            T (float):
                Temperature (SI).
                
            mat_id (int):
                Material id (SI).
                
        Returns:
            rho2 (float):
                Value of the density which satisfies P(u(rho), rho) = 0 
                (SI).
    """
    P = float(P)
    T = float(T)
    
    rho_min     = 1e-9
    rho_max     = 1e15
    u_cold_array = load_u_cold_array(mat_id)
    
    return _find_rho(P, mat_id, 1, [T, 0.], rho_min, rho_max, u_cold_array)

@jit(nopython=True)
def _find_rho_fixed_P_T(P, T, mat_id, u_cold_array):
    """ Root finder of the density for EoS using 
        tabulated values of cold internal energy
        
        Args:
            P (float):
                Pressure (SI).
                
            T (float):
                Temperature (SI).
                
            mat_id (int):
                Material id (SI).
                
            u_cold_array ([float]):
                Precomputed values of cold internal energy
                with function _create_u_cold_array() (SI).
                
        Returns:
            rho2 (float):
                Value of the density which satisfies P(u(rho), rho) = 0 
                (SI).
    """
    P = float(P)
    T = float(T)
    
    rho_min     = 1e-9
    rho_max     = 1e15
    
    return _find_rho(P, mat_id, 1, [T, 0.], rho_min, rho_max, u_cold_array)

def find_P_fixed_T_rho(T, rho, mat_id):
    """ Finder of the pressure for EoS using 
        tabulated values of cold internal energy
        
        Args:
            T (float):
                Temperature (SI).
                
            rho (float):
                Density (SI).
                
            mat_id (int):
                Material id (SI).
                
        Returns:
            P (float):
                Value of the pressure P(u(rho) + C_V*T, rho) 
                (SI).
    """
    T = float(T)
    rho = float(rho)
    
    u = u_cold(rho, mat_id, 10000) + _C_V(mat_id)*T
    P = P_EoS(u, rho, mat_id)
    
    rho_test = rho + 1e-5
    u_test = u_cold(rho_test, mat_id, 10000) + _C_V(mat_id)*T
    P_test = P_EoS(u_test, rho_test, mat_id)
    
    if P != P_test:
        return P
    
    elif P == P_test and P == 0:
        print("This is a flat region of the EoS, where the pressure is 0")
        print("Please consider a higher rho value.")
        return 0
    
    return -1



# =============================================================================
# gamma = 7/5
# 
# n_H2_n_He   = 2 / (1/0.75 - 1)
# m_mol_HHe   = (2*n_H2_n_He + 4) / (n_H2_n_He + 1)
# 
# m_mol_N2 = 28
# cgs_to_SI_m = 0.001
# 
# m_mol = m_mol_HHe
# 
# R_gas / (m_mol * cgs_to_SI_m * (gamma - 1))
# =============================================================================
