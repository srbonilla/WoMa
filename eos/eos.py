#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:22:03 2019

@author: sergio
"""

from numba import njit
import numpy as np
import glob_vars as gv
import tillotson
import sesame
import idg
import hm80
from T_rho import T_rho
import matplotlib.pyplot as plt

@njit
def P_u_rho(u, rho, mat_id):
    """ Computes pressure for any EoS.

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
    mat_type    = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        P = idg.P_u_rho(u, rho, mat_id)
    elif mat_type == gv.type_Til:
        P = tillotson.P_u_rho(u, rho, mat_id)
    elif mat_type == gv.type_HM80:
        P = hm80.P_u_rho(u, rho, mat_id)
        if np.isnan(P): P = 0.
    elif mat_type == gv.type_SESAME:
        P = sesame.P_u_rho(u, rho, mat_id)
        if np.isnan(P): P = 0.
    else:
        raise ValueError("Invalid material ID")
    return P

@njit
def u_rho_T(rho, T, mat_id):
    """ Computes internal energy for any EoS.

        Args:

            rho (double)
                Density (SI).
                
            T (double)
                Temperature (SI).

            mat_id (int)
                Material id.

        Returns:
            u (double)
                Internal energy (SI).
    """
    mat_type    = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        P = idg.u_rho_T(rho, T, mat_id)
    elif mat_type == gv.type_Til:
        P = tillotson.u_rho_T(rho, T, mat_id)
    elif mat_type == gv.type_HM80:
        P = hm80.u_rho_T(rho, T, mat_id)
    elif mat_type == gv.type_SESAME:
        P = sesame.u_rho_T(rho, T, mat_id)
    else:
        raise ValueError("Invalid material ID")
    return P

@njit
def find_rho(P, mat_id, T_rho_type, T_rho_args, rho0, rho1):
    """ Root finder of the density for EoS using
        tabulated values of cold internal energy

        Args:
            P (float):
                Pressure (SI).

            mat_id (int):
                Material id.

            T_rho_type (int)
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
    
    assert rho0 > 0
    assert P >= 0
    assert rho0 < rho1
    
    tolerance = 1E-5

    T0      = T_rho(rho0, T_rho_type, T_rho_args, mat_id)
    u0      = u_rho_T(rho0, T0, mat_id)
    P0      = P_u_rho(u0, rho0, mat_id)
    T1      = T_rho(rho1, T_rho_type, T_rho_args, mat_id)
    u1      = u_rho_T(rho1, T1, mat_id)
    P1      = P_u_rho(u1, rho1, mat_id)
    rho2    = (rho0 + rho1)/2.
    T2      = T_rho(rho2, T_rho_type, T_rho_args, mat_id)
    u2      = u_rho_T(rho2, T2, mat_id)
    P2      = P_u_rho(u2, rho2, mat_id)    
    rho_aux = rho0 + 1e-6
    T_aux   = T_rho(rho_aux, T_rho_type, T_rho_args, mat_id)
    u_aux   = u_rho_T(rho_aux, T_aux, mat_id)
    P_aux   = P_u_rho(u_aux, rho_aux, mat_id)

    if ((P0 < P and P < P1) or (P0 > P and P > P1)):
        max_counter = 200
        counter     = 0
        while np.abs(rho1 - rho0) > tolerance and counter < max_counter:
            T0 = T_rho(rho0, T_rho_type, T_rho_args, mat_id)
            u0 = u_rho_T(rho0, T0, mat_id)
            P0 = P_u_rho(u0, rho0, mat_id)
            T1 = T_rho(rho1, T_rho_type, T_rho_args, mat_id)
            u1 = u_rho_T(rho1, T1, mat_id)
            P1 = P_u_rho(u1, rho1, mat_id)
            T2 = T_rho(rho2, T_rho_type, T_rho_args, mat_id)
            u2 = u_rho_T(rho2, T2, mat_id)
            P2 = P_u_rho(u2, rho2, mat_id)
            
            #if np.isnan(P0): P0 = 0.

            f0 = P - P0
            f2 = P - P2

            if f0*f2 > 0:
                rho0 = rho2
            else:
                rho1 = rho2

            rho2     = (rho0 + rho1)/2.
            counter += 1

        return rho2

    elif (P0 == P and P_aux == P and P1 != P and P0 < P1):
        while np.abs(rho1 - rho0) > tolerance:
            rho2 = (rho0 + rho1)/2.
            T0 = T_rho(rho0, T_rho_type, T_rho_args, mat_id)
            u0 = u_rho_T(rho0, T0, mat_id)
            P0 = P_u_rho(u0, rho0, mat_id)
            T1 = T_rho(rho1, T_rho_type, T_rho_args, mat_id)
            u1 = u_rho_T(rho1, T1, mat_id)
            P1 = P_u_rho(u1, rho1, mat_id)
            rho2 = (rho0 + rho1)/2.
            T2 = T_rho(rho2, T_rho_type, T_rho_args, mat_id)
            u2 = u_rho_T(rho2, T2, mat_id)
            P2 = P_u_rho(u2, rho2, mat_id)

            if P2 == P:
                rho0 = rho2
            else:
                rho1 = rho2

            rho2 = rho2 = (rho0 + rho1)/2.

        return rho2

    elif P < P0 and P0 < P1:
        return rho0
    elif P > P1 and P0 < P1:
        return rho1   
    elif P > P0 and P0 > P1:
        return rho0
    elif P < P1 and P0 > P1:
        return rho1
    else:
# =============================================================================
#         e = "Critical error in find rho.\n" + \
#             "Material: " + str(mat_id) + \
#             "P: " + str(P) + \
#             "T_rho_type: {:d}\n".format(mat_id) + \
#             "T_rho_args: " + str(T_rho_args) + "\n" + \
#             "rho0: {:f}\n".format(rho0) + \
#             "rho1: {:f}\n".format(rho1) + \
#             "Please report this message to the developers. Thank you!\n"
# =============================================================================
        e = "Critical error in find_rho."
            
        raise ValueError(e)

@njit
def P_T_rho(T, rho, mat_id):
    u = u_rho_T(rho, T, mat_id)
    P = P_u_rho(u, rho, mat_id)
    return P

@njit
def rho_P_T(P, T, mat_id):
    mat_type    = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        assert T > 0
        rho0 = 1e-10
        rho1 = 1e5
    elif mat_type == gv.type_Til:
        rho0 = 1e-7
        rho1 = 1e6
    elif mat_type == gv.type_HM80:
        assert T > 0
        if mat_id == gv.id_HM80_HHe:
            rho0 = 1e-1
            rho1 = 1e5
        elif mat_id == gv.id_HM80_ice:
            rho0 = hm80._rho_0_material(mat_id)
            rho1 = 1e5
        elif mat_id == gv.id_HM80_rock:
            rho0 = hm80._rho_0_material(mat_id)
            rho1 = 40000
    elif mat_type == gv.type_SESAME:
        assert T > 0
        assert P > 0
        
        rho0 = 1e-9
        rho1 = 1e5
    else:
        raise ValueError("Invalid material ID")
    return find_rho(P, mat_id, 1, [float(T), 0.], rho0, rho1)

# Visualize EoS
def plot_EoS_P_rho_fixed_T(mat_id_1, mat_id_2, T,
                           P_min=0.1, P_max=1e11, rho_min=100, rho_max=15000):
    
    rho = np.linspace(rho_min, rho_max, 1000)
    P_1 = np.zeros_like(rho)
    P_2 = np.zeros_like(rho)
    for i, rho_i in enumerate(rho):
        P_1[i] = P_T_rho(T, rho_i, mat_id_1)
        P_2[i] = P_T_rho(T, rho_i, mat_id_2)
    
    plt.figure()
    plt.scatter(rho, P_1, label=str(mat_id_1))
    plt.scatter(rho, P_2, label=str(mat_id_2))
    plt.legend(title='Material')
    plt.xlabel(r"$\rho$ [kg m$^{-3}$]")
    plt.ylabel(r"$P$ [Pa]")
    plt.show()
    
    