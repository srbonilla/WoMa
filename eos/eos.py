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
def find_rho(P, mat_id, T_rho_type, T_rho_args, rho_min, rho_max):
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

            rho_min (float):
                Lower bound for where to look the root (SI).

            rho_max (float):
                Upper bound for where to look the root (SI).

            u_cold_array ([float])
                Precomputed values of cold internal energy
                with function _create_u_cold_array() (SI).

        Returns:
            rho_mid (float):
                Value of the density which satisfies P(u(rho), rho) = 0
                (SI).
    """
    
    assert rho_min > 0
    assert P >= 0
    assert rho_min < rho_max
    
    tolerance = 1E-5

    T_min   = T_rho(rho_min, T_rho_type, T_rho_args, mat_id)
    u_min   = u_rho_T(rho_min, T_min, mat_id)
    P_min   = P_u_rho(u_min, rho_min, mat_id)
    T_max   = T_rho(rho_max, T_rho_type, T_rho_args, mat_id)
    u_max   = u_rho_T(rho_max, T_max, mat_id)
    P_max   = P_u_rho(u_max, rho_max, mat_id)
    rho_mid = (rho_min + rho_max)/2.
    T_mid   = T_rho(rho_mid, T_rho_type, T_rho_args, mat_id)
    u_mid   = u_rho_T(rho_mid, T_mid, mat_id)
    P_mid   = P_u_rho(u_mid, rho_mid, mat_id)    
    rho_aux = rho_min + 1e-6
    T_aux   = T_rho(rho_aux, T_rho_type, T_rho_args, mat_id)
    u_aux   = u_rho_T(rho_aux, T_aux, mat_id)
    P_aux   = P_u_rho(u_aux, rho_aux, mat_id)

    if ((P_min < P and P < P_max) or (P_min > P and P > P_max)):
        max_counter = 200
        counter     = 0
        while np.abs(rho_max - rho_min) > tolerance and counter < max_counter:
            T_min = T_rho(rho_min, T_rho_type, T_rho_args, mat_id)
            u_min = u_rho_T(rho_min, T_min, mat_id)
            P_min = P_u_rho(u_min, rho_min, mat_id)
            T_max = T_rho(rho_max, T_rho_type, T_rho_args, mat_id)
            u_max = u_rho_T(rho_max, T_max, mat_id)
            P_max = P_u_rho(u_max, rho_max, mat_id)
            T_mid = T_rho(rho_mid, T_rho_type, T_rho_args, mat_id)
            u_mid = u_rho_T(rho_mid, T_mid, mat_id)
            P_mid = P_u_rho(u_mid, rho_mid, mat_id)
            
            #if np.isnan(P_min): P_min = 0.

            f0 = P - P_min
            f2 = P - P_mid

            if f0*f2 > 0:
                rho_min = rho_mid
            else:
                rho_max = rho_mid

            rho_mid     = (rho_min + rho_max)/2.
            counter += 1

        return rho_mid

    elif (P_min == P and P_aux == P and P_max != P and P_min < P_max):
        while np.abs(rho_max - rho_min) > tolerance:
            rho_mid = (rho_min + rho_max)/2.
            T_min = T_rho(rho_min, T_rho_type, T_rho_args, mat_id)
            u_min = u_rho_T(rho_min, T_min, mat_id)
            P_min = P_u_rho(u_min, rho_min, mat_id)
            T_max = T_rho(rho_max, T_rho_type, T_rho_args, mat_id)
            u_max = u_rho_T(rho_max, T_max, mat_id)
            P_max = P_u_rho(u_max, rho_max, mat_id)
            rho_mid = (rho_min + rho_max)/2.
            T_mid = T_rho(rho_mid, T_rho_type, T_rho_args, mat_id)
            u_mid = u_rho_T(rho_mid, T_mid, mat_id)
            P_mid = P_u_rho(u_mid, rho_mid, mat_id)

            if P_mid == P:
                rho_min = rho_mid
            else:
                rho_max = rho_mid

            rho_mid = rho_mid = (rho_min + rho_max)/2.

        return rho_mid

    elif P < P_min and P_min < P_max:
        return rho_min
    elif P > P_max and P_min < P_max:
        return rho_max   
    elif P > P_min and P_min > P_max:
        return rho_min
    elif P < P_max and P_min > P_max:
        return rho_max
    else:
# =============================================================================
#         e = "Critical error in find rho.\n" + \
#             "Material: " + str(mat_id) + \
#             "P: " + str(P) + \
#             "T_rho_type: {:d}\n".format(T_rho_type) + \
#             "T_rho_args: " + str(T_rho_args) + "\n" + \
#             "rho_min: {:f}\n".format(rho_min) + \
#             "rho_max: {:f}\n".format(rho_max) + \
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
        rho_min = 1e-10
        rho_max = 1e5
    elif mat_type == gv.type_Til:
        rho_min = 1e-7
        rho_max = 1e6
    elif mat_type == gv.type_HM80:
        assert T > 0
        if mat_id == gv.id_HM80_HHe:
            rho_min = 1e-1
            rho_max = 1e5
        elif mat_id == gv.id_HM80_ice:
            rho_min = 1e0
            rho_max = 1e5
        elif mat_id == gv.id_HM80_rock:
            rho_min = 1e0
            rho_max = 40000
    elif mat_type == gv.type_SESAME:
        assert T > 0
        assert P > 0
        
        rho_min = 1e-9
        rho_max = 1e5
    else:
        raise ValueError("Invalid material ID")
    return find_rho(P, mat_id, 1, [float(T), 0.], rho_min, rho_max)

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
    plt.legend(title="Material")
    plt.xlabel(r"$\rho$ [kg m$^{-3}$]")
    plt.ylabel(r"$P$ [Pa]")
    plt.show()
    
    