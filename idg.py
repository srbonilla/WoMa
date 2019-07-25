#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:12:01 2019

@author: sergio
"""

import numpy as np
from numba import njit
import os
import sys
import glob_vars as gv

# Go to the WoMa directory
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
sys.path.append(dir)

@njit
def idg_gamma(mat_id):
    """ Return the adiabatic index gamma for an ideal gas. """
    if mat_id == gv.id_idg_HHe:
        return 1.4
    elif mat_id == gv.id_idg_N2:
        return 1.4
    elif mat_id == gv.id_idg_CO2:
        return 1.29
    else:
        raise ValueError("Invalid material ID")

@njit
def P_u_rho(u, rho, mat_id):
    """ Computes pressure for the ideal gas EoS.

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
    # Adiabatic constant
    gamma    = idg_gamma(mat_id)

    P = (gamma - 1)*u*rho

    return P
    
@njit
def C_V(mat_id):
    """ Returns specific heat capacity for a given material id (SI)

        Args:
            mat_id (int)
                Material id.

        Returns:
            C_V (double)
                Specific heat capacity (SI).

    """
    if mat_id == gv.id_idg_HHe:
        return 9093.98
    elif mat_id == gv.id_idg_N2:
        return 742.36
    elif mat_id == gv.id_idg_CO2:
        return 661.38
    else:
        raise ValueError("Invalid material ID")
        
@njit
def u_rho_T(rho, T, mat_id):
    mat_type    = mat_id // gv.type_factor
    if (mat_type == gv.type_idg):
        return C_V(mat_id)*T
    else:
        raise ValueError("Invalid material ID")
