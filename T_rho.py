#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:37:58 2019

@author: sergio
"""

from numba import njit
import glob_vars as gv
#import idg
#import hm80
#import tillotson
#import sesame

@njit
def T_rho(rho, T_rho_type, T_rho_args, mat_id):
    """ Computes temperature given density (T = f(rho)).

        Args:
            rho (float)
                Density (SI).

            T_rho_type (int)
                Relation between T and rho to be used. See woma.Planet.

            T_rho_args (list)
                Extra arguments to determine the relation. See woma.Planet.

            mat_id (int)]
                The material ID.

        Returns:
            Temperature (SI)
    """
    #mat_type    = mat_id // gv.type_factor

    # T = K*rho**alpha, T_rho_args = [K, alpha]
    if (T_rho_type == gv.type_rho_pow):
        K       = T_rho_args[0]
        alpha   = T_rho_args[1]
        return K * rho**alpha

    # Adiabatic, T_rho_args = [s_adb], [rho_prv, T_prv], or [T rho^(1-gamma)]
# =============================================================================
#     elif T_rho_type == gv.type_adb:
#         if mat_type == gv.type_idg:
#             # T rho^(1-gamma) = constant
#             gamma   = idg.idg_gamma(mat_id)
#             return T_rho_args[0] * rho**(gamma - 1)
#         elif mat_id == gv.id_HM80_HHe:
#             return hm80.T_rho_HM80_HHe(rho, T_rho_args[0], T_rho_args[1])
#         elif mat_type == gv.type_SESAME:
#             return sesame.T_rho_s(rho, T_rho_args[0], mat_id)
#         elif mat_type == gv.type_Til:
#             raise ValueError("Entropy not implemented for this material type")
# 
# =============================================================================
    else:
        raise ValueError("T_rho_type not implemented")