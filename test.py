#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:07:49 2019

@author: 
"""

###############################################################################
####################### Libraries and constants ###############################
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import woma
import weos

R_earth = 6371000
M_earth = 5.972E24

###############################################################################
########################## Initial set up #####################################
###############################################################################

# Function to plot results for spherical profile

def plot_spherical_profile(planet):
    
    fig, ax = plt.subplots(2, 2, figsize=(7,7))
    
    ax[0,0].plot(planet.A1_r/R_earth, planet.A1_rho)
    ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")
    ax[0,0].set_yscale("log")
    ax[0,0].set_xlim(0, None)
    
    ax[1,0].plot(planet.A1_r/R_earth, planet.A1_m_enc/M_earth)
    ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")
    ax[1,0].set_xlim(0, None)
    
    ax[0,1].plot(planet.A1_r/R_earth, planet.A1_P)
    ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[0,1].set_ylabel(r"$P$ $[Pa]$")
    ax[0,1].set_xlim(0, None)
    
    ax[1,1].plot(planet.A1_r/R_earth, planet.A1_T)
    ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[1,1].set_ylabel(r"$T$ $[K]$")
    ax[1,1].set_xlim(0, None)
    
    plt.tight_layout()
    plt.show()
    
###############################################################################
# canonical proto-Earth: m = 0.887 M_E, mass ratio 30 iron : 70 granite (both Tillotson).
# isothermal 2000 K

prof_pE = woma._2l_planet("prof_pE")
M = 0.887*M_earth

prof_pE.set_core_properties(weos.id_Til_iron, 1, [None, 0])
prof_pE.set_mantle_properties(weos.id_Til_granite, 1, [None, 0])

P_s  = 1e5
T_s  = 2000
rho_s = weos.find_rho_fixed_P_T(P_s, T_s, weos.id_Til_granite)

prof_pE.set_P_surface(P_s)
prof_pE.set_T_surface(T_s)
prof_pE.set_rho_surface(rho_s)

R_low = 0.95*R_earth
R_high = 1.02*R_earth
mass_f_core = 0.3

# prof_pE.fix_B_given_R_M((R_low + R_high)/2., M)

# # Make sure the following computations work and the mass fraction resulting
# # is between the desired value
# prof_pE.fix_B_given_R_M(R_high, M)
# print(np.max(prof_pE.A1_m_enc[prof_pE.A1_mat_id == weos.id_Til_iron])/prof_pE.M)
# prof_pE.fix_B_given_R_M(R_low, M)
# print(np.max(prof_pE.A1_m_enc[prof_pE.A1_mat_id == weos.id_Til_iron])/prof_pE.M)
# 
# # Iterating to obtain desired core-mantle mass fraction
# for _ in range(3):
#     R_try = (R_low + R_high)/2.
#     prof_pE.fix_B_given_R_M(R_try, M)
# 
#     core_mass_fraction = np.max(prof_pE.A1_m_enc[prof_pE.A1_mat_id == weos.id_Til_iron])/prof_pE.M
#     print(core_mass_fraction)
# 
#     if core_mass_fraction < mass_f_core:
#         R_high = R_try
#     else:
#         R_low = R_try
        
# prof_pE.save_profile()

# Load and plot the profile
prof_pE.load_profile_arrays()

plot_spherical_profile(prof_pE)


# ###############################################################################
# # canonical Theia: m = 0.133, mass ratio 30 iron : 70 granite (both Tillotson).
# # isothermal 2000 K
# 
# theia = woma.Planet(2)
# M = 0.133*M_earth
# 
# theia.set_core_properties(weos.id_Til_iron, 1, [None, 0])
# theia.set_mantle_properties(weos.id_Til_granite, 1, [None, 0])
# theia.mat_id_core = weos.id_Til_iron
# 
# P_s  = 0
# T_s  = 2000
# rho_s = weos.find_rho_fixed_P_T(P_s, T_s, weos.id_Til_granite)
# 
# theia.set_P_surface(P_s)
# theia.set_T_surface(T_s)
# theia.set_rho_surface(rho_s)
# 
# R_low = 0.60*R_earth
# R_high = 0.65*R_earth
# mass_f_core = 0.3
# 
# # Make sure the following computations work and the mass fraction resulting
# # is between the desired value
# theia.fix_B_given_R_M(R_high, M)
# print(np.max(theia.A1_m_enc[theia.A1_mat_id == weos.id_Til_iron])/theia.M)
# theia.fix_B_given_R_M(R_low, M)
# print(np.max(theia.A1_m_enc[theia.A1_mat_id == weos.id_Til_iron])/theia.M)
# 
# # Iterating to obtain desired core-mantle mass fraction
# for _ in range(3):
#     R_try = (R_low + R_high)/2.
#     theia.fix_B_given_R_M(R_try, M)
# 
#     core_mass_fraction = np.max(theia.A1_m_enc[theia.A1_mat_id == weos.id_Til_iron])/theia.M
#     print(core_mass_fraction)
# 
#     if core_mass_fraction < mass_f_core:
#         R_high = R_try
#     else:
#         R_low = R_try
# 
# plot_spherical_profile(theia)

