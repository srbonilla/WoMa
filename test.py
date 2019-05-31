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
# Example 1.1

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 1,
    A1_mat_layer    = ['Til_granite'],
    A1_T_rho_type   = [1],
    A1_T_rho_args   = [[None, 0.]],
    A1_R_layer      = [0.988 * R_earth],
    M               = 0.8*M_earth,
    P_s             = 0,
    T_s             = 300,
    )

prof_pE.R_max = R_earth

prof_pE.gen_prof_L1_fix_R_given_M()

plot_spherical_profile(prof_pE)

###############################################################################
# Example 1.2

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 1,
    A1_mat_layer    = ['Til_granite'],
    A1_T_rho_type   = [1],
    A1_T_rho_args   = [[None, 0.]],
    A1_R_layer      = [R_earth],
    P_s             = 0,
    T_s             = 300,
    )

prof_pE.M_max = M_earth

prof_pE.gen_prof_L1_fix_M_given_R()

plot_spherical_profile(prof_pE)

###############################################################################
# Example 2.1

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 2,
    A1_mat_layer    = ['Til_iron', 'Til_granite'],
    A1_T_rho_type   = [1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.]],
    A1_R_layer      = [None, R_earth],
    M               = M_earth,
    P_s             = 0,
    T_s             = 300,
    )

prof_pE.gen_prof_L2_fix_R1_given_R_M()

plot_spherical_profile(prof_pE)

###############################################################################
# Example 2.2

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 2,
    A1_mat_layer    = ['Til_iron', 'Til_granite'],
    A1_T_rho_type   = [1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.]],
    A1_R_layer      = [0.40*R_earth, R_earth],
    M               = M_earth,
    P_s             = 0,
    T_s             = 300,
    )

prof_pE.R_max = 2*R_earth
prof_pE.gen_prof_L2_fix_R_given_M_R1()

plot_spherical_profile(prof_pE)

###############################################################################
# Example 2.3

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 2,
    A1_mat_layer    = ['Til_iron', 'Til_granite'],
    A1_T_rho_type   = [1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.]],
    A1_R_layer      = [0.40*R_earth, R_earth],
    P_s             = 0,
    T_s             = 300,
    )

prof_pE.M_max = 2*M_earth
prof_pE.gen_prof_L2_fix_M_given_R1_R()

plot_spherical_profile(prof_pE)

###############################################################################
# Example 2.4

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 2,
    A1_mat_layer    = ['Til_iron', 'Til_granite'],
    A1_T_rho_type   = [1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.]],
    A1_R_layer      = [None, R_earth],
    M               = 0.887*M_earth,
    P_s             = 1e5,
    T_s             = 2000,
    num_attempt     = 10
    )

prof_pE.gen_prof_L2_fix_R1_given_R_M()

mat_atm = 'idg_N2'
T_rho_type_atm = 1
T_rho_args_atm = [None, 0]

prof_pE.gen_prof_L3_given_prof_L2(
    mat_atm,
    T_rho_type_atm,
    T_rho_args_atm,
    rho_min=1e-6
    )

plot_spherical_profile(prof_pE)

###############################################################################
# Example 3.1

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 3,
    A1_mat_layer    = ['Til_iron', 'Til_granite', 'Til_water'],
    A1_T_rho_type   = [1, 1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
    A1_R_layer      = [None, None, R_earth],
    P_s             = 0,
    T_s             = 300,
    I_MR2           = 0.3*M_earth*R_earth**2,
    M               = M_earth,
    num_attempt     = 5,
    num_attempt_2   = 5
    )

prof_pE.gen_prof_L3_fix_R1_R2_given_R_M_I()

plot_spherical_profile(prof_pE)

###############################################################################
# Example 3.2

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 3,
    A1_mat_layer    = ['Til_iron', 'Til_granite', 'Til_water'],
    A1_T_rho_type   = [1, 1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
    A1_R_layer      = [0.55*R_earth, None, R_earth],
    P_s             = 0,
    T_s             = 300,
    M               = M_earth
    )

prof_pE.gen_prof_L3_fix_R2_given_R_M_R1()

plot_spherical_profile(prof_pE)

###############################################################################
# Example 3.3

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 3,
    A1_mat_layer    = ['Til_iron', 'Til_granite', 'Til_water'],
    A1_T_rho_type   = [1, 1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
    A1_R_layer      = [None, 0.9*R_earth, R_earth],
    P_s             = 0,
    T_s             = 300,
    M               = M_earth
    )

prof_pE.gen_prof_L3_fix_R1_given_R_M_R2()

plot_spherical_profile(prof_pE)

###############################################################################
# Example 3.4

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 3,
    A1_mat_layer    = ['Til_iron', 'Til_granite', 'Til_water'],
    A1_T_rho_type   = [1, 1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
    A1_R_layer      = [0.5*R_earth, 0.9*R_earth, R_earth],
    P_s             = 0,
    T_s             = 300,
    M_max           = 2*M_earth
    )

prof_pE.gen_prof_L3_fix_M_given_R_R1_R2()

plot_spherical_profile(prof_pE)

###############################################################################
# Example 3.5

prof_pE = woma.Planet(
    name            = "prof_pE",
    num_layer       = 3,
    A1_mat_layer    = ['Til_iron', 'Til_granite', 'Til_water'],
    A1_T_rho_type   = [1, 1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
    A1_R_layer      = [0.5*R_earth, 0.9*R_earth, None],
    P_s             = 0,
    T_s             = 300,
    M               = M_earth,
    R_max           = 2*R_earth
    )

prof_pE.gen_prof_L3_fix_R_given_M_R1_R2()

plot_spherical_profile(prof_pE)



# canonical proto-Earth: m = 0.887 M_E, mass ratio 30 iron : 70 granite (both Tillotson).
# isothermal 2000 K

# R_low = 0.95*R_earth 
# R_high = 1.02*R_earth
# mass_f_core = 0.3
# 
# prof_pE.gen_prof_L2_fix_R1_given_R_M()
# 
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
# 
# prof_pE.save_profile()
# 
# # Load and plot the profile
# prof_pE.load_profile_arrays()
# 
# plot_spherical_profile(prof_pE)


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
