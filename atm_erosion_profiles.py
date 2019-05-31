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
    
    fig, ax = plt.subplots(2,2, figsize=(12,12))
    
    ax[0,0].plot(planet.A1_R/R_earth, planet.A1_rho)
    ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")
    ax[0,0].set_yscale('log')
    
    ax[1,0].plot(planet.A1_R/R_earth, planet.A1_M/M_earth)
    ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")
    
    ax[0,1].plot(planet.A1_R/R_earth, planet.A1_P)
    ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[0,1].set_ylabel(r"$P$ $[Pa]$")
    
    ax[1,1].plot(planet.A1_R/R_earth, planet.A1_T)
    ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[1,1].set_ylabel(r"$T$ $[K]$")
    
    plt.tight_layout()
    plt.show()
    
###############################################################################
# canonical proto-Earth: m = 0.887 M_E, mass ratio 30 iron : 70 granite (both Tillotson).
# isothermal 2000 K

proto_earth = woma.Planet(2)
M = 0.887*M_earth

proto_earth.set_core_properties(weos.id_Til_iron, 1, [None, 0])
proto_earth.set_mantle_properties(weos.id_Til_granite, 1, [None, 0])

P_s  = 1e5
T_s  = 2000
rho_s = weos.find_rho_fixed_P_T(P_s, T_s, weos.id_Til_granite)

proto_earth.set_P_surface(P_s)
proto_earth.set_T_surface(T_s)
proto_earth.set_rho_surface(rho_s)

R_low = 0.95*R_earth
R_high = 1.02*R_earth
mass_f_core = 0.3

# Make sure the following computations work and the mass fraction resulting
# is between the desired value
#proto_earth.fix_B_given_R_M(R_high, M)
#print(np.max(proto_earth.A1_M[proto_earth.A1_material == weos.id_Til_iron])/proto_earth.M)
#proto_earth.fix_B_given_R_M(R_low, M)
#print(np.max(proto_earth.A1_M[proto_earth.A1_material == weos.id_Til_iron])/proto_earth.M)

# Iterating to obtain desired core-mantle mass fraction
for _ in range(1):
    R_try = (R_low + R_high)/2.
    proto_earth.fix_B_given_R_M(R_try, M)
    
    core_mass_fraction = np.max(proto_earth.A1_M[proto_earth.A1_material == weos.id_Til_iron])/proto_earth.M
    print(core_mass_fraction)
    
    if core_mass_fraction < mass_f_core:
        R_high = R_try
    else:
        R_low = R_try

plot_spherical_profile(proto_earth)

###############################################################################
# canonical Theia: m = 0.133, mass ratio 30 iron : 70 granite (both Tillotson).
# isothermal 2000 K

theia = woma.Planet(2)
M = 0.133*M_earth

theia.set_core_properties(weos.id_Til_iron, 1, [None, 0])
theia.set_mantle_properties(weos.id_Til_granite, 1, [None, 0])

P_s  = 0
T_s  = 2000
rho_s = weos.find_rho_fixed_P_T(P_s, T_s, weos.id_Til_granite)

theia.set_P_surface(P_s)
theia.set_T_surface(T_s)
theia.set_rho_surface(rho_s)

R_low = 0.60*R_earth
R_high = 0.65*R_earth
mass_f_core = 0.3

# Make sure the following computations work and the mass fraction resulting
# is between the desired value
theia.fix_B_given_R_M(R_high, M)
print(np.max(theia.A1_M[theia.A1_material == weos.id_Til_iron])/theia.M)
theia.fix_B_given_R_M(R_low, M)
print(np.max(theia.A1_M[theia.A1_material == weos.id_Til_iron])/theia.M)

# Iterating to obtain desired core-mantle mass fraction
for _ in range(3):
    R_try = (R_low + R_high)/2.
    theia.fix_B_given_R_M(R_try, M)
    
    core_mass_fraction = np.max(theia.A1_M[theia.A1_material == weos.id_Til_iron])/theia.M
    print(core_mass_fraction)
    
    if core_mass_fraction < mass_f_core:
        R_high = R_try
    else:
        R_low = R_try

plot_spherical_profile(theia)

#####################################################

mat_id_atm = 2
T_rho_id_atm = 1
T_rho_args_atm = [None, 0.]
rho_stop = 1e-6

planet = woma.add_atmosphere(proto_earth, mat_id_atm, T_rho_id_atm, T_rho_args_atm, rho_stop)

plot_spherical_profile(planet)

# Pressure at mantle-atm
np.max(planet.A1_P[planet.A1_material == mat_id_atm])



planet = proto_earth
mat_id_atm = 1
T_rho_id_atm = 1 
T_rho_args_atm = [None, 0]
rho_stop = 0.


