#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:11:10 2019

@author: Sergio Ruiz-Bonilla
"""

###############################################################################
####################### Libraries and constants ###############################
###############################################################################
import sys
import os

# Go to the WoMa directory
path = '/home/sergio/Documents/WoMa/'
os.chdir(path)
sys.path.append(path)

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
    
# Function to plot results for spining profile
    
def plot_spin_profile(spin_planet):
    
    sp = spin_planet
    
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].scatter(sp.A1_R/R_earth, sp.A1_rho, label = 'original', s = 0.5)
    ax[0].scatter(sp.A1_equator/R_earth, sp.A1_rho_equator, label = 'equatorial profile', s = 1)
    ax[0].scatter(sp.A1_pole/R_earth, sp.A1_rho_pole, label = 'polar profile', s = 1)
    ax[0].set_xlabel(r"$r$ [$R_{earth}$]")
    ax[0].set_ylabel(r"$\rho$ [$kg/m^3$]")
    ax[0].legend()
    
    
    r_array_coarse = np.linspace(0, np.max(sp.A1_equator), 100)
    z_array_coarse = np.linspace(0, np.max(sp.A1_pole), 100)
    rho_grid = np.zeros((r_array_coarse.shape[0], z_array_coarse.shape[0]))
    for i in range(rho_grid.shape[0]):
        radius = r_array_coarse[i]
        for j in range(rho_grid.shape[1]):
            z = z_array_coarse[j]
            rho_grid[i,j] = woma.rho_rz(radius, z,
                                        sp.A1_equator, sp.A1_rho_equator,
                                        sp.A1_pole, sp.A1_rho_pole)
    
    X, Y = np.meshgrid(r_array_coarse/R_earth, z_array_coarse/R_earth)
    Z = rho_grid.T
    levels = np.arange(1000, 15000, 1000)
    ax[1].set_aspect('equal')
    CS = plt.contour(X, Y, Z, levels = levels)
    ax[1].clabel(CS, inline=1, fontsize=10)
    ax[1].set_xlabel(r"$r$ [$R_{earth}$]")
    ax[1].set_ylabel(r"$z$ [$R_{earth}$]")
    ax[1].set_title('Density (Kg/m^3)')
        
    plt.tight_layout()
    plt.show()

###############################################################################
########################## 1 layer planet #####################################
###############################################################################

########################## Example 1.1 ########################################

my_planet = woma.Planet(1)

my_planet.set_core_properties(weos.id_Til_granite, 1, [np.nan, 0.])

P_s  = 0
T_s  = 300
rho_s = weos.find_rho_fixed_P_T(P_s, T_s, weos.id_Til_granite)

my_planet.set_P_surface(P_s)
my_planet.set_T_surface(T_s)
my_planet.set_rho_surface(rho_s)

my_planet.fix_M_given_R(R_earth, 1*M_earth)

plot_spherical_profile(my_planet)

my_spinning_planet = woma.Spin(my_planet)
my_spinning_planet.solve(3, 1.3*my_planet.R, 1.1*my_planet.R)

plot_spin_profile(my_spinning_planet)

###############################################################################
########################## 2 layer planet #####################################
###############################################################################

my_planet = woma.Planet(2)

my_planet.set_core_properties(weos.id_Til_granite, 1, [None, 0])
my_planet.set_mantle_properties(weos.id_idg_N2, 1, [None, 0])

P_s  = 1e4
T_s  = 300
rho_s = weos.find_rho_fixed_P_T(P_s, T_s, 1)

my_planet.set_P_surface(P_s)
my_planet.set_T_surface(T_s)
my_planet.set_rho_surface(rho_s)

#my_planet.fix_B_given_R_M(R_earth, M_earth)
my_planet.fix_M_given_B_R(0.99*R_earth, R_earth, M_earth)
#my_planet.fix_R_given_M_B(2*M_earth, 0.6*R_earth, 20*R_earth)

my_planet.compute_spherical_profile_given_R_M_B(R_earth, M_earth, 0.99*R_earth)

plot_spherical_profile(my_planet)

my_spinning_planet = woma.Spin(my_planet)
my_spinning_planet.solve(2.6, 1.5*my_planet.R, 1.1*my_planet.R)

plot_spin_profile(my_spinning_planet)

###############################################################################
########################## 3 layer planet #####################################
###############################################################################

my_planet = woma.Planet(3)

my_planet.set_core_properties(weos.id_Til_iron, 1, [np.nan, 0])
my_planet.set_mantle_properties(weos.id_Til_granite, 1, [np.nan, 0])
my_planet.set_atmosphere_properties(weos.id_Til_water, 1, [np.nan, 0])

P_s  = 0
T_s  = 300
rho_s = weos.find_rho_fixed_P_T(P_s, T_s, weos.id_Til_water)

my_planet.set_P_surface(P_s)
my_planet.set_T_surface(T_s)
my_planet.set_rho_surface(rho_s)

#my_planet.fix_Bcm_Bma_given_R_M_I(R_earth, M_earth, 0.3*M_earth*R_earth**2)
#my_planet.fix_Bma_given_R_M_Bcm(R_earth, M_earth, 0.49*R_earth)

my_planet.fix_Bcm_given_R_M_Bma(1*R_earth, 0.5*M_earth, 0.75*R_earth)

#my_planet.fix_M_given_R_Bcm_Bma(R_earth, 0.3*R_earth, 0.7*R_earth, 10*M_earth)

#my_planet.fix_R_given_M_Bcm_Bma(M_earth, 0.4*R_earth, 0.8*R_earth, 2*R_earth)

plot_spherical_profile(my_planet)

spin_example = woma.Spin(my_planet)
spin_example.solve(4, 1.2*my_planet.R, 1.1*my_planet.R)

plot_spin_profile(spin_example)

##################################################
############ jacob planets #######################

# + ~canonical proto-Earth: m = 0.887 M_E, mass ratio 30 iron : 70 granite (both Tillotson).
# isothermal 2000 K

proto_earth = woma.Planet(2)
M = M_earth

proto_earth.set_core_properties(weos.id_Til_iron, 1, [None, 2.5])
proto_earth.set_mantle_properties(weos.id_Til_granite, 1, [None, 2.5])

P_s  = 1e5
T_s  = 300
rho_s = weos.find_rho_fixed_P_T(P_s, T_s, weos.id_Til_granite)

proto_earth.set_P_surface(P_s)
proto_earth.set_T_surface(T_s)
proto_earth.set_rho_surface(rho_s)

proto_earth.fix_B_given_R_M(R_earth, M)

#proto_earth.fix_B_given_R_M(1.00*R_earth, M)

R_low = 0.98*R_earth
R_high = 1.00*R_earth
mass_f_core = 0.3

for _ in range(3):
    R_try = (R_low + R_high)/2.
    proto_earth.fix_B_given_R_M(R_try, M)
    
    core_mass_fraction = np.max(proto_earth.A1_M[proto_earth.A1_material == weos.id_Til_iron])/proto_earth.M
    print(core_mass_fraction)
    
    if core_mass_fraction < mass_f_core:
        R_high = R_try
    else:
        R_low = R_try

plot_spherical_profile(proto_earth)

# + ~canonical Theia: m = 0.133, same mass ratio (same mass ratio)
# isothermal 2000 K

theia = woma.Planet(2, mat_id_core=weos.id_Til_iron)
M = 0.133*M_earth

theia.set_core_properties(weos.id_Til_iron, 1, [None, 0])
theia.set_mantle_properties(weos.id_Til_granite, 1, [None, 0])
theia.mat_id_core = weos.id_Til_iron

P_s  = 0
T_s  = 2000
rho_s = weos.find_rho_fixed_P_T(P_s, T_s, weos.id_Til_granite)

theia.set_P_surface(P_s)
theia.set_T_surface(T_s)
theia.set_rho_surface(rho_s)

R_low = 0.60*R_earth
R_high = 0.65*R_earth
mass_f_core = 0.3

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
