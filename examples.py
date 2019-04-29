#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:11:10 2019

@author: Sergio Ruiz-Bonilla
"""

###############################################################################
####################### Libraries and constants ###############################
###############################################################################

path = '/home/sergio/Documents/SpiPGen/'     # Set local project folder
import os
os.chdir(path)
import sys
sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt
import woma
import eos

R_earth = 6371000;
M_earth = 5.972E24;

###############################################################################
########################## Initial set up #####################################
###############################################################################

# Only need to run once

woma.set_up()

# Function to plot results for spherical profile

def plot_spherical_profile(planet):
    
    fig, ax = plt.subplots(2,2, figsize=(12,12))
    
    ax[0,0].plot(planet.A1_R/R_earth, planet.A1_rho)
    ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")
    
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

my_planet.set_core_properties(101, 1, [np.nan, 0.])

Ps   = 0
Ts   = 300
rhos = eos.find_rho_fixed_P_T(Ps, Ts, 101)

my_planet.set_P_surface(Ps)
my_planet.set_T_surface(Ts)
my_planet.set_rho_surface(rhos)

my_planet.fix_M_given_R(R_earth, 1*M_earth)

plot_spherical_profile(my_planet)

my_spinning_planet = woma.Spin(my_planet)
my_spinning_planet.solve(3, 1.3*my_planet.R, 1.1*my_planet.R)

plot_spin_profile(my_spinning_planet)

###############################################################################
########################## 2 layer planet #####################################
###############################################################################

my_planet = woma.Planet(2)

my_planet.set_core_properties(101, 1, [np.nan, 0])
my_planet.set_mantle_properties(102, 1, [np.nan, 0])

Ps   = 0
Ts   = 600
rhos = eos.find_rho_fixed_P_T(Ps, Ts, 102)

my_planet.set_P_surface(Ps)
my_planet.set_T_surface(Ts)
my_planet.set_rho_surface(rhos)

my_planet.fix_B_given_R_M(R_earth, 0.5*M_earth)
#my_planet.fix_M_given_B_R(0.2*R_earth, R_earth, M_earth)
#my_planet.fix_R_given_M_B(2*M_earth, 0.6*R_earth, 20*R_earth)

plot_spherical_profile(my_planet)

my_spinning_planet = woma.Spin(my_planet)
my_spinning_planet.solve(2.6, 1.5*my_planet.R, 1.1*my_planet.R)

plot_spin_profile(my_spinning_planet)

###############################################################################
########################## 3 layer planet #####################################
###############################################################################

my_planet = woma.Planet(3)

my_planet.set_core_properties(100, 1, [np.nan, 0])
my_planet.set_mantle_properties(101, 1, [np.nan, 0])
my_planet.set_atmosphere_properties(102, 1, [np.nan, 0])

Ps   = 0
Ts   = 300
rhos = eos.find_rho_fixed_P_T(Ps, Ts, 102)

my_planet.set_P_surface(Ps)
my_planet.set_T_surface(Ts)
my_planet.set_rho_surface(rhos)

#my_planet.fix_Bcm_Bma_given_R_M_I(R_earth, M_earth, 0.3*M_earth*R_earth**2)
#my_planet.fix_Bma_given_R_M_Bcm(R_earth, M_earth, 0.49*R_earth)

my_planet.fix_Bcm_given_R_M_Bma(1*R_earth, 0.5*M_earth, 0.75*R_earth)

#my_planet.fix_M_given_R_Bcm_Bma(R_earth, 0.3*R_earth, 0.7*R_earth, 10*M_earth)

#my_planet.fix_R_given_M_Bcm_Bma(M_earth, 0.4*R_earth, 0.8*R_earth, 2*R_earth)

plot_spherical_profile(my_planet)

spin_example = woma.Spin(my_planet)
spin_example.solve(4, 1.2*my_planet.R, 1.1*my_planet.R)

plot_spin_profile(spin_example)
