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
import swift_io
import h5py

R_earth = 6371000
M_earth = 5.972E24

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
#my_planet.fix_R_given_M(0.85*M_earth, 2*R_earth)

#plot_spherical_profile(my_planet)

my_spinning_planet = woma.Spin(my_planet)
my_spinning_planet.solve(3, 1.4*R_earth, 1.1*R_earth)

#plot_spin_profile(my_spinning_planet)

particles = woma.GenSpheroid(my_spinning_planet, 1e7, iterations=5)

positions = np.array([particles.A1_x, particles.A1_y, particles.A1_z]).T
velocities = np.array([particles.A1_vx, particles.A1_vy, particles.A1_vz]).T

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '1layer_10e7.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, positions, velocities,
                             particles.A1_m, particles.A1_h,
                             particles.A1_rho, particles.A1_P, particles.A1_u,
                             particles.A1_picle_id, particles.A1_mat_id,
                             4*R_earth, swift_to_SI) 
    
np.save('r_array', my_spinning_planet.A1_equator)
np.save('z_array', my_spinning_planet.A1_pole)
np.save('rho_e', my_spinning_planet.A1_rho_equator)
np.save('rho_p', my_spinning_planet.A1_rho_pole)


r_array = my_spinning_planet.A1_equator
rho_e = my_spinning_planet.A1_rho_equator
z_array = my_spinning_planet.A1_pole
rho_p = my_spinning_planet.A1_rho_pole
Tw = 3
N = 1e5
mat_id_core = 101
T_rho_id_core = 1
T_rho_args_core = [300., 0.]
ucold_array_core = eos.load_ucold_array(101)
iterations = 1
N_neig=48

x = np.linspace(0, 100, 100)
y2 = np.power(x, 2)
y3 = np.power(x, 2.1)
plt.scatter(x, y2)
plt.scatter(x, y3)
plt.show()

###############################################################################
########################## 2 layer planet #####################################
###############################################################################

my_planet = woma.Planet(2)

my_planet.set_core_properties(100, 1, [np.nan, 0])
my_planet.set_mantle_properties(101, 1, [np.nan, 0])

Ps   = 0
Ts   = 300
rhos = eos.find_rho_fixed_P_T(Ps, Ts, 101)

my_planet.set_P_surface(Ps)
my_planet.set_T_surface(Ts)
my_planet.set_rho_surface(rhos)

my_planet.fix_B_given_R_M(R_earth, M_earth)
#my_planet.fix_M_given_B_R(0.42*R_earth, R_earth, 2*M_earth)
#my_planet.fix_R_given_M_B(M_earth, 0.42*R_earth, 2*R_earth)

#plot_spherical_profile(my_planet)

my_spinning_planet = woma.Spin(my_planet)
my_spinning_planet.solve(2.6, 1.45*R_earth, 1.1*R_earth)

#plot_spin_profile(my_spinning_planet)

particles = woma.GenSpheroid(my_spinning_planet, 1e7, iterations=5)

positions = np.array([particles.A1_x, particles.A1_y, particles.A1_z]).T
velocities = np.array([particles.A1_vx, particles.A1_vy, particles.A1_vz]).T

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '2layer_10e7.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, positions, velocities,
                             particles.A1_m, particles.A1_h,
                             particles.A1_rho, particles.A1_P, particles.A1_u,
                             particles.A1_picle_id, particles.A1_mat_id,
                             4*R_earth, swift_to_SI) 
    
np.save('r_array', my_spinning_planet.A1_equator)
np.save('z_array', my_spinning_planet.A1_pole)
np.save('rho_e', my_spinning_planet.A1_rho_equator)
np.save('rho_p', my_spinning_planet.A1_rho_pole)

# plot spherical profiles paper#################################

my_planet1 = woma.Planet(1)
my_planet1.set_core_properties(101, 1, [np.nan, 0.])
Ps   = 0
Ts   = 300
rhos = eos.find_rho_fixed_P_T(Ps, Ts, 101)
my_planet1.set_P_surface(Ps)
my_planet1.set_T_surface(Ts)
my_planet1.set_rho_surface(rhos)
my_planet1.fix_M_given_R(R_earth, 1*M_earth)

my_planet2 = woma.Planet(2)
my_planet2.set_core_properties(100, 1, [np.nan, 0])
my_planet2.set_mantle_properties(101, 1, [np.nan, 0])
Ps   = 0
Ts   = 300
rhos = eos.find_rho_fixed_P_T(Ps, Ts, 101)
my_planet2.set_P_surface(Ps)
my_planet2.set_T_surface(Ts)
my_planet2.set_rho_surface(rhos)
my_planet2.fix_B_given_R_M(R_earth, M_earth)


fig, ax = plt.subplots(2,2, figsize=(12,12))
    
ax[0,0].plot(my_planet1.A1_R/R_earth, my_planet1.A1_rho, label="Test 1")
ax[0,0].plot(my_planet2.A1_R/R_earth, my_planet2.A1_rho, label="Test 2")
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")
ax[0,0].legend()
    
ax[1,0].plot(my_planet1.A1_R/R_earth, my_planet1.A1_M/M_earth, label="Test 1")
ax[1,0].plot(my_planet2.A1_R/R_earth, my_planet2.A1_M/M_earth, label="Test 2")
ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")
ax[1,0].legend()
    
ax[0,1].plot(my_planet1.A1_R/R_earth, my_planet1.A1_P, label="Test 1")
ax[0,1].plot(my_planet2.A1_R/R_earth, my_planet2.A1_P, label="Test 2")
ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,1].set_ylabel(r"$P$ $[Pa]$")
ax[0,1].legend()

ax[1,1].plot(my_planet1.A1_R/R_earth, my_planet1.A1_T, label="Test 1")
ax[1,1].plot(my_planet2.A1_R/R_earth, my_planet2.A1_T, label="Test 2")
ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,1].set_ylabel(r"$T$ $[K]$")
ax[1,1].legend()
    
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.plot(my_planet1.A1_R/R_earth, my_planet1.A1_rho, label="Test 1")
ax.plot(my_planet2.A1_R/R_earth, my_planet2.A1_rho, label="Test 2")
ax.set_xlabel(r"$r$ $[R_{earth}]$")
ax.set_ylabel(r"$\rho$ $[kg/m^3]$")
ax.legend()
plt.tight_layout()
plt.savefig("spherical_profiles.png")
plt.show()

# =============================================================================
# r_array = my_spinning_planet.A1_equator
# rho_e = my_spinning_planet.A1_rho_equator
# z_array = my_spinning_planet.A1_pole
# rho_p = my_spinning_planet.A1_rho_pole
# Tw = 2.6
# N = 1e6
# mat_id_core = 100
# T_rho_id_core = 1
# T_rho_args_core = [300., 0.]
# mat_id_mantle = 101
# T_rho_id_mantle = 1
# T_rho_args_mantle = [300., 0.]
# ucold_array_core = eos.load_ucold_array(100)
# ucold_array_mantle = eos.load_ucold_array(101)
# iterations = 10
# N_neig=48
# rho_i = 10000
# =============================================================================

###############################################################################
########################## 3 layer planet #####################################
###############################################################################

example = woma.Planet(3)

example.set_core_properties(100, 1, [np.nan, 0])
example.set_mantle_properties(101, 1, [np.nan, 0])
example.set_atmosphere_properties(102, 1, [np.nan, 0])

Ps   = 0
Ts   = 300
rhos = eos.find_rho_fixed_P_T(Ps, Ts, 102)

example.set_P_surface(Ps)
example.set_T_surface(Ts)
example.set_rho_surface(rhos)

#example.fix_Bcm_Bma_given_R_M_I(R_earth, M_earth, 0.3*M_earth*R_earth**2)
example.fix_Bma_given_R_M_Bcm(R_earth, M_earth, 0.50*R_earth)

#example.fix_Bcm_given_R_M_Bma(R_earth, M_earth, 0.85*R_earth)

#example.fix_M_given_R_Bcm_Bma(R_earth, 0.3*R_earth, 0.7*R_earth, 10*M_earth)

#example.fix_R_given_M_Bcm_Bma(M_earth, 0.4*R_earth, 0.8*R_earth, 10*R_earth)

plot_spherical_profile(example)

my_spinning_planet = woma.Spin(example)
my_spinning_planet.solve(2.6, 1.5*R_earth, 1.1*R_earth)

plot_spin_profile(my_spinning_planet)

particles = woma.GenSpheroid(my_spinning_planet, 1e5, iterations=5)

# =============================================================================
# r_array = my_spinning_planet.A1_equator
# rho_e = my_spinning_planet.A1_rho_equator
# z_array = my_spinning_planet.A1_pole
# rho_p = my_spinning_planet.A1_rho_pole
# Tw = 2.6
# N = 1e5
# mat_id_core = 100
# T_rho_id_core = 1
# T_rho_args_core = [300., 0.]
# mat_id_mantle = 101
# T_rho_id_mantle = 1
# T_rho_args_mantle = [300., 0.]
# mat_id_atm = 102
# T_rho_id_atm = 1
# T_rho_args_atm = [300., 0.]
# ucold_array_core = eos.load_ucold_array(100)
# ucold_array_mantle = eos.load_ucold_array(101)
# ucold_array_atm = eos.load_ucold_array(102)
# iterations = 20
# N_neig=48
# rho_cm = 8893.
# rho_ma = 2709.
# =============================================================================
