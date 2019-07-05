#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:11:10 2019

@author: Sergio Ruiz-Bonilla
"""

###############################################################################
####################### Libraries and constants ###############################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import woma
import swift_io
import h5py

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
    
# Function to plot results for spining profile
    
def plot_spin_profile(spin_planet):
    
    sp = spin_planet
    
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].scatter(sp.A1_r/R_earth, sp.A1_rho, label = 'original', s = 0.5)
    ax[0].scatter(sp.A1_r_equator/R_earth, sp.A1_rho_equator, label = 'equatorial profile', s = 1)
    ax[0].scatter(sp.A1_r_pole/R_earth, sp.A1_rho_pole, label = 'polar profile', s = 1)
    ax[0].set_xlabel(r"$r$ [$R_{earth}$]")
    ax[0].set_ylabel(r"$\rho$ [$kg/m^3$]")
    ax[0].legend()
    
    
    r_array_coarse = np.linspace(0, np.max(sp.A1_r_equator), 100)
    z_array_coarse = np.linspace(0, np.max(sp.A1_r_pole), 100)
    rho_grid = np.zeros((r_array_coarse.shape[0], z_array_coarse.shape[0]))
    for i in range(rho_grid.shape[0]):
        radius = r_array_coarse[i]
        for j in range(rho_grid.shape[1]):
            z = z_array_coarse[j]
            rho_grid[i,j] = woma.rho_rz(radius, z,
                                        sp.A1_r_equator, sp.A1_rho_equator,
                                        sp.A1_r_pole, sp.A1_rho_pole)
    
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
    
# Example 1 layer
prof_pE = woma.Planet(
    name            = "prof_pE",
    A1_mat_layer    = ['Til_granite'],
    A1_T_rho_type   = [1],
    A1_T_rho_args   = [[None, 0.]],
    A1_R_layer      = [R_earth],
    M               = 0.8*M_earth,
    P_s             = 0,
    T_s             = 300
    )

prof_pE.M_max = M_earth

prof_pE.gen_prof_L1_fix_M_given_R()

plot_spherical_profile(prof_pE)

prof_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = prof_pE,
    Tw           = 3,
    R_e          = 1.3*R_earth,
    R_p          = 1.1*R_earth
    )

prof_sp.spin()

plot_spin_profile(prof_sp)

particles = woma.GenSpheroid(
    name        = 'picles_spin',
    spin_planet = prof_sp,
    N_particles = 1e5)

positions = np.array([particles.A1_x, particles.A1_y, particles.A1_z]).T
velocities = np.array([particles.A1_vx, particles.A1_vy, particles.A1_vz]).T

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '1layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, positions, velocities,
                             particles.A1_m, particles.A1_h,
                             particles.A1_rho, particles.A1_P, particles.A1_u,
                             particles.A1_picle_id, particles.A1_mat_id,
                             4*R_earth, swift_to_SI) 
    
#np.save('r_array', prof_sp.A1_r_equator)
#np.save('z_array', prof_sp.A1_r_pole)
#np.save('rho_e', prof_sp.A1_rho_equator)
#np.save('rho_p', prof_sp.A1_rho_pole)
    
# Example 2 layer

prof_pE = woma.Planet(
    name            = "prof_pE",
    A1_mat_layer    = ['Til_iron', 'Til_granite'],
    A1_T_rho_type   = [1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.]],
    A1_R_layer      = [None, R_earth],
    M               = M_earth,
    P_s             = 0,
    T_s             = 300
    )

prof_pE.gen_prof_L2_fix_R1_given_R_M()

plot_spherical_profile(prof_pE)

prof_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = prof_pE,
    Tw           = 2.6,
    R_e          = 1.45*R_earth,
    R_p          = 1.1*R_earth
    )

prof_sp.spin()

plot_spin_profile(prof_sp)

particles = woma.GenSpheroid(
    name        = 'picles_spin',
    spin_planet = prof_sp,
    N_particles = 1e5)

positions = np.array([particles.A1_picle_x, particles.A1_picle_y, particles.A1_picle_z]).T
velocities = np.array([particles.A1_picle_vx, particles.A1_picle_vy, particles.A1_picle_vz]).T

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '2layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, positions, velocities,
                             particles.A1_picle_m, particles.A1_picle_h,
                             particles.A1_picle_rho, particles.A1_picle_P, particles.A1_picle_u,
                             particles.A1_picle_id, particles.A1_picle_mat_id,
                             4*R_earth, swift_to_SI) 
