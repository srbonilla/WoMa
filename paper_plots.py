#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:39:38 2019

@author: sergio
"""

import numpy as np
import matplotlib.pyplot as plt
import woma
import utils_spin as us
import matplotlib as mpl
import L1_spin
import L2_spin


font_size   = 20
params      = { "backend"               : "ps",
                #"text.latex.preamble"   : [r"\usepackage{gensymb}"
                #                           # Avoid the cosma error
                #                           r"\newcommand{\mathdefault}{}"],
                "axes.labelsize"        : font_size,
                "axes.titlesize"        : font_size,
                "font.size"             : font_size,
                "legend.fontsize"       : font_size - 4,
                "xtick.labelsize"       : font_size,
                "ytick.labelsize"       : font_size,
                "text.usetex"           : True,
                "figure.figsize"        : [7, 7],
                "font.family"           : "serif",
                # "font.family"           : "sans-serif",
                # "font.sans-serif"       : "Open Sans",
                "savefig.dpi"           : 100,
                "legend.framealpha"     : 0.7,
                "lines.linewidth"       : 1.7,
               }
mpl.rcParams.update(params)

R_earth = 6371000
M_earth = 5.972E24

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
            rho_grid[i,j] = us.rho_rz(radius, z,
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
l1_test = woma.Planet(
    name            = "prof_pE",
    A1_mat_layer    = ['Til_granite'],
    A1_T_rho_type   = [1],
    A1_T_rho_args   = [[None, 0.]],
    A1_R_layer      = [R_earth],
    M               = 0.8*M_earth,
    P_s             = 0,
    T_s             = 300
    )

l1_test.M_max = M_earth

l1_test.gen_prof_L1_fix_M_given_R()

#plot_spherical_profile(l1_test)

l1_test_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = l1_test,
    Tw           = 3,
    R_e          = 1.3*R_earth,
    R_p          = 1.1*R_earth
    )

l1_test_sp.spin()

#plot_spin_profile(l1_test_sp)

# Example 2 layer

l2_test = woma.Planet(
    name            = "prof_pE",
    A1_mat_layer    = ['SESAME_iron', 'SESAME_basalt'],
    A1_T_rho_type   = [1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.]],
    A1_R_layer      = [None, R_earth],
    M               = M_earth,
    P_s             = 1e5,
    T_s             = 300
    )

l2_test.gen_prof_L2_fix_R1_given_R_M()

#plot_spherical_profile(l2_test)

l2_test_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = l2_test,
    Tw           = 2.6,
    R_e          = 1.45*R_earth,
    R_p          = 1.1*R_earth
    )

l2_test_sp.spin()

#plot_spin_profile(l2_test_sp)

#############

cmap = plt.get_cmap('copper')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
color_copper = colors[15]

cmap = plt.get_cmap('BrBG')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
color_ocean = colors[13]

cmap = plt.get_cmap('Dark2')
color_purple = cmap(2)
color_ocre = cmap(5)
color_green = cmap(0)

color_1 = color_purple
color_2 = color_ocre

def plot_spherical_prof(l1_test, l2_test):
    
    
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(l1_test.A1_r[l1_test.A1_rho > 0]/R_earth, l1_test.A1_rho[l1_test.A1_rho > 0],
            label='1 layer test', color=color_1)
    ax.plot(l2_test.A1_r[l2_test.A1_rho > 0]/R_earth, l2_test.A1_rho[l2_test.A1_rho > 0],
            label='2 layer test', color=color_2)
    ax.set_xlabel(r"$r$ [R$_\oplus]$")
    ax.set_ylabel(r"$\rho$ [kg m$^{-3}]$")
    plt.tight_layout()
    ax.legend()
    plt.xlim((0,1))
    fig.savefig('Fig1.pdf')
    plt.show()
    
def plot_spin_prof(l1_test_sp, l2_test_sp):
    
    fig, ax = plt.subplots(2, 1, figsize=(5,7), sharex=True)
    ax[0].plot(l1_test_sp.A1_r_pole[l1_test_sp.A1_rho_pole > 0]/R_earth,
            l1_test_sp.A1_rho_pole[l1_test_sp.A1_rho_pole > 0], label='1 layer test', color=color_1)
    ax[0].plot(l2_test_sp.A1_r_pole[l2_test_sp.A1_rho_pole > 0]/R_earth,
            l2_test_sp.A1_rho_pole[l2_test_sp.A1_rho_pole > 0], label='2 layer test', color=color_2)
    ax[1].plot(l1_test_sp.A1_r_equator[l1_test_sp.A1_rho_equator > 0]/R_earth,
            l1_test_sp.A1_rho_equator[l1_test_sp.A1_rho_equator > 0], label='1 layer test', color=color_1)
    ax[1].plot(l2_test_sp.A1_r_equator[l2_test_sp.A1_rho_equator > 0]/R_earth,
            l2_test_sp.A1_rho_equator[l2_test_sp.A1_rho_equator > 0], label='2 layer test', color=color_2)
    ax[0].set_xlabel(r"$z$ $[R_{earth}]$")
    ax[0].set_ylabel(r"$\rho$ $[Kg/m^3]$")
    ax[1].set_xlabel(r"$r_{xy}$ $[R_{earth}]$")
    ax[1].set_ylabel(r"$\rho$ $[Kg/m^3]$")
    plt.tight_layout()
    ax[0].legend()
    ax[1].legend()
    fig.savefig('Fig2.pdf')
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    ax.plot(l1_test_sp.A1_r_equator[l1_test_sp.A1_rho_equator > 0]/R_earth,
            l1_test_sp.A1_rho_equator[l1_test_sp.A1_rho_equator > 0],
            label='1 layer: Eq. profile', color=color_1)
    ax.plot(l1_test_sp.A1_r_pole[l1_test_sp.A1_rho_pole > 0]/R_earth,
            l1_test_sp.A1_rho_pole[l1_test_sp.A1_rho_pole > 0], label='1 layer: Polar profile', color=color_1,
            linestyle='dashed')
    ax.plot(l2_test_sp.A1_r_equator[l2_test_sp.A1_rho_equator > 0]/R_earth,
            l2_test_sp.A1_rho_equator[l2_test_sp.A1_rho_equator > 0],
            label='2 layer: Eq. profile', color=color_2)
    ax.plot(l2_test_sp.A1_r_pole[l2_test_sp.A1_rho_pole > 0]/R_earth,
            l2_test_sp.A1_rho_pole[l2_test_sp.A1_rho_pole > 0], label='2 layer: Polar profile', color=color_2,
            linestyle='dashed')
    ax.set_xlabel(r"Distance from the centre [R$_\oplus]$")
    ax.set_ylabel(r"$\rho$ [kg m$^{-3}$]")
    ax.set_xlim((0,1.4))
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig('Fig2.pdf')
    plt.show()
    
def plot_convergence_spin(l1_test_sp, l2_test_sp):
    
    P_c   = np.max(l1_test_sp.A1_P)
    P_s   = np.min(l1_test_sp.A1_P)
    rho_c = np.max(l1_test_sp.A1_rho)
    rho_s = np.min(l1_test_sp.A1_rho)

    r_array_1     = np.linspace(0, l1_test_sp.R_e, l1_test_sp.num_prof)
    z_array_1     = np.linspace(0, l1_test_sp.R_p, l1_test_sp.num_prof)
    
    iterations = 13

    if l1_test_sp.num_layer == 1:
        # Check for necessary input
        assert(l1_test_sp.A1_mat_id_layer[0] is not None)
        assert(l1_test_sp.A1_T_rho_type[0] is not None)

        profile_e_1, profile_p_1 = \
            L1_spin.spin1layer(iterations, r_array_1, z_array_1,
                               l1_test_sp.A1_r, l1_test_sp.A1_rho, l1_test_sp.Tw,
                               P_c, P_s, rho_c, rho_s,
                               l1_test_sp.A1_mat_id_layer[0],
                               l1_test_sp.A1_T_rho_type[0],
                               l1_test_sp.A1_T_rho_args[0]
                               )
    ##########
    P_c   = np.max(l2_test_sp.A1_P)
    P_s   = np.min(l2_test_sp.A1_P)
    rho_c = np.max(l2_test_sp.A1_rho)
    rho_s = np.min(l2_test_sp.A1_rho)

    r_array_2     = np.linspace(0, l2_test_sp.R_e, l2_test_sp.num_prof)
    z_array_2     = np.linspace(0, l2_test_sp.R_p, l2_test_sp.num_prof)
            
    if l2_test_sp.num_layer == 2:
        # Check for necessary input
        assert(l2_test_sp.A1_mat_id_layer[0] is not None)
        assert(l2_test_sp.A1_T_rho_type[0] is not None)
        assert(l2_test_sp.A1_mat_id_layer[1] is not None)
        assert(l2_test_sp.A1_T_rho_type[1] is not None)

        a = np.min(l2_test_sp.A1_P[l2_test_sp.A1_r <= l2_test_sp.A1_R_layer[0]])
        b = np.max(l2_test_sp.A1_P[l2_test_sp.A1_r >= l2_test_sp.A1_R_layer[0]])
        P_boundary = 0.5*(a + b)

        l2_test_sp.P_R1 = P_boundary

        profile_e_2, profile_p_2 = \
            L2_spin.spin2layer(iterations, r_array_2, z_array_2,
                       l2_test_sp.A1_r, l2_test_sp.A1_rho, l2_test_sp.Tw,
                       P_c, P_boundary, P_s,
                       rho_c, rho_s,
                       l2_test_sp.A1_mat_id_layer[0], l2_test_sp.A1_T_rho_type[0], l2_test_sp.A1_T_rho_args[0],
                       l2_test_sp.A1_mat_id_layer[1], l2_test_sp.A1_T_rho_type[1], l2_test_sp.A1_T_rho_args[1]
                       )
            
    # Compute Re Rp per iteration
    Re_1 = []
    Rp_1 = []
    Re_2 = []
    Rp_2 = []
    for i in range(len(profile_e_1)):
        Re_1.append(max(r_array_1[profile_e_1[i]>0]))
        Re_2.append(max(r_array_2[profile_e_2[i]>0]))
        Rp_1.append(max(z_array_1[profile_p_1[i]>0]))
        Rp_2.append(max(z_array_2[profile_p_2[i]>0]))
        
    Re_1 = np.array(Re_1)
    Rp_1 = np.array(Rp_1)
    Re_2 = np.array(Re_2)
    Rp_2 = np.array(Rp_2)
    
    e_1 = np.sqrt(1 - Rp_1*Rp_1/Re_1/Re_1)
    e_2 = np.sqrt(1 - Rp_2*Rp_2/Re_2/Re_2)
    
    e_1[0] = 0
    e_2[0] = 0
    
    de_1 = np.zeros_like(e_1)
    de_2 = np.zeros_like(e_2)
    
    for i in range(1, len(de_1)):
        de_1[i] = (e_1[i] - e_1[i - 1])/e_1[i]
        de_2[i] = (e_2[i] - e_2[i - 1])/e_2[i]
        
    de_1[0] = np.nan
    de_2[0] = np.nan
    
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [3, 1]})
    ax[0].scatter(range(len(profile_e_1)), e_1, label='1 layer test', color=color_1, zorder=3)
    ax[0].scatter(range(len(profile_e_1)), e_2, label='2 layer test', color=color_2)
    ax[0].set_ylabel(r"$e$")
    ax[0].set_xlabel(r"Iteration")
    #ax[0].set_xticks(np.arange(0, iterations + 1, step=5))
    ax[0].set_ylim((0.58, 0.73))
    ax[0].legend()
    ax[1].scatter(range(len(profile_e_1)), de_1, label='1 layer test', color=color_1, zorder=3)
    ax[1].scatter(range(len(profile_e_1)), de_2, label='2 layer test', color=color_2)
    ax[1].set_ylabel(r"$\Delta e_i$")
    ax[1].set_xlabel(r"Iteration")
    ax[1].set_xticks(np.arange(0, iterations + 1, step=5))
    ax[1].set_ylim((-0.01, 0.06))
    ax[1].axhline(zorder=-1, color='black', linewidth=1)
    #ax[1].legend()
    #ax.set_xticks((0, 5, 10, 15, 20), (r"0", r"5", r"10", r"15", r"20"))
    plt.tight_layout()
    fig.savefig('Fig3.pdf')
    plt.show()
    
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, 6)]
    
    fig, ax = plt.subplots(2, 1, figsize=(7,12), sharex=True, gridspec_kw={'hspace': 0})
    for i in range(10):
        if i%2 == 0:
            ax[0].plot(r_array_1[profile_e_1[i] > 0]/R_earth, profile_e_1[i][profile_e_1[i] > 0],
                       label=str(i), color=colors[int(i/2)])
            ax[1].plot(r_array_2[profile_e_2[i] > 0]/R_earth, profile_e_2[i][profile_e_2[i] > 0],
                       label=str(i), color=colors[int(i/2)])
    ax[0].set_ylabel(r"$\rho$ [kg m$^{-3}$]")
    ax[1].set_ylabel(r"$\rho$ [kg m$^{-3}$]")
    ax[1].set_xlabel(r"$r_{xy}$ [R$_\oplus$]")
    ax[0].legend(title='Iteration')
    ax[1].legend(title='Iteration')
    ax[0].set_xlim((0,1.3))
    plt.tight_layout()
    fig.savefig('Fig4.pdf')
    plt.show()
    
    cmap = plt.get_cmap('copper')
    colors_1 = [cmap(i) for i in np.linspace(0, 1, 5)]
    #colors_1 = colors_1[::-1]
    cmap = plt.get_cmap('Oranges')
    colors_2 = [cmap(i) for i in np.linspace(0, 1, 7)]
    colors_2 = colors_2[::-1]
    colors_2 = colors_1
    
    fig, ax = plt.subplots(1, 1)
    for i in range(10):
        if i%2 == 0:
            ax.plot(r_array_2[profile_e_2[i] > 0]/R_earth, profile_e_2[i][profile_e_2[i] > 0],
                    label=str(i), color=colors_2[int(i/2)], linestyle='-')
        
    ax.set_ylabel(r"$\rho$ [kg m$^{-3}$]")
    ax.set_xlabel(r"$r_{xy}$ [R$_\oplus$]")
    #ax.legend(title='Iteration')
    for i in range(10):
        if i%2 == 0:
            ax.plot(r_array_1[profile_e_1[i] > 0]/R_earth, profile_e_1[i][profile_e_1[i] > 0],
                    label=str(i), color=colors_1[int(i/2)], linestyle='-.')
            
    
    lines = ax.get_lines()
    legend1 = plt.legend([lines[i] for i in [0,1,2]], ["algo1", "algo2", "algo3"], loc=1)
    legend2 = plt.legend([lines[i] for i in [0,3,6]], ["1", "2", "3"], loc=4)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    ax.set_xlim((0,1.3))
    plt.tight_layout()
    fig.savefig('Fig4.pdf')
    plt.show()
        