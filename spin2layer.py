#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    mpirun -n 4 python spin2layer.py
"""
path = '/home/sergio/Documents/SpiPGen/'
import os
os.chdir(path)
import sys
sys.path.append(path)

import pandas as pd
import numpy as np
import spipgen
import spipgen_plot
import matplotlib.pyplot as plt

import spipgen_v2

G = 6.67408E-11
R_earth = 6371000
M_earth = 5.972E24

data = pd.read_csv("2layer.csv", header=0)

densities = np.array(data.rho)
radii = np.array(data.R)*R_earth

P_c = np.median(np.sort(data.P)[-100:])
P_s = np.min(data.P)
P_i = (195619882720.038 + 195702367184.472)/2.

rho_c = np.max(data.rho)
rho_s = np.min(data.rho)

##linear scale
r_array = np.arange(0, 1.4*np.max(radii), 1.4*np.max(radii)/100)
z_array = np.arange(0, 1.1*np.max(radii), 1.1*np.max(radii)/100)

mat_id_core = 100
T_rho_id_core = 1
T_rho_args_core = [300,0]
mat_id_mantle = 101
T_rho_id_mantle = 1
T_rho_args_mantle = [300,0]

r_array = np.arange(0, 1.4*np.max(radii), 1.4*np.max(radii)/200)
z_array = np.arange(0, 1.1*np.max(radii), 1.1*np.max(radii)/200)

rho, r_array, z_array, times = spipgen_v2.spin2layer(10, radii, densities, 4,
                                                     mat_id_core, T_rho_id_core, T_rho_args_core,
                                                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                                     P_c, P_i, P_s, rho_c, rho_s, r_array, z_array)

np.save('profile_parallel_2l', rho)
np.save('r_array_2l', r_array)
np.save('z_array_2l', z_array)
np.save("exec_times_2l", times)

"""
#test
mat_id_core = 100
T_rho_id_core = 1
T_rho_args_core = [300,0]
mat_id_mantle = 101
T_rho_id_mantle = 1
T_rho_args_mantle = [300,0]

I_array = np.load('I_array.npy')
ucold_array_core = np.load('ucold_array_100.npy')
ucold_array_mantle = np.load('ucold_array_101.npy')

r_array = np.arange(0, 1.4*np.max(radii), 1.4*np.max(radii)/100)
z_array = np.arange(0, 1.1*np.max(radii), 1.1*np.max(radii)/100)
    
rho_grid, r_array, z_array = spipgen_v2._rho0_grid(radii, densities, r_array, z_array)
rho = np.zeros((10 + 1, r_array.shape[0], z_array.shape[0]))
rho[0] = rho_grid

dS = spipgen_v2._dS(r_array, z_array)

V1 = spipgen_v2._fillV_parallel(rho[0], r_array, z_array, I_array, dS, 4)
rho[1], P_grid = spipgen_v2._fillrho2(V1, r_array, z_array, P_c, P_i, P_s, rho_c, rho_i, rho_s,
                                    mat_id_core, T_rho_id_core, T_rho_args_core, ucold_array_core,
                                    mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle, ucold_array_mantle)

spipgen_plot.plotrho(rho[1], r_array, z_array)
"""

"""
#Analysis

rho_par = np.load('profile_parallel_2l.npy')
r_array = np.load('r_array_2l.npy')
z_array = np.load('z_array_2l.npy')
times = np.load('exec_times_2l.npy')

spipgen_plot.plotrho(rho_par[-1], r_array/R_earth, z_array/R_earth)
"""