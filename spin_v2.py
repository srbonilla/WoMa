#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    mpirun -n 4 python spin.py
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

import spipgen_v2

G = 6.67408E-11
R_earth = 6371000
M_earth = 5.972E24

data = pd.read_csv("1layer_n10k.csv", header=0)

densities = np.array(data.rho)
radii = np.array(data.R)*R_earth

P_c = np.max(data.P)
P_s = np.min(data.P)
rho_c = np.max(data.rho)
rho_s = np.min(data.rho)

## linear + linear
#z_array_1 = np.arange(0, 1.1*np.max(radii), 1.1*np.max(radii)/100)
#z_array_2 = np.arange(0, z_array_1[10], z_array_1[10]/20)
#z_array_3 = np.arange(0, z_array_2[10], z_array_2[10]/20)

#z_array = np.hstack((z_array_3, z_array_2[10:], z_array_1[10:]))
"""
z_array_2 = np.arange(0, z_array_1[15], z_array_1[10]/40)
z_array = np.hstack((z_array_2, z_array_1[15:]))

z_array = np.unique(z_array)
z_array = np.sort(z_array)

"""
##linear scale
#r_array = np.arange(0, 1.5*np.max(radii), 1.5*np.max(radii)/120)
z_array = np.arange(0, 1.1*np.max(radii), 1.1*np.max(radii)/100)

rho, r_array, z_array, times = spipgen_v2.spin1layer(10, radii, densities, 4, 101, 1, [300,0], 
                                                     P_c, P_s, rho_c, rho_s, np.nan, z_array)

np.save('profile_parallel', rho)
np.save('r_array', r_array)
np.save('z_array', z_array)
np.save("exec_times", times)
"""
####### Analysis
rho_par = np.load('profile_parallel.npy')
r_array = np.load('r_array.npy')
z_array = np.load('z_array.npy')

#spipgen_plot.plotrho(rho_par[10][1:,:], r_array[1:]/R_earth, z_array/R_earth)


###### test
ucold_array = spipgen_v2._create_ucold_array(101)
I_array = np.load('I_array.npy')
    
rho_grid, r_array, z_array = spipgen_v2._rho0_grid(radii, densities)
rho = np.zeros((10 + 1, r_array.shape[0], z_array.shape[0]))
rho[0] = rho_grid

dS = spipgen_v2._dS(r_array, z_array)

V1 = spipgen_v2._fillV(rho[0], r_array, z_array, I_array, dS, 4)
rho[1] = spipgen_v2._fillrho(V1, r_array, z_array, P_c, P_s, rho_c, rho_s, 101, 1, [300,0], ucold_array)

spipgen_plot.plotrho(rho[1], r_array/R_earth, z_array/R_earth)

i = 1
j = 0
gradV = (V1[i + 1, j] - V1[i, 0])
gradP = -rho[0][i, j]*gradV
P = spipgen_v2.P_rho(rho[0][i, j], 101, 1, [300, 0])
spipgen_v2._find_rho(P - gradP, 101, 1, [300,0], rho_s - 10, rho[0][i, j], ucold_array)



spipgen_v2._Vg(r_array[0], z_array[1], rho[0], r_array, z_array, I_array, dS)

spipgen_v2._Vg(r_array[2], z_array[1], rho[0], r_array, z_array, I_array, dS)


spipgen_v2._Vg(0, R_earth, rho[0], r_array, z_array, I_array, dS)
-G*data.M[0]*M_earth/R_earth




"""

