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
import spipgen_v2

G = 6.67408E-11
R_earth = 6371000
M_earth = 5.972E24

data = pd.read_csv("1layer_n10k.csv", header=0)

iterations = 10
radii = np.flip(np.array(data.R))
densities = np.flip(np.array(data.rho))
Tw = 4
mat_id = np.ones(radii.shape[0])*100
T_rho_id = [1, None, None]
T_rho_params = [[300, 0], [], []]

r_grid_rc = np.arange(0, 1.5, 1.5/100)*R_earth
r_grid_z = np.arange(0, 1.2, 1.2/200)*R_earth

spipgen_v2._P_EoS_Till(1, 1000, 102)
spipgen.P_EoS(1, 1000, spipgen.water)
##
densities = np.array(data.rho)
radii = np.array(data.R)*R_earth

rho_grid, r_array, z_array = spipgen_v2._rho0_grid(radii, densities)

I_array = spipgen_v2._create_I_array()

dS = spipgen_v2._dS(r_array, z_array)

V_model = spipgen_v2._Vg(R_earth, 0, rho_grid, r_array, z_array, I_array, dS)

V_true = -8.725783e-01*M_earth/R_earth*G

print((V_model - V_true)/V_true)