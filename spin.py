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

data = pd.read_csv("1layer_n100.csv", header=0)

Rs = data.R[0]
rho_sph = np.array(data.rho)
rho_sph = np.flip(rho_sph, axis = 0)

Tw = 4
K = 300
alpha = 0
rho, times = spipgen.spin(10, rho_sph, Rs, spipgen.granite, Tw, K, alpha)

np.save('profile', rho)
np.save("exec_times", times)

