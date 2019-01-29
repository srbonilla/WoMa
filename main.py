#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:48:32 2018

@author: sergio
"""
path = '/home/sergio/Documents/SpiPGen/'
import os
os.chdir(path)

import sys
sys.path.append(path)

import pandas as pd
import numpy as np
import spipgen_plot
import spipgen_func

data = pd.read_csv("1layer_n100.csv", header=0)

Rs = data.R[0]
rho_sph = np.array(data.rho)
rho_sph = np.flip(rho_sph, axis = 0)

rho, times = spipgen_func.main(10, rho_sph, Rs, spipgen_func.granite, 4, 300, 0)

np.save('profile', rho)
np.save("exec_times", times)

