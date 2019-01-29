#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:48:32 2018

@author: sergio
"""
path = '/home/sergio/Documents/rotating_profiles/'
import os
os.chdir(path)

import sys
sys.path.append(path)

import time
import pandas as pd
import numpy as np
import rotf
import rotplot as rotp
"""
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
"""
"""
start_time = time.time()

data = pd.read_csv("1layer_n100.csv", header=0)

Rs = data.R[0]
rho_sph = np.array(data.rho)
rho_sph = np.flip(rho_sph, axis = 0)

N = rho_sph.shape[0]

rho_c = rho_sph[0]
rho_s = rho_sph[-1]

I_array = rotf.create_I_tab()
ucold_table = rotf.create_ucold_table()

end_time = time.time()
if rank == 0:
    print("Running time for creating I_array and u_cold_table:", end_time - start_time)
"""
"""
# method 1
rho0 = rotf.rho0(rho_sph, Rs)

rho = np.zeros((10, 2*N, N))
rho[0] = rho0
for i in range(1,10):
    V1 = rotf.fillV(rho0, Rs, 4, I_array)
    rho1 = rotf.fillrho(V1, Rs, rho_c, rho_s, rotf.granite, 300, 0, ucold_table)
    rho[i] = rho1
    rho0 = rho1

rotp.plotdiff(rho[9], rho[8], rho_s, 0.5)
rotp.plotdiff(rho[9], rho[8], rho_s, 0.5)
rotp.plotrho(rho[9], Rs)
"""
"""
#method 3
rho0 = rotf.rho0(rho_sph, Rs)

rho = np.zeros((10, 2*N, N))
rho[0] = rho0
for i in range(1,10):
    V1 = rotf.fillV(rho0, Rs, 4, I_array)
    rho1 = rotf.fillrho3(V1, Rs, rho[i - 1], rho_s, rotf.granite, 300, 0, ucold_table)
    rho[i] = rho1
    rho0 = rho1
 
rotp.plotdiff(rho[9], rho[8], rho_s, 0.5, "convergence_non-parallel.png")
rotp.plotdiff(rho[9], rho[0], rho_s, 0.5, "difference_orifinal-final.png")
rotp.plotrho(rho[9], Rs, "Final density.png")

"""

"""
#method 3 parallel
rho0 = rotf.rho0(rho_sph, Rs)

rho = np.zeros((10, 2*N, N))
rho[0] = rho0

if rank == 0: print("Starting convergence:")

for i in range(1,10):
    
    start_time = time.time()
    V1 = rotf.fillV_par(rho0, Rs, 4, I_array)
    end_time = time.time()
    print("fillV par. rank:", rank, "time:", end_time - start_time)
    
    start_time = time.time()
    rho1 = rotf.fillrho(V1, Rs, rho[i - 1], rho_s, rotf.granite, 300, 0, ucold_table)
    end_time = time.time()
    print("fillrho. rank:", rank, "time:", end_time - start_time)
    rho[i] = rho1
    rho0 = rho1
 
rotp.plotdiff(rho[9], rho[8], rho_s, 0.5, "convergence_parallel.png")
rotp.plotdiff(rho[9], rho[0], rho_s, 0.5, "difference_orifinal-final-parallel.png")
rotp.plotrho(rho[9], Rs, "Final density-parallel.png")

#end_time = time.time()
#print("Running time of convergence:", end_time - start_time)
"""
data = pd.read_csv("1layer_n100.csv", header=0)

Rs = data.R[0]
rho_sph = np.array(data.rho)
rho_sph = np.flip(rho_sph, axis = 0)

rho, times = rotf.main(10, rho_sph, Rs, rotf.granite, 4, 300, 0)

np.save('profile', rho)
np.save("exec_times", times)

