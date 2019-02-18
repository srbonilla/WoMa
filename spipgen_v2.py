# -*- coding: utf-8 -*-
"""
Auxiliar functions and constants to compute a density grid with rotation.
"""

# ========
# Contents:
# ========
#   I   Libraries and constants
#   II  Functions
#   III Main

# //////////////////////////////////////////////////////////////////////////// #
#                        I. Libraries and constants                            #
# //////////////////////////////////////////////////////////////////////////// #


# Import necessary libraries
import numpy as np
import scipy.integrate
from numba import jit, njit
from mpi4py import MPI
import time
import sys
from scipy import interpolate

# Material constants for Tillotson EoS + material code and specific capacity (SI units)
iron = np.array([0.5, 1.5, 1.279E11, 1.05E11, 7860, 9.5E6, 1.42E6, 8.45E6, 5, 5, 0, 449])
granite = np.array([0.5, 1.3, 1.8E10, 1.8E10, 2700, 1.6E7, 3.5E6, 1.8E7, 5, 5, 1, 790])
water = np.array([0.5, 0.9, 2.0E10, 1.0E10, 1000, 2.0E6, 4.0E5, 2.0E6, 5, 5, 2, 4186])

# Global constants
G = 6.67408E-11;
R_earth = 6371000;
M_earth = 5.972E24;

# //////////////////////////////////////////////////////////////////////////// #
#                              II. Functions                                   #
# //////////////////////////////////////////////////////////////////////////// #

@jit(nopython=True)
def _P_EoS_Till(u, rho, mat_id):
    """
    Computes pressure for Tillotson EoS.
    
    Args:
        u (double)
            Internal energy (SI).
            
        rho (double) 
            Density (SI).
        
        mat_id ([int])
            Material id.
            
    Returns:
        P (double)
            Pressure (SI).
    """
    # Material constants for Tillotson EoS + material code and specific capacity (SI units)
    iron = np.array([0.5, 1.5, 1.279E11, 1.05E11, 7860, 9.5E6, 1.42E6, 8.45E6, 5, 5, 0, 449])
    granite = np.array([0.5, 1.3, 1.8E10, 1.8E10, 2700, 1.6E7, 3.5E6, 1.8E7, 5, 5, 1, 790])
    water = np.array([0.5, 0.9, 2.0E10, 1.0E10, 1000, 2.0E6, 4.0E5, 2.0E6, 5, 5, 2, 4186])
    if (mat_id == 100):
        material = iron
    elif (mat_id == 101):
        material = granite
    elif (mat_id == 102):
        material = water
    else:
        print("Material not implemented")
        return -1
        
    a = material[0]
    b = material[1]
    A = material[2]
    B = material[3]
    rho0 = material[4]
    u0 = material[5]
    u1 = material[6]
    u2 = material[7]
    alpha = material[8]
    beta = material[9]

    eta = rho/rho0
    mu = eta - 1.

    if (rho >= rho0 or u <= u1):
        P = (a + b*u0*eta**2/(u0*eta**2 + u))*u*rho + A*mu + B*mu**2
        
    elif (rho <= rho0 and u >= u2):
        P21 = a*u*rho
        P221 = b*u*rho*u0*eta**2/(u + u0*eta**2)
        P222 = A*mu*np.exp(-beta*(1/eta - 1))
        P23 = np.exp(-alpha*(1/eta - 1)**2)
        P = P21 + (P221 + P222)*P23
        
    else:
        P1 = (a + b*u0*pow(eta, 2)/(u + u0*pow(eta, 2)))*u*rho + A*mu + B*pow(mu, 2)
        P21 = a*u*rho
        P221 = b*u*rho*u0*eta**2/(u + u0*eta**2)
        P222 = A*mu*np.exp(-beta*(1/eta - 1))
        P23 = np.exp(-alpha*(1/eta - 1)**2)
        P2 = P21 + (P221 + P222)*P23
        P = ((u - u1)*P2 + (u2 - u)*P1)/(u2 - u1)
    
    return P

@jit(nopython=True)
def P_EoS(u, rho, mat_id):
    """
    Computes pressure for Tillotson EoS.
    
    Args:
        u (double)
            Internal energy (SI).
            
        rho (double) 
            Density (SI).
        
        mat_id ([int])
            Material id.
            
    Returns:
        P (double)
            Pressure (SI).
    """
    if (mat_id == 100):
        return _P_EoS_Till(u, rho, mat_id)
    elif (mat_id == 101):
        return _P_EoS_Till(u, rho, mat_id)
    elif (mat_id == 102):
        return _P_EoS_Till(u, rho, mat_id)
    else:
        print("Material not implemented")
        return -1
    
@jit(nopython=True)
def ucold(rho, material, N):
    """
    Computes internal energy cold.
    
    Args:          
        rho (float) 
            Density (SI).
        
        material ([float])
            Material constants (SI).
            
        N (int)
            Number of subdivisions for the numerical integral.
            
    Returns:
        uc (float)
            Cold internal energy (SI).
    """

    rho0 = material[4]
    drho = (rho - rho0)/N
    x = rho0
    uc = 0

    for j in range(N):
        x += drho
        uc += P_EoS(uc, x, material)*drho/x**2

    return uc

@jit(nopython=True)
def T_of_rho(rho, K, alpha):
    """
    Computes temperature given density (T = K*rho**alpha).
    
    Args:
        rho (float)
            Density (SI).
            
        K, alpha (float)
            Parameters of the equation (SI).
            
    Returns:
        K*rho**alpha (float)
            
    """
    return K*rho**alpha

# Compute Tillotson P, given rho. Also needed: material, K and alpha (from the relation T = K*rho**alpha)
@jit(nopython=True)
def P_rho(rho, material, K, alpha):
    """
    Computes pressure using Tillotson EoS, and
    internal energy = internal energy cold + c*Temperature 
    (which depends on rho and the relation between temperature and density).
    
    Args:          
        rho (float) 
            Density (SI).
        
        material ([float])
            Material constants (SI).
            
        K, alpha (float)
            parameters from the relation between density and temperature
            T = K*rho**alpha.
            
    Returns:
        P (float)
            Pressure (SI).
    """
    
    N = 10000
    c = material[11]
    u = ucold(rho, material, N) + c*T_of_rho(rho, K, alpha)
    P = P_EoS(u, rho, material)

    return P

@jit(nopython=True)
def _create_ucold_table():
    """
    Computes values of the cold internal energy and stores it to save 
    computation time in future calculations.
            
    Returns:
        ucold_table ([[float]])
            Pressure (SI).
    """

    nrow = 10000
    ucold_table = np.zeros((nrow,3))
    rho_min = 100
    rho_max = 100000
    Nucold = 10000

    rho = rho_min
    drho = (rho_max - rho_min)/(nrow - 1)
    
    # material 0: iron
    rho = rho_min
    for i in range(nrow):
        ucold_table[i,0] = ucold(rho, iron, Nucold)
        rho = rho + drho
    
    # material 1: granite
    rho = rho_min
    for i in range(nrow):
        ucold_table[i,1] = ucold(rho, granite, Nucold)
        rho = rho + drho
    
    # material 2: water
    rho = rho_min
    for i in range(nrow):
        ucold_table[i,2] = ucold(rho, water, Nucold)
        rho = rho + drho
    
    return ucold_table

@jit(nopython=True)
def _ucold_tab(rho, material, ucold_table):
    """
    Fast computation of cold internal energy using the table previously
    computed.
    
    Args:
        rho (float):
            Density (SI).
            
        material ([float]):
            Material constants (SI).
            
        ucold_table ([[float]])
            Precomputed values of cold internal energy
            with function create_ucold_table() (SI).
            
    Returns:
        interpolation (float):
            cold internal energy (SI).
    """

    nrow = ucold_table.shape[0]
    rho_min = 100
    rho_max = 100000

    material_code = int(material[10])

    drho = (rho_max - rho_min)/(nrow - 1)

    a = int(((rho - rho_min)/drho))
    b = a + 1

    if a >= 0 and a < (nrow - 1):
        interpolation = ucold_table[a,material_code]
        interpolation += ((ucold_table[b,material_code] - ucold_table[a,material_code])/drho)*(rho - rho_min - a*drho)

    elif rho < rho_min:
        interpolation = ucold(rho, material, 10000)
    else:
        interpolation = ucold_table[int(nrow - 1),material_code]
        interpolation += ((ucold_table[int(nrow - 1),material_code] - ucold_table[int(nrow) - 2][material_code])/drho)*(rho - rho_max)

    return interpolation

@jit(nopython=True)
def _find_rho(Ps, material, K, alpha, rho0, rho1, ucold_table):
    """
    Root finder of the density for Tillotson EoS using 
    tabulated values of cold internal energy
    
    Args:
        Ps (float):
            Pressure (SI).
            
        material ([float]):
            Material constants (SI).
        
        K, alpha (float):
            Parameters from the relation between density and temperature
            T = K*rho**alpha.
            
        rho0 (float):
            Lower bound for where to look the root (SI).
            
        rho1 (float):
            Upper bound for where to look the root (SI).
        
        ucold_table ([[float]])
            Precomputed values of cold internal energy
            with function create_ucold_table() (SI).
            
    Returns:
        rho2 (float):
            Value of the density which satisfies P(u(rho), rho) = 0 
            (SI).
    """

    c = material[11]
    rho2 = (rho0 + rho1)/2
    tolerance = 1E-7

    f0 = Ps - P_EoS((_ucold_tab(rho0, material, ucold_table) + c*T_of_rho(rho0, K, alpha)), rho0, material)
    f1 = Ps - P_EoS((_ucold_tab(rho1, material, ucold_table) + c*T_of_rho(rho1, K, alpha)), rho1, material)

    if f0*f1 > 0:
        #print("Cannot find a 0 in the interval\n")
        rho2 = rho1;
    else:
        while np.abs(rho1 - rho0) > tolerance:
            f0 = Ps - P_EoS((_ucold_tab(rho0, material, ucold_table) + c*T_of_rho(rho0, K, alpha)), rho0, material)
            f1 = Ps - P_EoS((_ucold_tab(rho1, material, ucold_table) + c*T_of_rho(rho1, K, alpha)), rho1, material)
            f2 = Ps - P_EoS((_ucold_tab(rho2, material, ucold_table) + c*T_of_rho(rho2, K, alpha)), rho2, material)

            if f0*f2 > 0:
                rho0 = rho2
            elif f0*f2 < 0: 
                rho1 = rho2
            else:
                return rho2

            rho2 = (rho0 + rho1)/2

    return rho2;

# Especial functions for the initial spherical case

@jit(nopython=True)
def rho_rz_sph(rc, z, rho_sph, Rs):
    """
    Computes density given any cylindrical coordinates, and a particular
    radial density profile.
    
    Args:
        rc (float):
            Cylindrical radii (distance from z axis) (R_earth).
            
        z (float):
            z coordinate (R_earth).
            
        rho_sph ([float]):
            Radial density profile (SI). First component must be 
            central density, and hence the last component must be
            the surface density.
            
        Rs (float):
            Length which covers the radial density profile (R_earth).
            
    Returns:
        rho (float):
            Interoplated density (SI).
    """
    N = rho_sph.shape[0]
    r = np.sqrt(rc**2 + z**2)
    dr = Rs / (N - 1)
    rho = 0.
    
    if r >= Rs: 
        return 0.
    else:
        alpha = int(r/dr)
        rho = rho_sph[alpha] + (rho_sph[alpha + 1] - rho_sph[alpha])*(r - alpha*dr)/dr
        
    return rho

# Compute rho0 given the spherical radii profile (from high to low as in function before)
# Rs in Earth radii units
@njit
def rho0(rho_sph, Rs):
    """
    Creates 2-d array density profile out of a 1-d array radial
    density profile (spherical).
    
    Args:            
        rho_sph ([float]):
            Radial density profile (SI). First component must be 
            central density, and hence the last component must be
            the surface density.
            
        Rs (float):
            Length which covers the radial density profile (R_earth).
            
    Returns:
        rho0 ([[float]]):
            2-d spherical density profile (SI).
    """
    
    N = rho_sph.shape[0]
    rho_grid = np.zeros((2*N, N))
    dr = Rs/(N - 1)

    for i in range(rho_grid.shape[0]):
        for j in range(rho_grid.shape[1]):
            rho_grid[i,j] = rho_rz_sph(i*dr, j*dr, rho_sph, Rs)
            
    return rho_grid
     
def _create_I_tab():
    """
    Tabulates the integral respect to angle phi' inside the 3-d
    integral of the gravitational potential.
    
    Returns:
        I_array ([float]):
            Values of the integral for gamma from 0 to 1.
    """
    
    f = lambda x, gamma: 1/np.sqrt(1 - gamma*np.cos(x))
    prec = 1e5

    gamma = np.arange(0, 1 + 1/prec, 1/prec)
    I_array = np.zeros((gamma.shape[0],))
    for i in range(gamma.shape[0]):
        I_array[i] = scipy.integrate.quad(f, 0, 2*np.pi, args=(gamma[i] - 1e-10,))[0]
        
    return I_array

@jit(nopython=True)
def _I_tab(gamma, I_array):
    """
    Computes the integral respect to angle phi inside the 3-d
    integral of the gravitational potential.
    
    Args:
        gamma (float):
            Parameter depending on r and z, of which the integral depends.
            
        I_array ([float]):
            Values of the integral for gamma from 0 to 1.
    
    Returns:
        I_array[i] (float):
            Values of the integral for the specified gamma.
    """
    
    prec = 1e5
    i = int((gamma)*prec)
    return I_array[i]

# Computes the gravitational potential given a density grid at some point rc, z
@jit(nopython=True)
def _Vg(rc, z, rho_grid, Rs, I_array):
    """
    Computes the gravitational potential given a 2-d density grid at any point
    rc, z.
    
    Args:
        rc (float):
            Cylindrical radii (distance from z axis) (R_earth).
            
        z (float):
            z coordinate (R_earth).
            
        rho_grid ([[float]]):
            2-d density grid (SI) of a planet.
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
    Returns:
        
        V (float):
            Gravitational potential (SI).
        
            
    """
    
    N = rho_grid.shape[1]
    dr = Rs/(N - 1)
    Vsum = 0
    
    err = 1e-10
    
    for i in range(rho_grid.shape[0] - int(N/2)):
        rcp = i*dr
        for j in range(rho_grid.shape[1]):
            zp = j*dr
            
            a2 = rcp*rcp + zp*zp + rc*rc + z*z
            alpha = 1 - 2*z*zp/(a2 + err)
            beta = 2*rc*rcp/(a2 + err)
                
            gamma = beta/alpha 
            
            Vsum = Vsum + rho_grid[i,j]*rcp*_I_tab(gamma, I_array)/np.sqrt(a2 + err)/np.sqrt(alpha)
            
            #if j != 0:
            zp = -j*dr
            alpha = 1 - 2*z*zp/(a2 + err)
            gamma = beta/alpha      
            Vsum = Vsum + rho_grid[i,j]*rcp*_I_tab(gamma, I_array)/np.sqrt(a2 + err)/np.sqrt(alpha)
      
    V = -G*Vsum*dr*dr*R_earth*R_earth
    
    return V

@jit(nopython=True)
def _fillV(rho_grid, Rs, Tw, I_array):
    """
    Computes a 2-d potential grid given a 2-d density grid of a rotating planet.
    
    Args:
        rho_grid ([[float]]):
            2-d density grid (SI) of the planet.
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        Tw (float):
            Period of the planet (hours).
            
        I_array ([float]):
            Tabulated values of the integral of phi' for gamma from 0 to 1.
    
    Returns:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
    
    """
    N = rho_grid.shape[1]
    V_grid = np.zeros((2*N, N))
    #Tw in hours
    W = 2*np.pi/Tw/60/60
    dr = Rs/(N - 1)
    for i in range(V_grid.shape[0]):
        for j in range(V_grid.shape[1]):
            V_grid[i,j] = _Vg(i*dr, j*dr, rho_grid, Rs, I_array) - (1/2)*(W*i*dr*R_earth)**2
            
    return V_grid

# attempt 2, from center to surface
@jit(nopython=True)
def _fillrho(V_grid, Rs, prev_rho_grid, rho_s, material, K, alpha, ucold_table):
    """
    Update 2-d density grid given a 2-d potential grid of the planet.
    
    Args:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        prev_rho_grid ([[float]]):
            2-d density grid (SI) to be updated.
            
        rho_s (float):
            Surface density of the planet (SI).
            
        material ([float]):
            Material constants (SI).
            
        K, alpha (float):
            Parameters from the relation between density and temperature
            T = K*rho**alpha.
        
        ucold_table ([[float]]):
            Tabulated values of cold internal energy.
            
    Returns:
        rho_grid ([[float]]):
            Updated 2-d density grid (SI).
    """
    
    N = V_grid.shape[1]
    rho_grid = np.zeros((2*N, N))
    Ps = P_rho(rho_s, material, K, alpha)
    
    rho_grid[0,0] = prev_rho_grid[0,0]
    #rho_grid[1,0] = prev_rho_grid[1,0]
    #rho_grid[0,1] = prev_rho_grid[0,1]
    #rho_grid[1,1] = prev_rho_grid[1,1]
    ##############
    gradV = -(V_grid[0, 1] - V_grid[0, 0])
    gradP = rho_grid[0, 0]*gradV
    P = gradP + P_rho(rho_grid[0, 0], material, K, alpha)
    rho_grid[0,1] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[0,0], ucold_table)
    
    gradV = -(V_grid[1, 0] - V_grid[0, 0])
    gradP = rho_grid[0, 0]*gradV
    P = gradP + P_rho(rho_grid[0, 0], material, K, alpha)
    rho_grid[1,0] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[0,0], ucold_table)
    
    gradV = -(V_grid[1, 1] - V_grid[1, 0])
    gradP = rho_grid[1, 0]*gradV
    P = gradP + P_rho(rho_grid[1, 0], material, K, alpha)
    rho_grid[1,1] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[0,0], ucold_table)
    
    ##############
    
    # fill polar radii
    for i in range(2):
        for j in range(1, rho_grid.shape[1] - 1):
            gradV = -(V_grid[i, j + 1] - V_grid[i, j - 1])
            gradP = rho_grid[i, j]*gradV
            P = gradP + P_rho(rho_grid[i, j - 1], material, K, alpha)
            if P >= Ps:
                rho_grid[i, j + 1] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[i,j], ucold_table)
            else:
                rho_grid[i, j + 1] = 0
                break       
        
    # fill the rest
    for j in range(0, rho_grid.shape[1]):
        for i in range(1, rho_grid.shape[0] - 1):
            if rho_grid[i - 1, j] == 0:
                break
            
            if (i >= 2 and i < 2*N - 2):
                gradV = -(- V_grid[i+2, j] + 8*V_grid[i+1,j] - 8*V_grid[i-1,j] + V_grid[i-2,j])/6
            else:    
                gradV = -(V_grid[i + 1, j] - V_grid[i - 1, j])
                
            P = rho_grid[i, j]*gradV
            P = P + P_rho(rho_grid[i - 1, j], material, K, alpha)
            if P >= Ps:
                rho_grid[i + 1, j] = _find_rho(P, material, K, alpha, rho_s - 10, rho_grid[i,j], ucold_table)
            else:
                rho_grid[i + 1, j] = 0
                break
       
    return rho_grid


# Compute V grid (parallelized) ###########
@jit(nopython=True)
def _fillV_rows(rho_grid, Rs, I_array, lower, upper, rank, subsize, N, W, dr):
    """
    Partially complete the computation of the total potential.
    
    Args:
        rho_grid ([[float]]):
            2-d density grid (SI) of the planet.
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        I_array ([float]):
            Tabulated values of the integral of phi' for gamma from 0 to 1.
            
        lower (int):
            Row from which to start the computation.
            
        upper (int):
            Row from which to end the computation (not included).
            
        rank (int):
            Identifier of the processor.
            
        subsize (int):
            int(N/size).
            
        N (int):
            Number of integration steps from the initial density profile.
            
        W (float):
            Angular velocity (SI).
            
        dr (float):
            Rs/(N - 1).
            
    Returns:
        rows ([[float]]):
            2-d potential grid (SI) for the rows specified.
        
    """
    
    rows = np.zeros((2*N, upper - lower))
    for i in range(2*N):
        for j in range(upper - lower):
            rows[i, j] = _Vg(i*dr, (rank*subsize + j)*dr, rho_grid, Rs, I_array) - (1/2)*(W*i*dr*R_earth)**2
    
    return rows

def _fillV_par(rho_grid, Rs, Tw, I_array):
    """
    Parallel computation of 2-d potential grid given 
    a 2-d density grid of a rotating planet.
    
    Args:
        rho_grid ([[float]]):
            2-d density grid (SI) of the planet.
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        Tw (float):
            Period of the planet (hours).
            
        I_array ([float]):
            Tabulated values of the integral of phi' for gamma from 0 to 1.
    
    Returns:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
    
    """

    # parallel set-up
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    N = rho_grid.shape[1]
    W = 2*np.pi/Tw/60/60
    dr = Rs/(N - 1)
    
    # partitioning the work
    subsize = int(N/size)
    partition = [0, subsize]
    
    for i in range(2, size):
        partition.append(i*subsize)
    
    partition.append(N)
    partition = np.array(partition)
    
    # send the work to each precessor
    lower = partition[rank]
    upper = partition[rank + 1]
        
    V_subgrid = _fillV_rows(rho_grid, Rs, I_array, lower, upper, rank, subsize, N, W, dr)
    #V_subgrid = np.flip(V_subgrid, 1)
    
    V_raw = comm.gather(V_subgrid, root=0)
    V_grid = np.zeros((2*N, N))
    
    if rank == 0:
        for I in range(size):
            V_grid[:, partition[I]:partition[I + 1]] = V_raw[I]
        #V_grid = np.flip(V_grid, 1)
        
    V_grid = comm.bcast(V_grid, root=0)
    #V_grid = V_grid.reshape((2*N, N))
    
    return V_grid

###############################################################################
    
def spin(iterations, rho_sph, Rs, material, Tw, K, alpha):
    """
    Spining of a radial planetary profile
    
    Args:
        iterations (int):
            Number of iterations to compute.
            
        rho_sph ([float]):
            Initial density radial profile to start with (SI).
            
        Rs (float):
            Initial radius of the planet (R_earth).
            
        Tw (float):
            Period of the planet (hours).
            
        K, alpha (float):
            Parameters from the relation between density and temperature
            T = K*rho**alpha.
    
    Returns:
        V_grid ([[float]]):
            2-d grid of the total potential (SI).
    """
    
    times = np.zeros((iterations, 2))
    
    start = time.time()
    
    N = rho_sph.shape[0]
    rho_s = rho_sph[-1]
    I_array = _create_I_tab()
    ucold_table = _create_ucold_table()

    rho = np.zeros((iterations + 1, 2*N, N))
    rho[0] = rho0(rho_sph, Rs)
    
    end = time.time()
    print("Running time for creating I_array and u_cold_table:", end - start)
    
    for i in range(1, iterations + 1):
        
        start = time.time()
        V1 = _fillV_par(rho[i - 1], Rs, Tw, I_array)
        end = time.time()
        times[i - 1, 0] = end - start
        start = time.time()
        rho[i] = _fillrho(V1, Rs, rho[i - 1], rho_s, material, K, alpha, ucold_table)
        end = time.time()
        times[i - 1, 1] = end - start
        
    return rho, times