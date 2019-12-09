#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:21:38 2019

@author: sergio
"""

import numpy as np
from numba import njit
import glob_vars as gv

# Utilities
def check_end(string, end):
    """ Check that a string ends with the required characters and append them
        if not.
    """
    if string[-len(end):] != end:
        string  += end

    return string

@njit
def find_index_and_interp(x, A1_x):
    """ Return the index and interpolation factor of a value in an array.

       Allows x outside A1_x. If so then intp will be < 0 or > 1.

       Args:
           x (float)
               The value to find.

           A1_x ([float])
               The array to search.

       Returns:
           idx (int)
               The index of the last array element smaller than the value.

               0               If x is below A1_x.
               len(A1_x) - 2   If x is above A1_x.

           intp (float)
               The interpolation factor for how far the values is from the
               indexed array value to the next.

               < 0     If x is below A1_x.
               > 1     If x is above A1_x.
    """
    idx = np.searchsorted(A1_x, x) - 1
    # Return error values if outside the array
    if idx == -1:
        idx = 0
    elif idx >= len(A1_x) - 1:
        idx = len(A1_x) - 2

    # Check for duplicate elements
    if A1_x[idx + 1] != A1_x[idx]:
        intp = (x - A1_x[idx]) / (A1_x[idx + 1] - A1_x[idx])
    else:
        intp = 1.
       
    return np.array([idx, intp])
    
def load_table_SESAME(Fp_table):
    """ Load and return the table file data.

        # header (five lines)
        num_rho  num_T
        A1_rho
        A1_T
        A2_u[0, 0]              A2_P[0, 0]      A2_c[0, 0]      A2_s[0, 0]
        A2_u[1, 0]              ...
        ...                     ...
        A2_u[num_rho, 0]        ...
        A2_u[0, 1]              ...
        ...                     ...
        A2_u[num_rho, 1]        ...
        ...                     ...
        A2_u[num_rho, num_u]    ...

        Args:
            Fp_table (str)
                The table file path.

        Returns:
            # num_rho, num_T (int)
            #     Number of densities and temperatures for the arrays.

            # rho_min, rho_max (float)
            #     Penultimate minimin and maximum values of the temperature
            #     array (K).

            # T_min, T_max (float)
            #     Penultimate minimin and maximum values of the density array
            #     (kg m^-3).

            # A1_rho, A1_T ([float])
            #     1D arrays of density (kg m^-3) and temperature (K).

            A2_u, A2_P, A2_s, # A2_c ([[float]])
                2D arrays of sp. int. energy (J kg^-1), pressure (Pa),
                # sp. entropy (J kg^-1 K^-1), and sound speed (m s^-1).

            A1_log_rho, A1_log_T ([float])
                1D arrays of natural logs of density (kg m^-3) and
                temperature (K).

            A2_log_u, # A2_log_P, A2_log_c ([[float]])
                2D arrays of natural logs of sp. int. energy (J kg^-1),
                pressure (Pa), sound speed (m s^-1).
    """
    # Load
    Fp_table    = check_end(Fp_table, ".txt")
    with open(Fp_table) as f:
        for i in range(5):
            f.readline()
        num_rho, num_T  = np.array(f.readline().split(), dtype=int)
        A2_u    = np.empty((num_rho, num_T))
        A2_P    = np.empty((num_rho, num_T))
        A2_c    = np.empty((num_rho, num_T))
        A2_s    = np.empty((num_rho, num_T))

        A1_rho  = np.array(f.readline().split(), dtype=float)
        A1_T    = np.array(f.readline().split(), dtype=float)

        for i_T in range(num_T):
            for i_rho in range(num_rho):
                A2_u[i_rho, i_T], A2_P[i_rho, i_T], A2_c[i_rho, i_T], \
                    A2_s[i_rho, i_T]    = np.array(f.readline().split(),
                                                   dtype=float)

    return A2_u, A2_P, A2_s, np.log(A1_rho), np.log(A1_T), np.log(A2_u)

# Load SESAME tables as global variables
(A2_u_SESAME_iron, A2_P_SESAME_iron, A2_s_SESAME_iron, A1_log_rho_SESAME_iron,
 A1_log_T_SESAME_iron, A2_log_u_SESAME_iron
 )  = load_table_SESAME(gv.Fp_SESAME_iron)
(A2_u_SESAME_basalt, A2_P_SESAME_basalt, A2_s_SESAME_basalt,
 A1_log_rho_SESAME_basalt, A1_log_T_SESAME_basalt, A2_log_u_SESAME_basalt
 )  = load_table_SESAME(gv.Fp_SESAME_basalt)
(A2_u_SESAME_water, A2_P_SESAME_water, A2_s_SESAME_water,
 A1_log_rho_SESAME_water, A1_log_T_SESAME_water, A2_log_u_SESAME_water
 )  = load_table_SESAME(gv.Fp_SESAME_water)
(A2_u_SS08_water, A2_P_SS08_water, A2_s_SS08_water, A1_log_rho_SS08_water,
 A1_log_T_SS08_water, A2_log_u_SS08_water
 )  = load_table_SESAME(gv.Fp_SS08_water)
(A2_u_SESAME_H2, A2_P_SESAME_H2, A2_s_SESAME_H2, A1_log_rho_SESAME_H2,
 A1_log_T_SESAME_H2, A2_log_u_SESAME_H2
 )  = load_table_SESAME(gv.Fp_SESAME_H2)
(A2_u_SESAME_N2, A2_P_SESAME_N2, A2_s_SESAME_N2, A1_log_rho_SESAME_N2,
 A1_log_T_SESAME_N2, A2_log_u_SESAME_N2
 )  = load_table_SESAME(gv.Fp_SESAME_N2)
(A2_u_SESAME_steam, A2_P_SESAME_steam, A2_s_SESAME_steam,
 A1_log_rho_SESAME_steam, A1_log_T_SESAME_steam, A2_log_u_SESAME_steam
 )  = load_table_SESAME(gv.Fp_SESAME_steam)
(A2_u_SESAME_CO2, A2_P_SESAME_CO2, A2_s_SESAME_CO2, A1_log_rho_SESAME_CO2,
 A1_log_T_SESAME_CO2, A2_log_u_SESAME_CO2
 )  = load_table_SESAME(gv.Fp_SESAME_CO2)

@njit
def round_to_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)

@njit
def P_u_rho(u, rho, mat_id):
    """ Computes pressure for the SESAME EoS.

        Args:
            u (double)
                Specific internal energy (SI).

            rho (double)
                Density (SI).

            mat_id (int)
                Material id.

        Returns:
            P (double)
                Pressure (SI).
    """
    # Choose the arrays from the global variables
    if (mat_id == gv.id_SESAME_iron):
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_iron, A1_log_rho_SESAME_iron, A2_log_u_SESAME_iron
            )
    elif (mat_id == gv.id_SESAME_basalt):
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_basalt, A1_log_rho_SESAME_basalt, A2_log_u_SESAME_basalt
            )
    elif (mat_id == gv.id_SESAME_water):
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_water, A1_log_rho_SESAME_water, A2_log_u_SESAME_water
            )
    elif (mat_id == gv.id_SS08_water):
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SS08_water, A1_log_rho_SS08_water, A2_log_u_SS08_water
            )
    elif (mat_id == gv.id_SESAME_H2):
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_H2, A1_log_rho_SESAME_H2, A2_log_u_SESAME_H2
            )
    elif (mat_id == gv.id_SESAME_N2):
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_N2, A1_log_rho_SESAME_N2, A2_log_u_SESAME_N2
            )
    elif (mat_id == gv.id_SESAME_steam):
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_steam, A1_log_rho_SESAME_steam, A2_log_u_SESAME_steam
            )
    elif (mat_id == gv.id_SESAME_CO2):
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_CO2, A1_log_rho_SESAME_CO2, A2_log_u_SESAME_CO2
            )
    else:
       raise ValueError("Invalid material ID")

    # Ignore the first elements of rho = 0, T = 0
    A2_P        = A2_P[1:, 1:]
    A2_log_u    = A2_log_u[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_u   = np.log(u)

    # 2D interpolation (bilinear with log(rho), log(u)) to find P(rho, u).
    # If rho and/or u are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho   = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho     = int(idx_rho_intp_rho[0])
    intp_rho    = idx_rho_intp_rho[1]
    
    # u (in this and the next density slice of the 2D u array)
    idx_u_1_intp_u_1   = find_index_and_interp(log_u, A2_log_u[idx_rho])
    idx_u_1            = int(idx_u_1_intp_u_1[0])
    intp_u_1           = idx_u_1_intp_u_1[1]
    idx_u_2_intp_u_2   = find_index_and_interp(log_u, A2_log_u[idx_rho + 1])
    idx_u_2            = int(idx_u_2_intp_u_2[0])
    intp_u_2           = idx_u_2_intp_u_2[1]
    
    P_1 = A2_P[idx_rho, idx_u_1]
    P_2 = A2_P[idx_rho, idx_u_1 + 1]
    P_3 = A2_P[idx_rho + 1, idx_u_2]
    P_4 = A2_P[idx_rho + 1, idx_u_2 + 1]

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([P_1, P_2, P_3, P_4]) < 0)
    #num_non_pos = np.sum([int(P_i <= 0) for P_i in [P_1, P_2, P_3, P_4]])
    if num_non_pos > 2:
        return 0.

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_u_1 < 0 or intp_u_2 < 0:
            return 0.
        else:
            #P_tiny  = np.amin(A2_P[A2_P > 0]) * 1e-3
            P_tiny  = np.amin(np.abs(A2_P)) * 1e-3
            if P_1 <= 0:
                P_1 = P_tiny
            if P_2 <= 0:
                P_2 = P_tiny
            if P_3 <= 0:
                P_3 = P_tiny
            if P_4 <= 0:
                P_4 = P_tiny

    # Interpolate with the log values
    P_1 = np.log(P_1)
    P_2 = np.log(P_2)
    P_3 = np.log(P_3)
    P_4 = np.log(P_4)

    # P(rho, u)
    P   = ((1 - intp_rho) * ((1 - intp_u_1) * P_1 + intp_u_1 * P_2)
           + intp_rho * ((1 - intp_u_2) * P_3 + intp_u_2 * P_4))

    # Convert back from log
    return np.exp(P)

@njit
def u_rho_T(rho, T, mat_id):
    """ Calculate the specific internal energy as a function of density and
        temperature for the SESAME EoS.

        Args:
            T (double)
                Temperature (SI).

            rho (double)
                Density (SI).

            mat_id (int)
                Material id.

        Returns:
            u (double)
                Specific internal energy (SI).
    """
    # Choose the arrays from the global variables
    if (mat_id == gv.id_SESAME_iron):
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_iron, A1_log_rho_SESAME_iron, A1_log_T_SESAME_iron
            )
    elif (mat_id == gv.id_SESAME_basalt):
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_basalt, A1_log_rho_SESAME_basalt,
            A1_log_T_SESAME_basalt
            )
    elif (mat_id == gv.id_SESAME_water):
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_water, A1_log_rho_SESAME_water,
            A1_log_T_SESAME_water
            )
    elif (mat_id == gv.id_SS08_water):
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SS08_water, A1_log_rho_SS08_water, A1_log_T_SS08_water
            )
    elif (mat_id == gv.id_SESAME_H2):
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_H2, A1_log_rho_SESAME_H2, A1_log_T_SESAME_H2
            )
    elif (mat_id == gv.id_SESAME_N2):
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_N2, A1_log_rho_SESAME_N2, A1_log_T_SESAME_N2
            )
    elif (mat_id == gv.id_SESAME_steam):
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_steam, A1_log_rho_SESAME_steam,
            A1_log_T_SESAME_steam
            )
    elif (mat_id == gv.id_SESAME_CO2):
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_CO2, A1_log_rho_SESAME_CO2, A1_log_T_SESAME_CO2
            )
    else:
       raise ValueError("Invalid material ID")

    # Ignore the first elements of rho = 0, T = 0
    A2_u    = A2_u[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_T   = np.log(T * 1) ### why is numba so weird?

    # 2D interpolation (bilinear with log(rho), log(T)) to find u(rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho   = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho     = int(idx_rho_intp_rho[0])
    intp_rho    = idx_rho_intp_rho[1]

    # Temperature
    idx_T_intp_T   = find_index_and_interp(log_T, A1_log_T[1:])
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    u_1 = A2_u[idx_rho, idx_T]
    u_2 = A2_u[idx_rho, idx_T + 1]
    u_3 = A2_u[idx_rho + 1, idx_T]
    u_4 = A2_u[idx_rho + 1, idx_T + 1]

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([u_1, u_2, u_3, u_4]) < 0)
    if num_non_pos > 2:
        return 0.

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_T < 0:
            return 0.
        else:
            u_tiny  = np.amin(np.abs(A2_u)) * 1e-3
            if u_1 <= 0:
                u_1 = u_tiny
            if u_2 <= 0:
                u_2 = u_tiny
            if u_3 <= 0:
                u_3 = u_tiny
            if u_4 <= 0:
                u_4 = u_tiny

    # Interpolate with the log values
    u_1 = np.log(u_1)
    u_2 = np.log(u_2)
    u_3 = np.log(u_3)
    u_4 = np.log(u_4)

    # u(rho, T)
    u   = ((1 - intp_rho) * ((1 - intp_T) * u_1 + intp_T * u_2)
           + intp_rho * ((1 - intp_T) * u_3 + intp_T * u_4))

    # Convert back from log
    return np.exp(u)

@njit
def s_rho_T(rho, T, mat_id):
    """ Calculate the specific entropy as a function of density and
        temperature for the SESAME EoS.

        Args:
            rho (double)
                Density (SI).

            T (double)
                Temperature (SI).

            mat_id (int)
                Material id.

        returns:
            s (double)
                Specific internal energy (SI).
    """
    # Choose the arrays from the global variables
    if (mat_id == gv.id_SESAME_iron):
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_iron, A1_log_rho_SESAME_iron, A1_log_T_SESAME_iron
            )
    elif (mat_id == gv.id_SESAME_basalt):
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_basalt, A1_log_rho_SESAME_basalt,
            A1_log_T_SESAME_basalt
            )
    elif (mat_id == gv.id_SESAME_water):
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_water, A1_log_rho_SESAME_water,
            A1_log_T_SESAME_water
            )
    elif (mat_id == gv.id_SS08_water):
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SS08_water, A1_log_rho_SS08_water, A1_log_T_SS08_water
            )
    elif (mat_id == gv.id_SESAME_H2):
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_H2, A1_log_rho_SESAME_H2, A1_log_T_SESAME_H2
            )
    elif (mat_id == gv.id_SESAME_N2):
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_N2, A1_log_rho_SESAME_N2, A1_log_T_SESAME_N2
            )
    elif (mat_id == gv.id_SESAME_steam):
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_steam, A1_log_rho_SESAME_steam,
            A1_log_T_SESAME_steam
            )
    elif (mat_id == gv.id_SESAME_CO2):
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_CO2, A1_log_rho_SESAME_CO2, A1_log_T_SESAME_CO2
            )
    else:
       raise ValueError("Invalid material ID")

    # Ignore the first elements of rho = 0, T = 0
    A2_s    = A2_s[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_T   = np.log(T)

    # 2D interpolation (bilinear with log(rho), log(T)) to find s(rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho   = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho     = int(idx_rho_intp_rho[0])
    intp_rho    = idx_rho_intp_rho[1]

    # Temperature
    idx_T_intp_T   = find_index_and_interp(log_T, A1_log_T[1:])
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    s_1 = A2_s[idx_rho, idx_T]
    s_2 = A2_s[idx_rho, idx_T + 1]
    s_3 = A2_s[idx_rho + 1, idx_T]
    s_4 = A2_s[idx_rho + 1, idx_T + 1]

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([s_1, s_2, s_3, s_4]) < 0)
    if num_non_pos > 2:
        return 0.

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_T < 0:
            return 0.
        else:
            s_tiny  = np.amin(np.abs(A2_s)) * 1e-3
            if s_1 <= 0:
                s_1 = s_tiny
            if s_2 <= 0:
                s_2 = s_tiny
            if s_3 <= 0:
                s_3 = s_tiny
            if s_4 <= 0:
                s_4 = s_tiny

    # s(rho, T)
    s   = ((1 - intp_rho) * ((1 - intp_T) * s_1 + intp_T * s_2)
           + intp_rho * ((1 - intp_T) * s_3 + intp_T * s_4))
    
    return s

@njit
def T_rho_s(rho, s, mat_id):
    """ Calculate the temperature as a function of density and
        specific entropy for the SESAME EoS.

        Args:
            rho (double)
                Density (SI).

            s (double)
                Specific entropy (SI).

            mat_id (int)
                Material id.

        Returns:
            T (double)
                Temperature (SI).
    """
    # Choose the arrays from the global variables
    if (mat_id == gv.id_SESAME_iron):
        A1_log_T, A1_log_rho, A2_s  = (
            A1_log_T_SESAME_iron, A1_log_rho_SESAME_iron, A2_s_SESAME_iron
            )
    elif (mat_id == gv.id_SESAME_basalt):
        A1_log_T, A1_log_rho, A2_s  = (
            A1_log_T_SESAME_basalt, A1_log_rho_SESAME_basalt, A2_s_SESAME_basalt
            )
    elif (mat_id == gv.id_SESAME_water):
        A1_log_T, A1_log_rho, A2_s  = (
            A1_log_T_SESAME_water, A1_log_rho_SESAME_water, A2_s_SESAME_water
            )
    elif (mat_id == gv.id_SS08_water):
        A1_log_T, A1_log_rho, A2_s  = (
            A1_log_T_SS08_water, A1_log_rho_SS08_water, A2_s_SS08_water
            )
    elif (mat_id == gv.id_SESAME_H2):
        A1_log_T, A1_log_rho, A2_s  = (
            A1_log_T_SESAME_H2, A1_log_rho_SESAME_H2, A2_s_SESAME_H2
            )
    elif (mat_id == gv.id_SESAME_N2):
        A1_log_T, A1_log_rho, A2_s  = (
            A1_log_T_SESAME_N2, A1_log_rho_SESAME_N2, A2_s_SESAME_N2
            )
    elif (mat_id == gv.id_SESAME_steam):
        A1_log_T, A1_log_rho, A2_s  = (
            A1_log_T_SESAME_steam, A1_log_rho_SESAME_steam, A2_s_SESAME_steam
            )
    elif (mat_id == gv.id_SESAME_CO2):
        A1_log_T, A1_log_rho, A2_s  = (
            A1_log_T_SESAME_CO2, A1_log_rho_SESAME_CO2, A2_s_SESAME_CO2
            )
    else:
       raise ValueError("Invalid material ID")

    # Convert to log
    log_rho = np.log(rho)

    idx_rho_intp_rho   = find_index_and_interp(log_rho, A1_log_rho)
    idx_rho     = int(idx_rho_intp_rho[0])
    intp_rho    = idx_rho_intp_rho[1]

    # s (in this and the next density slice of the 2D s array)
    idx_s_1_intp_s_1   = find_index_and_interp(s, A2_s[idx_rho])
    idx_s_1            = int(idx_s_1_intp_s_1[0])
    intp_s_1           = idx_s_1_intp_s_1[1]
    idx_s_2_intp_s_2   = find_index_and_interp(s, A2_s[idx_rho + 1])
    idx_s_2            = int(idx_s_2_intp_s_2[0])
    intp_s_2           = idx_s_2_intp_s_2[1]

    # Normal interpolation
    log_T   = ((1 - intp_rho) * ((1 - intp_s_1) * A1_log_T[idx_s_1]
                                 + intp_s_1 * A1_log_T[idx_s_1 + 1])
               + intp_rho * ((1 - intp_s_2) * A1_log_T[idx_s_2]
                             + intp_s_2 * A1_log_T[idx_s_2 + 1]))
    
    # Convert back from log
    T   = np.exp(log_T)
    if T < 0:
        T   = 0

    return T