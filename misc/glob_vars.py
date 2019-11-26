#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 08:54:38 2019

@author: sergio
"""

G       = 6.67408E-11
R_earth = 6371000
M_earth = 5.9724E24
R_gas   = 8.3145     # Gas constant (J K^-1 mol^-1)

# Material IDs, same as SWIFT ( = type_id * type_factor + unit_id)
type_factor = 100
Di_mat_type = {
    "idg"       : 0,
    "Til"       : 1,
    "HM80"      : 2,
    "SESAME"    : 3,
    }
Di_mat_id   = {
    # Ideal Gas
    "idg_HHe"       : Di_mat_type["idg"]*type_factor,
    "idg_N2"        : Di_mat_type["idg"]*type_factor + 1,
    "idg_CO2"       : Di_mat_type["idg"]*type_factor + 2,
    # Tillotson
    "Til_iron"      : Di_mat_type["Til"]*type_factor,
    "Til_granite"   : Di_mat_type["Til"]*type_factor + 1,
    "Til_water"     : Di_mat_type["Til"]*type_factor + 2,
    # Hubbard & MacFarlane (1980) Uranus/Neptune
    "HM80_HHe"      : Di_mat_type["HM80"]*type_factor,      # Hydrogen-helium atmosphere
    "HM80_ice"      : Di_mat_type["HM80"]*type_factor + 1,  # H20-CH4-NH3 ice mix
    "HM80_rock"     : Di_mat_type["HM80"]*type_factor + 2,  # SiO2-MgO-FeS-FeO rock mix
    # SESAME
    "SESAME_iron"   : Di_mat_type["SESAME"]*type_factor,        # 2140
    "SESAME_basalt" : Di_mat_type["SESAME"]*type_factor + 1,    # 7530
    "SESAME_water"  : Di_mat_type["SESAME"]*type_factor + 2,    # 7154
    "SS08_water"    : Di_mat_type["SESAME"]*type_factor + 3,    # Senft & Stewart (2008)
    "SESAME_H2"     : Di_mat_type["SESAME"]*type_factor + 4,    # 5251
    "SESAME_N2"     : Di_mat_type["SESAME"]*type_factor + 5,    # 5000
    "SESAME_steam"  : Di_mat_type["SESAME"]*type_factor + 6,    # 7152
    "SESAME_CO2"    : Di_mat_type["SESAME"]*type_factor + 7,    # 5212
    }
# Separate variables because numba can't handle dictionaries
# Types
type_idg    = Di_mat_type["idg"]
type_Til    = Di_mat_type["Til"]
type_HM80   = Di_mat_type["HM80"]
type_SESAME = Di_mat_type["SESAME"]
# IDs
id_idg_HHe          = Di_mat_id["idg_HHe"]
id_idg_N2           = Di_mat_id["idg_N2"]
id_idg_CO2          = Di_mat_id["idg_CO2"]
id_Til_iron         = Di_mat_id["Til_iron"]
id_Til_granite      = Di_mat_id["Til_granite"]
id_Til_water        = Di_mat_id["Til_water"]
id_HM80_HHe         = Di_mat_id["HM80_HHe"]
id_HM80_ice         = Di_mat_id["HM80_ice"]
id_HM80_rock        = Di_mat_id["HM80_rock"]
id_SESAME_iron      = Di_mat_id["SESAME_iron"]
id_SESAME_basalt    = Di_mat_id["SESAME_basalt"]
id_SESAME_water     = Di_mat_id["SESAME_water"]
id_SS08_water       = Di_mat_id["SS08_water"]
id_SESAME_H2        = Di_mat_id["SESAME_H2"]
id_SESAME_N2        = Di_mat_id["SESAME_N2"]
id_SESAME_steam     = Di_mat_id["SESAME_steam"]
id_SESAME_CO2       = Di_mat_id["SESAME_CO2"]

# T-rho relation types
type_rho_pow    = 1
type_adb        = 2
Di_T_rho_id =  {
        "power"     : type_rho_pow,
        "adiabatic" : type_adb
        }

# Local data files
dir_data    = "data/"
# Tillotson cold curves
Fp_u_cold_Til_iron      = dir_data + "u_cold_array_Til_iron.npy"
Fp_u_cold_Til_granite   = dir_data + "u_cold_array_Til_granite.npy"
Fp_u_cold_Til_water     = dir_data + "u_cold_array_Til_water.npy"
# HM80 tables
Fp_HM80_HHe         = dir_data + "HM80_HHe.txt"
Fp_HM80_ice         = dir_data + "HM80_ice.txt"
Fp_HM80_rock        = dir_data + "HM80_rock.txt"
# SESAME tables
Fp_SESAME_iron      = dir_data + "SESAME_iron_2140.txt"
Fp_SESAME_basalt    = dir_data + "SESAME_basalt_7530.txt"
Fp_SESAME_water     = dir_data + "SESAME_water_7154.txt"
Fp_SS08_water       = dir_data + "SS08_water.txt"
Fp_SESAME_H2        = dir_data + "SESAME_H2_5251.txt"
Fp_SESAME_N2        = dir_data + "SESAME_N2_5000.txt"
Fp_SESAME_steam     = dir_data + "SESAME_steam_7152.txt"
Fp_SESAME_CO2       = dir_data + "SESAME_CO2_5212.txt"