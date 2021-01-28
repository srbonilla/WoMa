"""
WoMa global variables
"""

import os

G = 6.67408e-11  # m^3 kg^-1 s^-2
R_earth = 6.371e6  # m
M_earth = 5.9724e24  # kg
R_gas = 8.3145  # Gas constant (J K^-1 mol^-1)

# Material IDs, same as SWIFT ( = type_id * type_factor + unit_id)
type_factor = 100
Di_mat_type = {
    "idg": 0,
    "Til": 1,
    "HM80": 2,
    "SESAME": 3,
    "ANEOS": 4,
}
Di_mat_id = {
    # Ideal Gas
    "idg_HHe": Di_mat_type["idg"] * type_factor,
    "idg_N2": Di_mat_type["idg"] * type_factor + 1,
    "idg_CO2": Di_mat_type["idg"] * type_factor + 2,
    # Tillotson
    "Til_iron": Di_mat_type["Til"] * type_factor,
    "Til_granite": Di_mat_type["Til"] * type_factor + 1,
    "Til_water": Di_mat_type["Til"] * type_factor + 2,
    "Til_basalt": Di_mat_type["Til"] * type_factor + 3,
    # Hubbard & MacFarlane (1980) Uranus/Neptune
    "HM80_HHe": Di_mat_type["HM80"] * type_factor,  # Hydrogen-helium atmosphere
    "HM80_ice": Di_mat_type["HM80"] * type_factor + 1,  # H20-CH4-NH3 ice mix
    "HM80_rock": Di_mat_type["HM80"] * type_factor + 2,  # SiO2-MgO-FeS-FeO rock mix
    # SESAME
    "SESAME_iron": Di_mat_type["SESAME"] * type_factor,  # 2140
    "SESAME_basalt": Di_mat_type["SESAME"] * type_factor + 1,  # 7530
    "SESAME_water": Di_mat_type["SESAME"] * type_factor + 2,  # 7154
    "SS08_water": Di_mat_type["SESAME"] * type_factor + 3,  # Senft & Stewart (2008)
    "AQUA": Di_mat_type["SESAME"] * type_factor + 4,  # Haldemann+2020
    "CMS19_H": Di_mat_type["SESAME"] * type_factor + 5,  # Chabrier+2019 Hydrogen
    "CMS19_He": Di_mat_type["SESAME"] * type_factor + 6,  # Helium
    "CMS19_HHe": Di_mat_type["SESAME"] * type_factor + 7,  # H/He mixture Y=0.275
    # ANEOS
    "ANEOS_forsterite": Di_mat_type["ANEOS"] * type_factor,  # Stewart et al. (2019)
    "ANEOS_iron": Di_mat_type["ANEOS"] * type_factor + 1,  # Stewart (2020)
    "ANEOS_Fe85Si15": Di_mat_type["ANEOS"] * type_factor + 2,  # Stewart (2020)
}
# Invert so the ID are the keys
Di_id_mat = {mat_id: mat for mat, mat_id in Di_mat_id.items()}

# Separate variables because numba can't handle dictionaries
# Types
type_idg = Di_mat_type["idg"]
type_Til = Di_mat_type["Til"]
type_HM80 = Di_mat_type["HM80"]
type_SESAME = Di_mat_type["SESAME"]
type_ANEOS = Di_mat_type["ANEOS"]
# IDs
id_idg_HHe = Di_mat_id["idg_HHe"]
id_idg_N2 = Di_mat_id["idg_N2"]
id_idg_CO2 = Di_mat_id["idg_CO2"]
id_Til_iron = Di_mat_id["Til_iron"]
id_Til_granite = Di_mat_id["Til_granite"]
id_Til_basalt = Di_mat_id["Til_basalt"]
id_Til_water = Di_mat_id["Til_water"]
id_HM80_HHe = Di_mat_id["HM80_HHe"]
id_HM80_ice = Di_mat_id["HM80_ice"]
id_HM80_rock = Di_mat_id["HM80_rock"]
id_SESAME_iron = Di_mat_id["SESAME_iron"]
id_SESAME_basalt = Di_mat_id["SESAME_basalt"]
id_SESAME_water = Di_mat_id["SESAME_water"]
id_SS08_water = Di_mat_id["SS08_water"]
id_AQUA = Di_mat_id["AQUA"]
id_CMS19_H = Di_mat_id["CMS19_H"]
id_CMS19_He = Di_mat_id["CMS19_He"]
id_CMS19_HHe = Di_mat_id["CMS19_HHe"]
id_ANEOS_forsterite = Di_mat_id["ANEOS_forsterite"]
id_ANEOS_iron = Di_mat_id["ANEOS_iron"]
id_ANEOS_Fe85Si15 = Di_mat_id["ANEOS_Fe85Si15"]

# T-rho relation types
type_rho_pow = 1
type_adb = 2
type_ent = 3
Di_T_rho_id = {"power": type_rho_pow, "adiabatic": type_adb, "entropy": type_ent}

# Local data files
this_dir, this_file = os.path.split(__file__)
dir_data = os.path.join(this_dir, "../data/")
# Tillotson cold curves
Fp_u_cold_Til_iron = dir_data + "u_cold_array_Til_iron.npy"
Fp_u_cold_Til_granite = dir_data + "u_cold_array_Til_granite.npy"
Fp_u_cold_Til_basalt = dir_data + "u_cold_array_Til_basalt.npy"
Fp_u_cold_Til_water = dir_data + "u_cold_array_Til_water.npy"
# HM80 tables
Fp_HM80_HHe = dir_data + "HM80_HHe.txt"
Fp_HM80_ice = dir_data + "HM80_ice.txt"
Fp_HM80_rock = dir_data + "HM80_rock.txt"
# HM80 cold curves
Fp_u_cold_HM80_ice = dir_data + "u_cold_array_HM80_ice.npy"
Fp_u_cold_HM80_rock = dir_data + "u_cold_array_HM80_rock.npy"
# SESAME tables
Fp_SESAME_iron = dir_data + "SESAME_iron_2140.txt"
Fp_SESAME_basalt = dir_data + "SESAME_basalt_7530.txt"
Fp_SESAME_water = dir_data + "SESAME_water_7154.txt"
Fp_SS08_water = dir_data + "SS08_water.txt"
Fp_AQUA = dir_data + "AQUA_H20.txt"
Fp_CMS19_H = dir_data + "CMS19_H.txt"
Fp_CMS19_He = dir_data + "CMS19_He.txt"
Fp_CMS19_HHe = dir_data + "CMS19_HHe.txt"
# ANEOS tables
Fp_ANEOS_forsterite = dir_data + "ANEOS_forsterite_S19.txt"
Fp_ANEOS_iron = dir_data + "ANEOS_iron_S20.txt"
Fp_ANEOS_Fe85Si15 = dir_data + "ANEOS_Fe85Si15_S20.txt"
