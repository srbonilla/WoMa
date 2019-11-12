#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:34:43 2019

@author: sergio
"""
import sys
import os

# Go to the WoMa directory
cwd = os.getcwd()
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
sys.path.append(dir + '/eos')
sys.path.append(dir + '/spherical_funcs')
sys.path.append(dir + '/spin_funcs')
sys.path.append(dir + '/misc')

import numpy as np
import h5py
import L1_spherical
import L2_spherical
import L3_spherical
import L1_spin
import L2_spin
import L3_spin
import glob_vars as gv
import eos
import utils
import utils_spin as us
from T_rho import set_T_rho_args
from T_rho import T_rho
from scipy.interpolate import interp1d
from tqdm import tqdm

os.chdir(cwd)

# Output
Di_hdf5_planet_label  = {
    "num_layer"     : "Number of Layers",
    "mat_layer"     : "Layer Materials",
    "mat_id_layer"  : "Layer Material IDs",
    "T_rho_type"    : "Layer T-rho Type",
    "T_rho_args"    : "Layer T-rho Args",
    "R_layer"       : "Layer Boundary Radii",
    "M_layer"       : "Mass in each Layer",
    "M"             : "Total Mass",
    "R"             : "Total Radius",
    "idx_layer"     : "Outer Index of each Layer",
    "P_s"           : "Surface Pressure",
    "T_s"           : "Surface Temperature",
    "rho_s"         : "Surface Density",
    "r"             : "Profile Radii",
    "m_enc"         : "Profile Enclosed Masses",
    "rho"           : "Profile Densities",
    "T"             : "Profile Temperatures",
    "u"             : "Profile Specific Internal Energies",
    "P"             : "Profile Pressures",
    "mat_id"        : "Profile Material IDs",
    }

def get_planet_data(f, param):
    """ Load a planet attribute or array.

        Args:
            f (h5py File)
                The opened hdf5 data file (with "r").

            param (str)
                The array or attribute to get. See Di_hdf5_planet_label for
                details.

        Returns:
            data (?)
                The array or attribute (std units).
    """
    # Attributes
    try:
        return f["planet"].attrs[Di_hdf5_planet_label[param]]
    # Datasets
    except KeyError:
        return f["planet/" + Di_hdf5_planet_label[param]][()]

def multi_get_planet_data(f, A1_param):
    """ Load multiple planet attributes or arrays.

        Args:
            f (h5py File)
                The opened hdf5 data file (with "r").

            A1_param ([str])
                List of the arrays or attributes to get. See Di_hdf5_planet_label
                for details.

        Returns:
            A1_data ([?])
                The list of the arrays or attributes (std units).
    """
    A1_data = []
    # Load each requested array
    for param in A1_param:
        A1_data.append(get_planet_data(f, param))

    return A1_data

def load_planet(name, Fp_planet):
    """ Return a new Planet object loaded from a file.

        Args:
            name (str)
                The name of the planet object.

            Fp_planet (str)
                The object data file path.
    """
    p = Planet(name=name, Fp_planet=Fp_planet)

    Fp_planet = utils.check_end(p.Fp_planet, ".hdf5")

    print("Loading \"%s\"... " % Fp_planet[-60:], end='')
    sys.stdout.flush()

    with h5py.File(Fp_planet, "r") as f:
        (p.num_layer, p.A1_mat_layer, p.A1_mat_id_layer, p.A1_T_rho_type,
         p.A1_T_rho_args, p.A1_R_layer, p.A1_M_layer, p.M, p.R, p.A1_idx_layer,
         p.P_s, p.T_s, p.rho_s, p.A1_r, p.A1_m_enc, p.A1_rho, p.A1_T, p.A1_P,
         p.A1_u, p.A1_mat_id
         ) = multi_get_planet_data(
            f, ["num_layer", "mat_layer", "mat_id_layer", "T_rho_type",
                "T_rho_args", "R_layer", "M_layer", "M", "R", "idx_layer",
                "P_s", "T_s", "rho_s", "r", "m_enc", "rho", "T", "P", "u",
                "mat_id"]
            )

    print("Done")

    p.update_attributes()
    p.print_info()

    return p

# ============================================================================ #
#                       Spherical profile class                                #
# ============================================================================ #

class Planet():
    """ Planet class ...

        Args:
            name (str)
                The name of the planet object.

            Fp_planet (str)
                The object data file path. Default to "data/<name>.hdf5".

            A1_mat_layer ([str])
                The name of the material in each layer, from the central layer
                outwards. See Di_mat_id in weos.py.
                
            A1_T_rho_type ([int])
                The type of temperature-density relation in each layer, from the
                central layer outwards.

                'power':   T = K * rho^alpha, K is set internally using
                                    each layer's outer temperature.
                                    Set alpha = 0 for isothermal.
                'adiabatic':       Adiabatic, constant s_adb is set internally,
                                    if applicable.

            A1_T_rho_args ([float])
                type_rho_pow:   [[K, alpha], ...] Only alpha is required input.
                type_adb:       [[s_adb], ...] or [[T, P, rho, rho_old], ...]
                                No required input.

            A1_R_layer ([float])
                The outer radii of each layer, from the central layer outwards
                (SI).

            A1_M_layer ([float])
                The mass within each layer, starting from the from the central
                layer outwards (SI).

            M (float)
                The total mass.

            P_s, T_s, rho_s (float)
                The pressure, temperature, and density at the surface. Only two
                of the three must be provided (Pa, K, kg m^-3).

            P_0, P_1, ..., T_0, ..., rho_0, ... (float)
                The pressure, temperature, and density at each layer boundary,
                from the centre (_0) up to the surface.

            I_MR2 (float)
                The reduced moment of inertia. (SI)

            M_max (float)
                The maximum mass allowed.

            R_min, R_max (float)
                The minimum and maximum radii to try.

            rho_min (float)
                The minimum density for the outer edge of the profile.

            M_frac_tol (float)
                The tolerance for finding the appropriate mass, as a fraction
                of the total mass. Default: 0.01.

            num_prof (int)
                The number of profile integration steps.

            num_attempt, num_attempt_2 (int)
                The maximum number of iteration attempts.

        Attrs (in addition to the args):
            num_layer (int)
                The number of planetary layers.

            A1_mat_id_layer ([int])
                The ID of the material in each layer, from the central layer
                outwards.

            A1_idx_layer ([int])
                The profile index of each boundary, from the central layer
                outwards.

            A1_r ([float])
                The profile radii, in increasing order.

            ...
    """
    def __init__(
        self, name=None, Fp_planet=None, A1_mat_layer=None, A1_T_rho_type=None,
        A1_T_rho_args=None, A1_R_layer=None, A1_M_layer=None, A1_idx_layer=None,
        M=None, P_0=None, T_0=None, rho_0=None, P_1=None, T_1=None, rho_1=None,
        P_2=None, T_2=None, rho_2=None, P_s=None, T_s=None, rho_s=None,
        I_MR2=None, M_max=None, R_min=None, R_max=None, rho_min=None,
        M_frac_tol=0.01, num_prof=10000, num_attempt=40, num_attempt_2=40
        ):
        self.name               = name
        self.Fp_planet          = Fp_planet
        self.A1_mat_layer       = A1_mat_layer
        self.A1_T_rho_type      = A1_T_rho_type
        self.A1_T_rho_args      = A1_T_rho_args
        self.A1_R_layer         = A1_R_layer
        self.A1_M_layer         = A1_M_layer
        self.A1_idx_layer       = A1_idx_layer
        self.M                  = M
        self.P_0                = P_0
        self.T_0                = T_0
        self.rho_0              = rho_0
        self.P_1                = P_1
        self.T_1                = T_1
        self.rho_1              = rho_1
        self.P_2                = P_2
        self.T_2                = T_2
        self.rho_2              = rho_2
        self.P_s                = P_s
        self.T_s                = T_s
        self.rho_s              = rho_s
        self.I_MR2              = I_MR2
        self.M_max              = M_max
        self.R_min              = R_min
        self.R_max              = R_max
        self.rho_min            = rho_min
        self.M_frac_tol         = M_frac_tol
        self.num_prof           = num_prof
        self.num_attempt        = num_attempt
        self.num_attempt_2      = num_attempt_2

        # Derived or default attributes
        if self.A1_mat_layer is not None:
            self.num_layer          = len(self.A1_mat_layer)
            self.A1_mat_id_layer    = [gv.Di_mat_id[mat]
                                       for mat in self.A1_mat_layer]
        else:
            # Placeholder
            self.num_layer  = 1
        if self.A1_T_rho_type is not None:
            self.A1_T_rho_type_id = [gv.Di_T_rho_id[T_rho_id]
                                       for T_rho_id in self.A1_T_rho_type]
        
        if self.Fp_planet is None:
            self.Fp_planet  = "data/%s.hdf5" % self.name
        if self.A1_R_layer is None:
            self.A1_R_layer = [None] * self.num_layer
        if self.A1_M_layer is None:
            self.A1_M_layer = [None] * self.num_layer
        self.R  = self.A1_R_layer[-1]

        # Force types for numba
        if self.A1_R_layer is not None:
            self.A1_R_layer = np.array(self.A1_R_layer, dtype="float")
        if self.A1_T_rho_args is not None:
            self.A1_T_rho_args = np.array(self.A1_T_rho_args, dtype="float")

        # Two of P, T, rho must be provided at the surface to calculate the
        # third. If all three are provided then rho is overwritten.
        if self.P_s is not None and self.T_s is not None:
            self.rho_s  = eos.rho_P_T(self.P_s, self.T_s,
                                      self.A1_mat_id_layer[-1])
        # elif self.P_s is not None and self.rho_s is not None:
        #     self.T_s    = eos.find_T_fixed_P_rho(self.P_s, self.rho_s,
        #                                           self.A1_mat_id_layer[-1])
        elif self.rho_s is not None and self.T_s is not None:
            self.P_s    = eos.P_T_rho(self.T_s, self.rho_s,
                                                  self.A1_mat_id_layer[-1])

        ### default M_max and R_max?

        # Help info ###todo
        if not True:
            if self.num_layer == 1:
                print("For a 1 layer planet, please specify:")
                print("pressure, temperature and density at the surface of the planet,")
                print("material, relation between temperature and density with any desired aditional parameters,")
                print("for layer 1 of the planet.")
            elif self.num_layer == 2:
                print("For a 2 layer planet, please specify:")
                print("pressure, temperature and density at the surface of the planet,")
                print("materials, relations between temperature and density with any desired aditional parameters,")
                print("for layer 1 and layer 2 of the planet.")
            elif self.num_layer == 3:
                print("For a 3 layer planet, please specify:")
                print("pressure, temperature and density at the surface of the planet,")
                print("materials, relations between temperature and density with any desired aditional parameters,")
                print("for layer 1, layer 2, and layer 3 of the planet.")

    # ========
    # General
    # ========
    def escape_velocity(self):
        assert(self.M is not None)
        assert(self.A1_R_layer[-1] is not None)
        
        self.v_escape = np.sqrt(2*gv.G*self.M/self.A1_R_layer[-1])
        
    def update_attributes(self):
        """ Set all planet information after making the profiles.
        """
        self.num_prof   = len(self.A1_r)

        # Reverse profile arrays to be ordered by increasing radius
        if self.A1_r[-1] < self.A1_r[0]:
            self.A1_r       = self.A1_r[::-1]
            self.A1_m_enc   = self.A1_m_enc[::-1]
            self.A1_P       = self.A1_P[::-1]
            self.A1_T       = self.A1_T[::-1]
            self.A1_rho     = self.A1_rho[::-1]
            self.A1_u       = self.A1_u[::-1]
            self.A1_mat_id  = self.A1_mat_id[::-1]

        # Index of the outer edge of each layer
        self.A1_idx_layer   = np.append(
            np.where(np.diff(self.A1_mat_id) != 0)[0], self.num_prof - 1)

        # Boundary radii
        self.A1_R_layer = self.A1_r[self.A1_idx_layer]
        self.R          = self.A1_R_layer[-1]

        # Layer masses
        self.A1_M_layer = self.A1_m_enc[self.A1_idx_layer]
        if self.num_layer > 1:
            self.A1_M_layer[1:] -= self.A1_M_layer[:-1]
        self.M  = np.sum(self.A1_M_layer)

        # Moment of inertia
        self.I_MR2  = utils.moi(self.A1_r, self.A1_rho)

        # P, T, and rho at the centre and the outer boundary of each layer
        self.P_0    = self.A1_P[0]
        self.T_0    = self.A1_T[0]
        self.rho_0  = self.A1_rho[0]
        if self.num_layer > 1:
            self.P_1    = self.A1_P[self.A1_idx_layer[0]]
            self.T_1    = self.A1_T[self.A1_idx_layer[0]]
            self.rho_1  = self.A1_rho[self.A1_idx_layer[0]]
        if self.num_layer > 2:
            self.P_2    = self.A1_P[self.A1_idx_layer[1]]
            self.T_2    = self.A1_T[self.A1_idx_layer[1]]
            self.rho_2  = self.A1_rho[self.A1_idx_layer[1]]
        self.P_s    = self.A1_P[-1]
        self.T_s    = self.A1_T[-1]
        self.rho_s  = self.A1_rho[-1]
        
        self.escape_velocity()

    def print_info(self):
        """ Print the Planet objects's main properties. """
        # Print and catch if any variables are None
        def print_try(string, variables):
            try:
                print(string % variables)
            except TypeError:
                print("    %s = None" % variables[0])
        
        space   = 12
        print_try("Planet \"%s\": ", self.name)
        print_try("    %s = %.5g kg = %.5g M_earth",
                  (utils.add_whitespace("M", space), self.M, self.M/gv.M_earth))
        print_try("    %s = %.5g m = %.5g R_earth",
                  (utils.add_whitespace("R", space), self.R, self.R/gv.R_earth))
        print_try("    %s = %s ", (utils.add_whitespace("mat", space),
                  utils.format_array_string(self.A1_mat_layer, "string")))
        print_try("    %s = %s ", (utils.add_whitespace("mat_id", space),
                  utils.format_array_string(self.A1_mat_id_layer, "%d")))
        print_try("    %s = %s R_earth", (utils.add_whitespace("R_layer", space),
                  utils.format_array_string(self.A1_R_layer / gv.R_earth, "%.5g")))
        print_try("    %s = %s M_earth", (utils.add_whitespace("M_layer", space),
                  utils.format_array_string(self.A1_M_layer / gv.M_earth, "%.5g")))
        print_try("    %s = %s M_total", (utils.add_whitespace("M_frac_layer", space),
                  utils.format_array_string(self.A1_M_layer / self.M, "%.5g")))
        print_try("    %s = %s ", (utils.add_whitespace("idx_layer", space),
                  utils.format_array_string(self.A1_idx_layer, "%d")))
        print_try("    %s = %.5g Pa", (utils.add_whitespace("P_s", space), self.P_s))
        print_try("    %s = %.5g K", (utils.add_whitespace("T_s", space), self.T_s))
        print_try("    %s = %.5g kg/m^3", 
                  (utils.add_whitespace("rho_s", space), self.rho_s))
        if self.num_layer > 2:
            print_try("    %s = %.5g Pa",
                      (utils.add_whitespace("P_2", space), self.P_2))
            print_try("    %s = %.5g K", 
                      (utils.add_whitespace("T_2", space), self.T_2))
            print_try("    %s = %.5g kg/m^3", 
                      (utils.add_whitespace("rho_2", space), self.rho_2))
        if self.num_layer > 1:
            print_try("    %s = %.5g Pa", 
                      (utils.add_whitespace("P_1", space), self.P_1))
            print_try("    %s = %.5g K", 
                      (utils.add_whitespace("T_1", space), self.T_1))
            print_try("    %s = %.5g kg/m^3", 
                      (utils.add_whitespace("rho_1", space), self.rho_1))
        print_try("    %s = %.5g Pa", (utils.add_whitespace("P_0", space), self.P_0))
        print_try("    %s = %.5g K", (utils.add_whitespace("T_0", space), self.T_0))
        print_try("    %s = %.5g kg/m^3", 
                  (utils.add_whitespace("rho_0", space), self.rho_0))
        print_try("    %s = %.5g M_tot*R_tot^2",
                  (utils.add_whitespace("I_MR2", space), 
                   self.I_MR2/self.M/self.R/self.R))

    def print_declaration(self):
        """ Print the Planet objects formatted as a declaration. """
        space   = 15
        print("%s = Planet(" % self.name)
        print("    %s = \"%s\"," %
              (utils.add_whitespace("name", space), self.name))
        print("    %s = \"%s\"," %
              (utils.add_whitespace("Fp_planet", space), self.Fp_planet))
        print("    %s = %s," %
              (utils.add_whitespace("A1_mat_layer", space),
               utils.format_array_string(self.A1_mat_layer, "string")))
        print("    %s = %s," %
              (utils.add_whitespace("A1_T_rho_type_id", space),
               utils.format_array_string(self.A1_T_rho_type_id, "%d")))
        print("    %s = %s," %
              (utils.add_whitespace("A1_T_rho_args", space),
               utils.format_array_string(self.A1_T_rho_args, "dorf")))
        print("    %s = np.array(%s) * R_earth," %
              (utils.add_whitespace("A1_R_layer", space),
              utils.format_array_string(self.A1_R_layer / gv.R_earth, "%.5g")))
        print("    %s = %s," %
              (utils.add_whitespace("A1_idx_layer", space),
              utils.format_array_string(self.A1_idx_layer, "%d")))
        print("    %s = np.array(%s) * M_earth," %
              (utils.add_whitespace("A1_M_layer", space),
              utils.format_array_string(self.A1_M_layer / gv.M_earth, "%.5g")))
        print("    %s = %.5g * M_earth," %
              (utils.add_whitespace("M", space), self.M / gv.M_earth))
        print("    %s = %.5g," % (utils.add_whitespace("P_s", space), self.P_s))
        print("    %s = %.5g," % (utils.add_whitespace("T_s", space), self.T_s))
        print("    %s = %.5g," % (utils.add_whitespace("rho_s", space), self.rho_s))
        if self.num_layer > 2:
            print("    %s = %.5g," % (utils.add_whitespace("P_2", space), self.P_2))
            print("    %s = %.5g," % (utils.add_whitespace("T_2", space), self.T_2))
            print("    %s = %.5g," %
                  (utils.add_whitespace("rho_2", space), self.rho_2))
        if self.num_layer > 1:
            print("    %s = %.5g," % (utils.add_whitespace("P_1", space), self.P_1))
            print("    %s = %.5g," % (utils.add_whitespace("T_1", space), self.T_1))
            print("    %s = %.5g," %
                  (utils.add_whitespace("rho_1", space), self.rho_1))
        print("    %s = %.5g," % (utils.add_whitespace("P_0", space), self.P_0))
        print("    %s = %.5g," % (utils.add_whitespace("T_0", space), self.T_0))
        print("    %s = %.5g," % (utils.add_whitespace("rho_0", space), self.rho_0))
        print("    %s = %.5g," %
              (utils.add_whitespace("I_MR2", space),
               self.I_MR2 / (self.M * self.R**2)))
        print("    )")

    def save_planet(self):
        Fp_planet = utils.check_end(self.Fp_planet, ".hdf5")

        print("Saving \"%s\"... " % Fp_planet[-60:], end='')
        sys.stdout.flush()

        with h5py.File(Fp_planet, "w") as f:
            # Group
            grp = f.create_group("/planet")

            # Lists not numpy for attributes
            if type(self.A1_mat_layer).__module__ == np.__name__:
                self.A1_mat_layer   = self.A1_mat_layer.tolist()

            # Attributes
            grp.attrs[Di_hdf5_planet_label["num_layer"]]    = self.num_layer
            grp.attrs[Di_hdf5_planet_label["mat_layer"]]    = self.A1_mat_layer
            grp.attrs[Di_hdf5_planet_label["mat_id_layer"]] = self.A1_mat_id_layer
            grp.attrs[Di_hdf5_planet_label["T_rho_type"]]   = self.A1_T_rho_type
            grp.attrs[Di_hdf5_planet_label["T_rho_args"]]   = self.A1_T_rho_args
            grp.attrs[Di_hdf5_planet_label["R_layer"]]      = self.A1_R_layer
            grp.attrs[Di_hdf5_planet_label["M_layer"]]      = self.A1_M_layer
            grp.attrs[Di_hdf5_planet_label["M"]]            = self.M
            grp.attrs[Di_hdf5_planet_label["R"]]            = self.R
            grp.attrs[Di_hdf5_planet_label["idx_layer"]]    = self.A1_idx_layer
            grp.attrs[Di_hdf5_planet_label["P_s"]]          = self.P_s
            grp.attrs[Di_hdf5_planet_label["T_s"]]          = self.T_s
            grp.attrs[Di_hdf5_planet_label["rho_s"]]        = self.rho_s

            # Arrays
            grp.create_dataset(
                Di_hdf5_planet_label["r"], data=self.A1_r, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["m_enc"], data=self.A1_m_enc, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["rho"], data=self.A1_rho, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["T"], data=self.A1_T, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["P"], data=self.A1_P, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["u"], data=self.A1_u, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["mat_id"], data=self.A1_mat_id, dtype="i")

        print("Done")

    def load_planet_profiles(self):
        """ Load the profiles arrays for an existing Planet object from a file.
        """
        Fp_planet = utils.check_end(self.Fp_planet, ".hdf5")

        print("Loading \"%s\"... " % Fp_planet[-60:], end='')
        sys.stdout.flush()

        with h5py.File(Fp_planet, "r") as f:
            (self.A1_r, self.A1_m_enc, self.A1_rho, self.A1_T, self.A1_P,
             self.A1_u, self.A1_mat_id) = multi_get_planet_data(
                f, ["r", "m_enc", "rho", "T", "P", "u", "mat_id"])

        print("Done")

    # ========
    # 1 Layer
    # ========
    def _1_layer_input(self):
        
        assert(self.num_layer == 1)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type_id[0] is not None)
        
    def gen_prof_L1_fix_R_given_M(self):
        """ Compute the profile of a planet with 1 layer by finding the correct
            radius for a given mass.
        """
        # Check for necessary input
        assert(self.R_max is not None)
        assert(self.M is not None)
        self._1_layer_input()


        self.R = L1_spherical.L1_find_radius(
            self.num_prof, self.R_max, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.num_attempt
            )
        self.A1_R_layer[-1] = self.R

        print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L1_spherical.L1_find_mass(
            self.num_prof, self.R, 1.05 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0]
            )

        print("Done!")

        # Integrate the profiles
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L1_spherical.L1_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0]
            )

        self.update_attributes()
        self.print_info()

    def gen_prof_L1_fix_M_given_R(self):
        # Check for necessary input
        assert(self.R is not None or self.A1_R_layer[0] is not None)
        assert(self.M_max is not None)
        assert(len(self.A1_R_layer) == 1)
        self._1_layer_input()
        if self.R is None:
            self.R = self.A1_R_layer[0]

        print("Finding M given R...")

        self.M = L1_spherical.L1_find_mass(
            self.num_prof, self.R, self.M_max, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0]
            )

        print("Done!")

        # Integrate the profiles
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L1_spherical.L1_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0]
            )

        self.update_attributes()
        self.print_info()

    def gen_prof_L1_given_R_M(self):
        # Check for necessary input
        assert(self.R is not None)
        assert(self.M is not None)
        self._1_layer_input()

        # Integrate the profiles
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L1_spherical.L1_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0]
            )

        self.update_attributes()
        self.print_info()

    # ========
    # 2 Layers
    # ========
    def _2_layer_input(self):
        
        assert(self.num_layer == 2)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type_id[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type_id[1] is not None)
        
    def gen_prof_L2_fix_R1_given_R_M(self):
        # Check for necessary input
        assert(self.R is not None)
        assert(self.M is not None)
        self._2_layer_input()

        self.A1_R_layer[0] = L2_spherical.L2_find_R1(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1], self.num_attempt
            )

        print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L2_spherical.L2_find_mass(
            self.num_prof, self.R, 1.05 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L2_spherical.L2_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L2_fix_M_given_R1_R(self):
        # Check for necessary input
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.M_max is not None)
        self._2_layer_input()

        print("Finding M given R1 and R...")

        self.M = L2_spherical.L2_find_mass(
            self.num_prof, self.R, self.M_max, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L2_spherical.L2_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L2_fix_R_given_M_R1(self):
        # Check for necessary input
        assert(self.A1_R_layer[0] is not None)
        assert(self.R_max is not None)
        assert(self.M is not None)
        self._2_layer_input()

        self.R = L2_spherical.L2_find_radius(
            self.num_prof, self.R_max, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1], self.num_attempt
            )
        self.A1_R_layer[-1] = self.R

        print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L2_spherical.L2_find_mass(
            self.num_prof, self.R, 1.05 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )

        print("Done!")

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L2_spherical.L2_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )

        self.update_attributes()
        self.print_info()
        
    def gen_prof_L2_fix_R1_R_given_M1_M2(self):
        
        # Check for necessary input
        self._2_layer_input()
        assert(self.R_max is not None)
        assert(self.A1_M_layer[0] is not None)
        assert(self.A1_M_layer[1] is not None)
        # Check masses
        if self.M is not None:
            assert self.M == self.A1_M_layer[0] + self.A1_M_layer[1]
        else:
            self.M = self.A1_M_layer[0] + self.A1_M_layer[1]

        self.A1_R_layer[0], self.R = L2_spherical.L2_find_R1_R(
            self.num_prof, self.R_max, self.A1_M_layer[0], self.A1_M_layer[1],
            self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1], self.num_attempt
            )
        self.A1_R_layer[-1] = self.R

        print("Tweaking M to avoid peaks at the center of the planet...")
        
        self.M = L2_spherical.L2_find_mass(
            self.num_prof, self.R, 1.05 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )

        print("Done!")

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L2_spherical.L2_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )

        self.update_attributes()
        self.print_info()
        

    def gen_prof_L2_given_R_M_R1(self):
        # Check for necessary input
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.M is not None)
        self._2_layer_input()

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L2_spherical.L2_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )

        self.update_attributes()
        self.print_info()

    # ========
    # 3 Layers
    # ========
    def _3_layer_input(self):
        
        assert(self.num_layer == 3)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type_id[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type_id[1] is not None)
        assert(self.A1_mat_id_layer[2] is not None)
        assert(self.A1_T_rho_type_id[2] is not None)
        
    def gen_prof_L3_fix_R1_R2_given_R_M_I(self):
        # Check for necessary input
        assert(self.R is not None)
        assert(self.M is not None)
        assert(self.I_MR2 is not None)
        self._3_layer_input()

        self.A1_R_layer[0], self.A1_R_layer[1] = L3_spherical.L3_find_R1_R2(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.I_MR2, self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2], self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2], self.num_attempt, self.num_attempt_2
            )

        print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof, self.R, 1.05 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L3_spherical.L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_fix_R2_given_R_M_R1(self):
        # Check for necessary input
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.M is not None)
        self._3_layer_input()

        self.A1_R_layer[1] = L3_spherical.L3_find_R2(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2], self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2], self.num_attempt
            )

        print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof, self.R, 1.05 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L3_spherical.L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_fix_R1_given_R_M_R2(self):
        # Check for necessary input
        assert(self.R is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M is not None)
        self._3_layer_input()

        self.A1_R_layer[0] = L3_spherical.L3_find_R1(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[1], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2], self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2], self.num_attempt
            )

        print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof, self.R, 1.05 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L3_spherical.L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_fix_M_given_R_R1_R2(self):
        # Check for necessary input
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M_max is not None)
        self._3_layer_input()

        print("Finding M given R1, R2 and R...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof, self.R, self.M_max, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L3_spherical.L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_fix_R_given_M_R1_R2(self):
        # Check for necessary input
        assert(self.R_max is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M is not None)
        self._3_layer_input()

        self.R = L3_spherical.L3_find_radius(
            self.num_prof, self.R_max, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2], self.num_attempt
            )
        self.A1_R_layer[-1] = self.R

        print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof, self.R, 1.05 * self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L3_spherical.L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_given_R_M_R1_R2(self):
        # Check for necessary input
        assert(self.R is not None)
        assert(self.A1_R_layer[0] is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M is not None)
        self._3_layer_input()

        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L3_spherical.L3_integrate(
            self.num_prof, self.R, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_R_layer[1], self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1], self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
            )

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_given_prof_L2(self, mat=None, T_rho_type_id=None,
                                  T_rho_args=None, rho_min=None):
        """ Add a third layer (atmosphere) on top of existing 2 layer profiles.

            Args or set attributes:
                ...

            Sets:
                ...
        """
        # Check for necessary input
        assert(self.num_layer == 2)
        assert(self.A1_R_layer[0] is not None)
        assert(self.A1_R_layer[1] is not None)
        assert(self.M is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_T_rho_type_id[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_T_rho_type_id[1] is not None)

        self.num_layer   = 3
        if mat is not None: ###else...?
            self.A1_mat_layer       = np.append(self.A1_mat_layer, mat)
            self.A1_mat_id_layer    = [gv.Di_mat_id[mat]
                                       for mat in self.A1_mat_layer]
        if T_rho_type_id is not None:
            self.A1_T_rho_type_id = np.append(self.A1_T_rho_type_id, T_rho_type_id)
        if T_rho_args is not None:
            A1_T_rho_args_aux = np.zeros((3,2))
            A1_T_rho_args_aux[0:2] = self.A1_T_rho_args
            A1_T_rho_args_aux[2] = np.array(T_rho_args, dtype='float')
            self.A1_T_rho_args = A1_T_rho_args_aux
        if rho_min is not None:
            self.rho_min    = rho_min

        dr              = self.A1_r[1]
        mat_id_L3       = self.A1_mat_id_layer[2]

        # Reverse profile arrays to be ordered by increasing radius
        if self.A1_r[-1] < self.A1_r[0]:
            self.A1_r       = self.A1_r[::-1]
            self.A1_m_enc   = self.A1_m_enc[::-1]
            self.A1_P       = self.A1_P[::-1]
            self.A1_T       = self.A1_T[::-1]
            self.A1_rho     = self.A1_rho[::-1]
            self.A1_u       = self.A1_u[::-1]
            self.A1_mat_id  = self.A1_mat_id[::-1]

        # Initialise the new profiles
        A1_r        = [self.A1_r[-1]]
        A1_m_enc    = [self.A1_m_enc[-1]]
        A1_P        = [self.A1_P[-1]]
        A1_T        = [self.A1_T[-1]]
        A1_u        = [self.A1_u[-1]]
        A1_mat_id   = [mat_id_L3]
        A1_rho      = [eos.rho_P_T(A1_P[0], A1_T[0], mat_id_L3)]

        # Set any T-rho relation variables
        self.A1_T_rho_args[2]   = set_T_rho_args(
            A1_T[0], A1_rho[0], self.A1_T_rho_type_id[2], self.A1_T_rho_args[2],
            mat_id_L3
            )

        # Integrate outwards until the minimum density (or zero pressure)
        step = 1
        while A1_rho[-1] > self.rho_min and A1_P[-1] > 0:
            A1_r.append(A1_r[-1] + dr)
            A1_m_enc.append(A1_m_enc[-1] + 4*np.pi*A1_r[-1]*A1_r[-1]*A1_rho[-1]*dr)
            A1_P.append(A1_P[-1] - gv.G*A1_m_enc[-1]*A1_rho[-1]/(A1_r[-1]**2)*dr)
            # Update T-rho relation variables 
            if (self.A1_T_rho_type_id[2] == gv.type_adb and
                mat_id_L3 == gv.id_HM80_HHe):
                # T_rho_args = [rho_prv, T_prv]
                self.A1_T_rho_args[2]   = [A1_rho[-1], A1_T[-1]]
            rho = eos.find_rho(
                A1_P[-1], mat_id_L3, self.A1_T_rho_type_id[2],
                self.A1_T_rho_args[2], 0.9*A1_rho[-1], A1_rho[-1]
                )
            A1_rho.append(rho)
            A1_T.append(T_rho(rho, self.A1_T_rho_type_id[2], 
                                   self.A1_T_rho_args[2], mat_id_L3))
            A1_u.append(eos.u_rho_T(rho, A1_T[-1], mat_id_L3))
            A1_mat_id.append(mat_id_L3)

            step += 1
            if step >= self.num_prof:
                print("Layer 3 goes out too far!")
                break

        # Apppend the new layer to the profiles, removing the final too-low 
        # density or non-positive pressure step
        self.A1_r       = np.append(self.A1_r, A1_r[1:-1])
        self.A1_m_enc   = np.append(self.A1_m_enc, A1_m_enc[1:-1])
        self.A1_P       = np.append(self.A1_P, A1_P[1:-1])
        self.A1_T       = np.append(self.A1_T, A1_T[1:-1])
        self.A1_rho     = np.append(self.A1_rho, A1_rho[1:-1])
        self.A1_u       = np.append(self.A1_u, A1_u[1:-1])
        self.A1_mat_id  = np.append(self.A1_mat_id, A1_mat_id[1:-1])

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_fix_R1_R2_given_M1_M2_add_L3(
        self, M1=None, M2=None, R_min=None, R_max=None, M_frac_tol=None,
        rho_min=None
        ):
        """ Generate a 3 layer profile by first finding the inner 2 layer
            profile using the masses of each layer then add the third layer
            (atmosphere) on top.

            Note: the input T_s, P_s, rho_s here are used for the outer boundary
            of layer 2. They will then be overwritten with the final values
            after layer 3 is added.

            Args or set attributes:
                ...

            Sets:
                ...
        """
        # Check for necessary input
        if M1 is not None:
            self.A1_M_layer[0]  = M1
        if M2 is not None:
            self.A1_M_layer[1]  = M2
        if R_min is not None:
            self.R_min  = R_min
        if R_max is not None:
            self.R_max  = R_max
        if M_frac_tol is not None:
            self.M_frac_tol = M_frac_tol
        if rho_min is not None:
            self.rho_min    = rho_min
        assert(self.num_layer == 3)
        assert(self.A1_M_layer[0] is not None)
        assert(self.A1_M_layer[1] is not None)
        assert(self.P_s is not None)
        assert(self.T_s is not None)
        assert(self.rho_s is not None)
        assert(self.A1_mat_id_layer[0] is not None)
        assert(self.A1_mat_id_layer[1] is not None)
        assert(self.A1_mat_id_layer[2] is not None)
        assert(self.A1_T_rho_type_id[0] is not None)
        assert(self.A1_T_rho_type_id[1] is not None)
        assert(self.A1_T_rho_type_id[2] is not None)

        # Update R_min and R_max without changing the attributes
        R_min   = self.R_min
        R_max   = self.R_max

        # Store the layer 3 properties
        mat_L3          = self.A1_mat_layer[2]
        T_rho_type_L3   = self.A1_T_rho_type_id[2]
        T_rho_args_L3   = self.A1_T_rho_args[2]

        # Temporarily set self to be 2 layer planet
        self.num_layer          = 2
        self.A1_M_layer         = self.A1_M_layer[:2]
        self.A1_R_layer         = self.A1_R_layer[:2]
        self.A1_mat_layer       = self.A1_mat_layer[:2]
        self.A1_mat_id_layer    = self.A1_mat_id_layer[:2]
        self.A1_T_rho_type_id      = self.A1_T_rho_type_id[:2]
        self.A1_T_rho_args      = self.A1_T_rho_args[:2]
        self.rho_s              = eos.rho_P_T(
            self.P_s, self.T_s, self.A1_mat_id_layer[-1])
        ###what if T_s or P_s was derived instead?

        # Find the radii of the inner 2 layers in isolation
        M_tot   = np.sum(self.A1_M_layer)
        ###Replace with a function like the identical L2 version

        # Check the maximum radius yields a too small layer 1 mass
        print("R_max  = %.5g m  = %.5g R_E " % (R_max, R_max / gv.R_earth))
        self.A1_R_layer[0] = L2_spherical.L2_find_R1(
            self.num_prof, R_max, M_tot, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1], 
            int(self.num_attempt / 2)
            )
        self.M = L2_spherical.L2_find_mass(
            self.num_prof, R_max, 2 * M_tot, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L2_spherical.L2_integrate(
            self.num_prof, R_max, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )
        M1  = self.A1_m_enc[self.A1_mat_id == self.A1_mat_id_layer[0]][0]
        print("    --> M1  = %.5g kg  = %.5g M_E  = %.4f M_tot"
              % (M1, M1 / gv.M_earth, M1 / M_tot))
        assert M1 < self.A1_M_layer[0], \
            "R_max must be big enough to yield a too low M1"

        # Check the minimum radius yields a too large layer 1 mass
        print("R_min  = %.5g m  = %.5g R_E " % (R_min, R_min / gv.R_earth))
        self.A1_R_layer[0] = L2_spherical.L2_find_R1(
            self.num_prof, R_min, M_tot, self.P_s, self.T_s, self.rho_s,
            self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
            int(self.num_attempt / 2)
            )
        self.M = L2_spherical.L2_find_mass(
            self.num_prof, R_min, 2 * M_tot, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )
        (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
         self.A1_mat_id) = L2_spherical.L2_integrate(
            self.num_prof, R_min, self.M, self.P_s, self.T_s, self.rho_s,
            self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
            )
        M1  = self.A1_m_enc[self.A1_mat_id == self.A1_mat_id_layer[0]][0]
        print("    --> M1  = %.5g kg  = %.5g M_E  = %.4f M_tot"
              % (M1, M1 / gv.M_earth, M1 / M_tot))
        assert M1 > self.A1_M_layer[0], \
            "R_min must be small enough to yield a too high M1"

        # Iterate to obtain desired layer masses
        iter    = 0
        M1      = 0
        while self.M_frac_tol < abs(M1 - self.A1_M_layer[0]) / M_tot:
            R_try = (R_min + R_max) * 0.5
            print("iter %d: R  = %.5g m  = %.5g R_E"
                  % (iter, R_try, R_try / gv.R_earth))

            self.A1_R_layer[0] = L2_spherical.L2_find_R1(
                self.num_prof, R_try, M_tot, self.P_s, self.T_s, self.rho_s,
                self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
                self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
                self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
                self.num_attempt
                )
            self.M = L2_spherical.L2_find_mass(
                self.num_prof, R_try, 2 * M_tot, self.P_s, self.T_s, self.rho_s,
                self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
                self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
                self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
                )
            (self.A1_r, self.A1_m_enc, self.A1_P, self.A1_T, self.A1_rho, self.A1_u,
             self.A1_mat_id) = L2_spherical.L2_integrate(
                self.num_prof, R_try, self.M, self.P_s, self.T_s, self.rho_s,
                self.A1_R_layer[0], self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0],
                self.A1_T_rho_args[0], self.A1_mat_id_layer[1],
                self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
                )
            M1  = self.A1_m_enc[self.A1_mat_id == self.A1_mat_id_layer[0]][0]
            print("    --> M1  = %.5g kg  = %.5g M_E  = %.4f M_tot"
                  % (M1, M1 / gv.M_earth, M1 / M_tot))

            if M1 < self.A1_M_layer[0]:
                R_max = R_try
            else:
                R_min = R_try

            iter    += 1

        self.update_attributes()

        # Add the third layer
        self.gen_prof_L3_given_prof_L2(
            mat            = mat_L3,
            T_rho_type_id  = T_rho_type_L3,
            T_rho_args     = T_rho_args_L3,
            rho_min        = self.rho_min,
            )

# ============================================================================ #
#                       Spining profile classes                                #
# ============================================================================ #

class SpinPlanet():

    def __init__(
    self, name=None, planet=None, Fp_planet=None, Tw=None,
    A1_mat_layer=None, A1_R_layer=None,
    A1_T_rho_type=None, A1_T_rho_args=None, A1_r=None, A1_P=None, A1_T=None,
    A1_rho=None, num_prof=1000, num_attempt=15, R_e_max=None, R_p_max=None
    ):
        self.name               = name
        self.Fp_planet          = Fp_planet
        self.num_prof           = num_prof
        self.num_attempt        = num_attempt
        self.R_e_max            = R_e_max
        self.R_p_max            = R_p_max
        self.Tw                 = Tw
        self.P_1                = None
        self.P_2                = None
        self.M                  = None

        if planet is not None:
            self.planet           = planet
            self.num_layer        = planet.num_layer
            self.A1_mat_layer     = planet.A1_mat_layer
            self.A1_R_layer       = planet.A1_R_layer
            self.A1_mat_id_layer  = planet.A1_mat_id_layer
            self.A1_T_rho_type    = planet.A1_T_rho_type
            self.A1_T_rho_type_id = planet.A1_T_rho_type_id
            self.A1_T_rho_args    = planet.A1_T_rho_args
            self.A1_r             = planet.A1_r
            self.A1_P             = planet.A1_P
            self.A1_T             = planet.A1_T
            self.A1_rho           = planet.A1_rho
            self.M                = planet.M

        else:
            self.A1_R_layer      = A1_R_layer
            self.A1_mat_layer    = A1_mat_layer
            self.A1_T_rho_type   = A1_T_rho_type
            self.A1_T_rho_args   = A1_T_rho_args
            self.A1_r            = A1_r
            self.A1_P            = A1_P
            self.A1_T            = A1_T
            self.A1_rho          = A1_rho

            # Derived or default attributes
            if self.A1_mat_layer is not None:
                self.num_layer          = len(self.A1_mat_layer)
                self.A1_mat_id_layer    = [gv.Di_mat_id[mat]
                                           for mat in self.A1_mat_layer]
                
            if self.A1_T_rho_type is not None:
                self.A1_T_rho_type_id = [gv.Di_T_rho_id[T_rho_id]
                                         for T_rho_id in self.A1_T_rho_type]
                
        # Set default R_e_max and R_p_max
        assert(self.A1_r is not None)
        if self.R_e_max is None:
            self.R_e_max = 1.5*np.max(self.A1_r)
            
        if self.R_p_max is None:
            self.R_p_max = 1.2*np.max(self.A1_r)

        assert(self.num_layer in [1, 2, 3])
        assert(self.R_e_max is not None)
        assert(self.R_p_max is not None)

        self.A1_r_equator     = np.linspace(0, self.R_e_max, self.num_prof)
        self.A1_r_pole        = np.linspace(0, self.R_p_max, self.num_prof)
        
    def _check_input(self):
        
        if self.num_layer == 1:
            assert(self.A1_mat_id_layer[0] is not None)
            assert(self.A1_T_rho_type_id[0] is not None)
        elif self.num_layer == 2:
            assert(self.A1_mat_id_layer[0] is not None)
            assert(self.A1_T_rho_type_id[0] is not None)
            assert(self.A1_mat_id_layer[1] is not None)
            assert(self.A1_T_rho_type_id[1] is not None)
        elif self.num_layer == 3:
            assert(self.A1_mat_id_layer[0] is not None)
            assert(self.A1_T_rho_type_id[0] is not None)
            assert(self.A1_mat_id_layer[1] is not None)
            assert(self.A1_T_rho_type_id[1] is not None)
            assert(self.A1_mat_id_layer[2] is not None)
            assert(self.A1_T_rho_type_id[2] is not None)
            
    def update_attributes(self):
        # Compute mass of the planet
        self.M = us.compute_spin_planet_M(self.A1_r_equator, self.A1_rho_equator,
                                          self.A1_r_pole, self.A1_rho_pole)
        
        # Compute escape velocity
        v_esc_eq, v_esc_p = us.spin_escape_vel(self.A1_r_equator, self.A1_rho_equator,
                                               self.A1_r_pole, self.A1_rho_pole, self.Tw)
        
        self.v_escape_pole    = v_esc_p
        self.v_escape_equator = v_esc_eq
        
        # Compute equatorial and polar radius
        self.R_e = np.max(self.A1_r_equator[self.A1_rho_equator > 0.])
        self.R_p = np.max(self.A1_r_pole[self.A1_rho_pole > 0.])
        
        # Mass per layer, Equatorial and polar temperature and pressure
        if self.num_layer == 1:
            self.A1_R_layer_equator = np.array([self.R_e])
            self.A1_R_layer_pole    = np.array([self.R_p])
            # Mass
            self.A1_M_layer = np.array([self.M])
            # Pressure and temperature
            self.A1_P_equator = np.zeros_like(self.A1_r_equator)
            self.A1_P_pole    = np.zeros_like(self.A1_r_pole)
            self.A1_T_equator = np.zeros_like(self.A1_r_equator)
            self.A1_T_pole    = np.zeros_like(self.A1_r_pole)
            for i, rho in enumerate(self.A1_rho_equator):
                if rho >= self.rho_s:
                    self.A1_T_equator[i] = T_rho(rho, self.A1_T_rho_type_id[0],
                                                 self.A1_T_rho_args[0], self.A1_mat_id_layer[0])
                    self.A1_P_equator[i] = eos.P_T_rho(self.A1_T_equator[i],
                                                       rho, self.A1_mat_id_layer[0])
            for i, rho in enumerate(self.A1_rho_pole):
                if rho >= self.rho_s:
                    self.A1_T_pole[i] = T_rho(rho, self.A1_T_rho_type_id[0],
                                              self.A1_T_rho_args[0], self.A1_mat_id_layer[0])
                    self.A1_P_pole[i] = eos.P_T_rho(self.A1_T_pole[i],
                                                    rho, self.A1_mat_id_layer[0])
            # Mat_id
            self.A1_mat_id_equator = np.ones(self.A1_r_equator.shape)*self.A1_mat_id_layer[0]
            self.A1_mat_id_pole    = np.ones(self.A1_r_pole.shape)*self.A1_mat_id_layer[0]
                
        elif self.num_layer == 2:
            self.R_1_equator        = np.max(self.A1_r_equator[self.A1_rho_equator >= self.rho_1])
            self.A1_R_layer_equator = np.array([self.R_1_equator, self.R_e])
            self.R_1_pole           = np.max(self.A1_r_pole[self.A1_rho_pole >= self.rho_1])
            self.A1_R_layer_pole    = np.array([self.R_1_pole, self.R_p])
            self.A1_mat_id_equator  = (self.A1_rho_equator >= self.rho_1)*self.A1_mat_id_layer[0] + \
                                      (self.A1_rho_equator < self.rho_1)*self.A1_mat_id_layer[1]
            self.A1_mat_id_pole     = (self.A1_rho_pole >= self.rho_1)*self.A1_mat_id_layer[0] + \
                                      (self.A1_rho_pole < self.rho_1)*self.A1_mat_id_layer[1]
                                      
            self.A1_mat_id_equator[self.A1_rho_equator <= 0] = -1
            self.A1_mat_id_pole[self.A1_rho_pole <= 0]       = -1
            
            self.A1_P_equator = np.zeros_like(self.A1_r_equator)
            self.A1_P_pole    = np.zeros_like(self.A1_r_pole)
            self.A1_T_equator = np.zeros_like(self.A1_r_equator)
            self.A1_T_pole    = np.zeros_like(self.A1_r_pole)
            
            for i, rho in enumerate(self.A1_rho_equator):
                if rho >= self.rho_1:
                    self.A1_T_equator[i] = T_rho(rho, self.A1_T_rho_type_id[0],
                                                 self.A1_T_rho_args[0], self.A1_mat_id_layer[0])
                    self.A1_P_equator[i] = eos.P_T_rho(self.A1_T_equator[i],
                                                       rho, self.A1_mat_id_layer[0])
                elif rho >= self.rho_s:
                    self.A1_T_equator[i] = T_rho(rho, self.A1_T_rho_type_id[1],
                                                 self.A1_T_rho_args[1], self.A1_mat_id_layer[1])
                    self.A1_P_equator[i] = eos.P_T_rho(self.A1_T_equator[i],
                                                       rho, self.A1_mat_id_layer[1])
            for i, rho in enumerate(self.A1_rho_pole):
                if rho >= self.rho_1:
                    self.A1_T_pole[i] = T_rho(rho, self.A1_T_rho_type_id[0],
                                              self.A1_T_rho_args[0], self.A1_mat_id_layer[0])
                    self.A1_P_pole[i] = eos.P_T_rho(self.A1_T_pole[i],
                                                    rho, self.A1_mat_id_layer[0])
                elif rho >= self.rho_s:
                    self.A1_T_pole[i] = T_rho(rho, self.A1_T_rho_type_id[1],
                                              self.A1_T_rho_args[1], self.A1_mat_id_layer[1])
                    self.A1_P_pole[i] = eos.P_T_rho(self.A1_T_pole[i],
                                                    rho, self.A1_mat_id_layer[1])
            
            r_temp = np.copy(self.A1_r_equator)
            z_temp = np.copy(self.A1_r_pole)
            rho_r_temp = np.copy(self.A1_rho_equator)
            rho_z_temp = np.copy(self.A1_rho_pole)
            rho_r_temp[rho_r_temp < self.rho_1] = 0.
            rho_z_temp[rho_z_temp < self.rho_1] = 0.
            M1 =  us.compute_spin_planet_M(r_temp, rho_r_temp, z_temp, rho_z_temp)

            M2 = self.M - M1
            
            self.A1_M_layer = np.array([M1, M2])
            
        elif self.num_layer == 3:
            self.R_1_equator        = np.max(self.A1_r_equator[self.A1_rho_equator >= self.rho_1])
            self.R_2_equator        = np.max(self.A1_r_equator[self.A1_rho_equator >= self.rho_2])
            self.A1_R_layer_equator = np.array([self.R_1_equator, self.R_2_equator, self.R_e])
            self.R_1_pole           = np.max(self.A1_r_pole[self.A1_rho_pole >= self.rho_1])
            self.R_2_pole           = np.max(self.A1_r_pole[self.A1_rho_pole >= self.rho_2])
            self.A1_R_layer_pole    = np.array([self.R_1_pole, self.R_2_pole, self.R_p])
            self.A1_mat_id_equator  = (self.A1_rho_equator >= self.rho_1)*self.A1_mat_id_layer[0] + \
                                      np.logical_and(self.A1_rho_equator < self.rho_1,
                                                     self.A1_rho_equator >= self.rho_2)*self.A1_mat_id_layer[1] + \
                                      (self.A1_rho_equator < self.rho_2)*self.A1_mat_id_layer[2]
            self.A1_mat_id_equator  = (self.A1_rho_pole >= self.rho_1)*self.A1_mat_id_layer[0] + \
                                       np.logical_and(self.A1_rho_pole < self.rho_1,
                                                     self.A1_rho_pole >= self.rho_2)*self.A1_mat_id_layer[1] + \
                                      (self.A1_rho_pole < self.rho_2)*self.A1_mat_id_layer[2]
                                      
            self.A1_mat_id_equator[self.A1_rho_equator <= 0] = -1
            self.A1_mat_id_pole[self.A1_rho_pole <= 0]       = -1
            
            self.A1_P_equator = np.zeros_like(self.A1_r_equator)
            self.A1_P_pole    = np.zeros_like(self.A1_r_pole)
            self.A1_T_equator = np.zeros_like(self.A1_r_equator)
            self.A1_T_pole    = np.zeros_like(self.A1_r_pole)
            
            for i, rho in enumerate(self.A1_rho_equator):
                if rho >= self.rho_1:
                    self.A1_T_equator[i] = T_rho(rho, self.A1_T_rho_type_id[0],
                                                 self.A1_T_rho_args[0], self.A1_mat_id_layer[0])
                    self.A1_P_equator[i] = eos.P_T_rho(self.A1_T_equator[i],
                                                       rho, self.A1_mat_id_layer[0])
                elif rho >= self.rho_2:
                    self.A1_T_equator[i] = T_rho(rho, self.A1_T_rho_type_id[1],
                                                 self.A1_T_rho_args[1], self.A1_mat_id_layer[1])
                    self.A1_P_equator[i] = eos.P_T_rho(self.A1_T_equator[i],
                                                       rho, self.A1_mat_id_layer[1])
                elif rho >= self.rho_s:
                    self.A1_T_equator[i] = T_rho(rho, self.A1_T_rho_type_id[2],
                                                 self.A1_T_rho_args[2], self.A1_mat_id_layer[2])
                    self.A1_P_equator[i] = eos.P_T_rho(self.A1_T_equator[i],
                                                       rho, self.A1_mat_id_layer[2])
            for i, rho in enumerate(self.A1_rho_pole):
                if rho >= self.rho_1:
                    self.A1_T_pole[i] = T_rho(rho, self.A1_T_rho_type_id[0],
                                                 self.A1_T_rho_args[0], self.A1_mat_id_layer[0])
                    self.A1_P_pole[i] = eos.P_T_rho(self.A1_T_pole[i],
                                                    rho, self.A1_mat_id_layer[0])
                elif rho >= self.rho_2:
                    self.A1_T_pole[i] = T_rho(rho, self.A1_T_rho_type_id[1],
                                                 self.A1_T_rho_args[1], self.A1_mat_id_layer[1])
                    self.A1_P_pole[i] = eos.P_T_rho(self.A1_T_pole[i],
                                                    rho, self.A1_mat_id_layer[1])
                elif rho >= self.rho_s:
                    self.A1_T_pole[i] = T_rho(rho, self.A1_T_rho_type_id[2],
                                                 self.A1_T_rho_args[2], self.A1_mat_id_layer[2])
                    self.A1_P_pole[i] = eos.P_T_rho(self.A1_T_pole[i],
                                                    rho, self.A1_mat_id_layer[2])
                    
            r_temp = np.copy(self.A1_r_equator)
            z_temp = np.copy(self.A1_r_pole)
            rho_r_temp = np.copy(self.A1_rho_equator)
            rho_z_temp = np.copy(self.A1_rho_pole)
            rho_r_temp[rho_r_temp < self.rho_1] = 0.
            rho_z_temp[rho_z_temp < self.rho_1] = 0.
            M1 =  us.compute_spin_planet_M(r_temp, rho_r_temp, z_temp, rho_z_temp)
            
            rho_r_temp = np.copy(self.A1_rho_equator)
            rho_z_temp = np.copy(self.A1_rho_pole)
            rho_r_temp[rho_r_temp < self.rho_2] = 0.
            rho_z_temp[rho_z_temp < self.rho_2] = 0.
            M2 =  us.compute_spin_planet_M(r_temp, rho_r_temp, z_temp, rho_z_temp)
            M2 = M2 - M1
            
            M3 = self.M - M2 - M1
            
            self.A1_M_layer = np.array([M1, M2, M3])
             
        self.T_0 = self.A1_T_equator[0]
        self.T_s = self.A1_T_equator[self.A1_T_equator > 0][-1]
        
    def print_info(self):
        """ Print the Planet objects's main properties. """
        # Print and catch if any variables are None
        def print_try(string, variables):
            try:
                print(string % variables)
            except TypeError:
                print("    %s = None" % variables[0])
        
        space   = 12
        print_try("Planet \"%s\": ", self.name)
        print_try("    %s = %.5g kg = %.5g M_earth",
                  (utils.add_whitespace("M", space), self.M, self.M/gv.M_earth))
        print_try("    %s = %.5g m = %.5g R_earth",
                  (utils.add_whitespace("R_equator", space), self.R_e, self.R_e/gv.R_earth))
        print_try("    %s = %.5g m = %.5g R_earth",
                  (utils.add_whitespace("R_pole", space), self.R_p, self.R_p/gv.R_earth))
        print_try("    %s = %s ", (utils.add_whitespace("mat", space),
                  utils.format_array_string(self.A1_mat_layer, "string")))
        print_try("    %s = %s ", (utils.add_whitespace("mat_id", space),
                  utils.format_array_string(self.A1_mat_id_layer, "%d")))
        print_try("    %s = %s R_earth", (utils.add_whitespace("R_layer_eq", space),
                  utils.format_array_string(self.A1_R_layer_equator / gv.R_earth, "%.5g")))
        print_try("    %s = %s R_earth", (utils.add_whitespace("R_layer_pole", space),
                  utils.format_array_string(self.A1_R_layer_pole / gv.R_earth, "%.5g")))
        print_try("    %s = %s M_earth", (utils.add_whitespace("M_layer", space),
                  utils.format_array_string(self.A1_M_layer / gv.M_earth, "%.5g")))
        print_try("    %s = %s M_total", (utils.add_whitespace("M_frac_layer", space),
                  utils.format_array_string(self.A1_M_layer / self.M, "%.5g")))
        print_try("    %s = %.5g Pa", (utils.add_whitespace("P_s", space), self.P_s))
        print_try("    %s = %.5g K", (utils.add_whitespace("T_s", space), self.T_s))
        print_try("    %s = %.5g kg/m^3", 
                  (utils.add_whitespace("rho_s", space), self.rho_s))
        if self.num_layer > 2:
            print_try("    %s = %.5g Pa",
                      (utils.add_whitespace("P_2", space), self.P_2))
            print_try("    %s = %.5g K", 
                      (utils.add_whitespace("T_2", space), self.T_2))
            print_try("    %s = %.5g kg/m^3", 
                      (utils.add_whitespace("rho_2", space), self.rho_2))
        if self.num_layer > 1:
            print_try("    %s = %.5g Pa", 
                      (utils.add_whitespace("P_1", space), self.P_1))
            print_try("    %s = %.5g K", 
                      (utils.add_whitespace("T_1", space), self.T_1))
            print_try("    %s = %.5g kg/m^3", 
                      (utils.add_whitespace("rho_1", space), self.rho_1))
        print_try("    %s = %.5g Pa", (utils.add_whitespace("P_0", space), self.P_0))
        print_try("    %s = %.5g K", (utils.add_whitespace("T_0", space), self.T_0))
        print_try("    %s = %.5g kg/m^3", 
                  (utils.add_whitespace("rho_0", space), self.rho_0))
# =============================================================================
#         print_try("    %s = %.5g M_tot*R_tot^2",
#                   (utils.add_whitespace("I_MR2", space), 
#                    self.I_MR2/self.M/self.R/self.R))
# =============================================================================
        
    def find_Tw_min(self, Tw_max=10, iterations=20):
        
        Tw_min = 0.

        P_c   = np.max(self.A1_P)
        P_s   = np.min(self.A1_P)
        rho_c = np.max(self.A1_rho)
        rho_s = np.min(self.A1_rho)
        
        # Check for necessary input
        self._check_input()
        
        # Other necessary input in case of 2 and 3 layers
        if self.num_layer == 2:
            
            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[0]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[0]])
            P_boundary = 0.5*(a + b)
            
            
        elif self.num_layer == 3:
            
            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[0]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[0]])
            P_boundary_12 = 0.5*(a + b)

            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[1]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[1]])
            P_boundary_23 = 0.5*(a + b)
            
        for k in tqdm(range(iterations), desc="Finding Tw min:"):
            
            Tw_try = np.mean([Tw_min, Tw_max])
            
            if self.num_layer == 1:
                profile_e, profile_p = \
                    L1_spin.spin1layer(1, self.A1_r_equator, self.A1_r_pole,
                                self.A1_r, self.A1_rho, Tw_try,
                                P_c, P_s, rho_c, rho_s,
                                self.A1_mat_id_layer[0],
                                self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
                                verbose=0
                                )
                    
            elif self.num_layer == 2:  
                profile_e, profile_p = \
                    L2_spin.spin2layer(1, self.A1_r_equator, self.A1_r_pole,
                               self.A1_r, self.A1_rho, Tw_try,
                               P_c, P_boundary, P_s,
                               rho_c, rho_s,
                               self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
                               self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
                               verbose=0
                               )
                    
            elif self.num_layer == 3:
                profile_e, profile_p = \
                    L3_spin.spin3layer(1, self.A1_r_equator, self.A1_r_pole,
                               self.A1_r, self.A1_rho, Tw_try,
                               P_c, P_boundary_12, P_boundary_23, P_s,
                               rho_c, rho_s,
                               self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
                               self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
                               self.A1_mat_id_layer[2], self.A1_T_rho_type_id[2], self.A1_T_rho_args[2],
                               verbose=0
                               )
                
            if profile_e[-1][-1] > 0:
                Tw_min = Tw_try     
            else:
                Tw_max = Tw_try
            
        self.Tw_min = Tw_max


    def spin(self):
        # Check for necessary input
        self._check_input()
        assert(self.Tw is not None)

        P_c   = np.max(self.A1_P)
        P_s   = np.min(self.A1_P)
        rho_c = np.max(self.A1_rho)
        rho_s = np.min(self.A1_rho)
        
        self.P_0 = P_c
        self.P_s = P_s
        self.rho_0 = rho_c
        self.rho_s = rho_s
        
        # Use correct function
        if self.num_layer == 1:

            profile_e, profile_p = \
                L1_spin.spin1layer(self.num_attempt, self.A1_r_equator, self.A1_r_pole,
                                   self.A1_r, self.A1_rho, self.Tw,
                                   P_c, P_s, rho_c, rho_s,
                                   self.A1_mat_id_layer[0],
                                   self.A1_T_rho_type_id[0],
                                   self.A1_T_rho_args[0]
                                   )

        elif self.num_layer == 2:

            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[0]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[0]])
            P_boundary = 0.5*(a + b)

            self.P_1   = P_boundary
            self.rho_1 = np.min(self.A1_rho[self.A1_r <= self.A1_R_layer[0]])
            self.T_1   = T_rho(self.rho_1, self.A1_T_rho_type_id[0],
                               self.A1_T_rho_args[0], self.A1_mat_id_layer[0])

            profile_e, profile_p = \
                L2_spin.spin2layer(self.num_attempt, self.A1_r_equator, self.A1_r_pole,
                           self.A1_r, self.A1_rho, self.Tw,
                           P_c, P_boundary, P_s,
                           rho_c, rho_s,
                           self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
                           self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1], self.A1_T_rho_args[1]
                           )

        elif self.num_layer == 3:

            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[0]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[0]])
            P_boundary_12 = 0.5*(a + b)

            self.P_1   = P_boundary_12
            self.rho_1 = np.min(self.A1_rho[self.A1_r <= self.A1_R_layer[0]])
            self.T_1   = T_rho(self.rho_1, self.A1_T_rho_type_id[0],
                               self.A1_T_rho_args[0], self.A1_mat_id_layer[0])

            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[1]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[1]])
            P_boundary_23 = 0.5*(a + b)

            self.P_2   = P_boundary_23
            self.rho_2 = np.min(self.A1_rho[self.A1_r <= self.A1_R_layer[1]])
            self.T_2   = T_rho(self.rho_1, self.A1_T_rho_type_id[1],
                               self.A1_T_rho_args[1], self.A1_mat_id_layer[1])

            profile_e, profile_p = \
                L3_spin.spin3layer(self.num_attempt, self.A1_r_equator, self.A1_r_pole,
                           self.A1_r, self.A1_rho, self.Tw,
                           P_c, P_boundary_12, P_boundary_23, P_s,
                           rho_c, rho_s,
                           self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
                           self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
                           self.A1_mat_id_layer[2], self.A1_T_rho_type_id[2], self.A1_T_rho_args[2]
                           )

        print("\nDone!")

        # Save profile
        self.A1_rho_equator   = profile_e[-1]
        self.A1_rho_pole      = profile_p[-1]
        
        self.update_attributes()
        self.print_info()
        
        #def spin_fixed_M(self):
            
            #assert self.M is not None

class GenSpheroid():

    def __init__(
    self, name=None, spin_planet=None, Fp_planet=None, Tw=None,
    A1_mat_layer=None, A1_T_rho_args=None, A1_T_rho_type=None,
    A1_r_equator=None, A1_rho_equator=None,
    A1_r_pole=None, A1_rho_pole=None, P_1=None, P_2=None,
    N_particles = None, N_neig=48, num_attempt=10,
    A1_r=None, A1_rho=None, A1_P=None
    ):
        self.name               = name
        self.Fp_planet          = Fp_planet
        self.N_particles        = N_particles
        self.num_attempt        = num_attempt

        if spin_planet is not None:
            self.num_layer        = spin_planet.num_layer
            self.A1_mat_layer     = spin_planet.A1_mat_layer
            self.A1_mat_id_layer  = spin_planet.A1_mat_id_layer
            self.A1_T_rho_type    = spin_planet.A1_T_rho_type
            self.A1_T_rho_type_id = spin_planet.A1_T_rho_type_id
            self.A1_T_rho_args    = spin_planet.A1_T_rho_args
            self.A1_r_equator     = spin_planet.A1_r_equator
            self.A1_rho_equator   = spin_planet.A1_rho_equator
            self.A1_r_pole        = spin_planet.A1_r_pole
            self.A1_rho_pole      = spin_planet.A1_rho_pole
            self.Tw               = spin_planet.Tw
            self.P_1              = spin_planet.P_1
            self.P_2              = spin_planet.P_2
            self.A1_r             = spin_planet.A1_r
            self.A1_rho           = spin_planet.A1_rho
            self.A1_P             = spin_planet.A1_P

        else:
            self.A1_mat_layer    = A1_mat_layer
            self.A1_T_rho_type   = A1_T_rho_type
            self.A1_T_rho_args   = A1_T_rho_args
            self.A1_r_equator    = A1_r_equator
            self.A1_rho_equator  = A1_rho_equator
            self.A1_r_pole       = A1_r_pole
            self.A1_rho_pole     = A1_rho_pole
            self.Tw              = Tw
            self.P_1             = P_1
            self.P_2             = P_2
            self.A1_r            = A1_r
            self.A1_rho          = A1_rho
            self.A1_P            = A1_P

            # Derived or default attributes
            if self.A1_mat_layer is not None:
                self.num_layer          = len(self.A1_mat_layer)
                self.A1_mat_id_layer    = [gv.Di_mat_id[mat]
                                           for mat in self.A1_mat_layer]
                
            if self.A1_T_rho_type is not None:
                self.A1_T_rho_type_id = [gv.Di_T_rho_id[T_rho_id]
                                         for T_rho_id in self.A1_T_rho_type]
                
            

        assert(self.num_layer in [1, 2, 3])
        assert(self.N_particles is not None)

        if self.num_layer == 1:

            x, y, z, vx, vy, vz, m, rho, u, P, h, mat_id, picle_id = \
            L1_spin.picle_placement_L1(self.A1_r_equator, self.A1_rho_equator,
                                       self.A1_r_pole, self.A1_rho_pole, self.Tw, self.N_particles,
                                       self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
                                       N_neig)

            self.A1_picle_x      = x
            self.A1_picle_y      = y
            self.A1_picle_z      = z
            self.A1_picle_vx     = vx
            self.A1_picle_vy     = vy
            self.A1_picle_vz     = vz
            self.A1_picle_m      = m
            self.A1_picle_rho    = rho
            self.A1_picle_u      = u
            self.A1_picle_P      = P
            self.A1_picle_h      = h
            self.A1_picle_mat_id = mat_id
            self.A1_picle_id     = picle_id
            self.N_particles     = x.shape[0]

        elif self.num_layer == 2:

            rho_P_model       = interp1d(self.A1_P, self.A1_rho)
            self.rho_1       = rho_P_model(self.P_1)

            x, y, z, vx, vy, vz, m, rho, u, P, h, mat_id, picle_id = \
                L2_spin.picle_placement_L2(self.A1_r_equator, self.A1_rho_equator,
                                           self.A1_r_pole, self.A1_rho_pole,
                                           self.Tw, self.N_particles, self.rho_1,
                                           self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
                                           self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
                                           N_neig)

            self.A1_picle_x      = x
            self.A1_picle_y      = y
            self.A1_picle_z      = z
            self.A1_picle_vx     = vx
            self.A1_picle_vy     = vy
            self.A1_picle_vz     = vz
            self.A1_picle_m      = m
            self.A1_picle_rho    = rho
            self.A1_picle_u      = u
            self.A1_picle_P      = P
            self.A1_picle_h      = h
            self.A1_picle_mat_id = mat_id
            self.A1_picle_id     = picle_id
            self.N_particles     = x.shape[0]

        elif self.num_layer == 3:

            rho_P_model  = interp1d(self.A1_P, self.A1_rho)
            self.rho_1  = rho_P_model(self.P_1)
            self.rho_2  = rho_P_model(self.P_2)

            x, y, z, vx, vy, vz, m, rho, u, P, h, mat_id, picle_id = \
                L3_spin.picle_placement_L3(self.A1_r_equator, self.A1_rho_equator,
                                           self.A1_r_pole, self.A1_rho_pole,
                                           self.Tw, self.N_particles, self.rho_1, self.rho_2,
                                           self.A1_mat_id_layer[0], self.A1_T_rho_type_id[0], self.A1_T_rho_args[0],
                                           self.A1_mat_id_layer[1], self.A1_T_rho_type_id[1], self.A1_T_rho_args[1],
                                           self.A1_mat_id_layer[2], self.A1_T_rho_type_id[2], self.A1_T_rho_args[2],
                                           N_neig)

            self.A1_picle_x      = x
            self.A1_picle_y      = y
            self.A1_picle_z      = z
            self.A1_picle_vx     = vx
            self.A1_picle_vy     = vy
            self.A1_picle_vz     = vz
            self.A1_picle_m      = m
            self.A1_picle_rho    = rho
            self.A1_picle_u      = u
            self.A1_picle_P      = P
            self.A1_picle_h      = h
            self.A1_picle_mat_id = mat_id
            self.A1_picle_id     = picle_id
            self.N_particles     = x.shape[0]
