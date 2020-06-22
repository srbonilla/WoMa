"""
WoMa (World Maker)
====

Create models of rotating (and non-rotating) planets (or stars etc.) by solving 
the differential equations for hydrostatic equilibrium, and/or create initial 
conditions for smoothed particle hydrodynamics (SPH) or any other particle-based 
methods by placing particles to precisely match the planet's profiles.

Presented in Ruiz-Bonilla et al. (2020), MNRAS..., https://doi.org/...

Includes SEAGen (https://github.com/jkeger/seagen; Kegerreis et al. 2019, MNRAS 
487:4) with modifications for spinning planets.

Sergio Ruiz-Bonilla: sergio.ruiz-bonilla@durham.ac.uk  
Jacob Kegerreis: jacob.kegerreis@durham.ac.uk

Visit https://github.com/srbonilla/woma to download the code including examples 
and for support.
"""

import numpy as np
import copy
import h5py
from scipy.interpolate import interp1d
from tqdm import tqdm
import seagen
import sys

from woma.spherical_funcs import L1_spherical, L2_spherical, L3_spherical
from woma.spin_funcs import L1_spin, L2_spin, L3_spin
import woma.spin_funcs.utils_spin as us
from woma.misc import glob_vars as gv
from woma.misc import utils, io
from woma.eos import eos
from woma.eos.T_rho import T_rho, set_T_rho_args, T_rho_id_and_args_from_type


# ============================================================================ #
#                       Spherical profile class                                #
# ============================================================================ #


class Planet:
    """ Planet class ...

    Parameters
    ----------
    name : str
        The name of the planet object.

    Fp_planet : str
        The object data file path. Default to `data/<name>.hdf5`.

    A1_mat_layer : [str]
        The name of the material in each layer, from the central layer outwards. 
        See Di_mat_id in `eos/eos.py`.
        
    A1_T_rho_type : [int]
        The type of temperature-density relation in each layer, from the central 
        layer outwards. See Di_mat_id in `eos/eos.py`.

        "power=alpha"   T = K * rho^alpha. K is set internally at the start of 
                        each layer. Set alpha = 0 for isothermal.
        "adiabatic"     Adiabatic. constant entropy is set internally at the 
                        start of each layer.

    A1_R_layer : [float]
        The outer radii of each layer, from the central layer outwards (m).

    A1_M_layer : [float]
        The mass within each layer, starting from the from the central layer 
        outwards (kg).

    M : float
        The total mass (kg).

    P_s, T_s, rho_s : float
        The pressure, temperature, and density at the surface. Only two of the 
        three need be provided (Pa, K, kg m^-3).

    I_MR2 : float
        The moment of inertia factor.

    M_max : float
        The maximum mass allowed (kg).

    R_min, R_max : float
        The minimum and maximum radii to try (m).

    rho_min : float
        The minimum density for the outer edge of the profile (kg m^-3).

    M_frac_tol : float
        The tolerance for finding the appropriate mass, as a fraction of the 
        total mass.

    num_prof : int
        The number of profile integration steps.

    num_attempt, num_attempt_2 : int
        The maximum number of iteration attempts.

    Attributes (in addition to the input parameters)
    ----------
    num_layer : int
        The number of planetary layers.

    P_0, P_1, ... P_s; T_0, ..., T_s; rho_0, ..., rho_s: float
        The pressure, temperature, and density (Pa, K, kg m^-3) at each layer 
        boundary, from the centre (_0) up to the surface (_s).

    A1_mat_id_layer : [int]
        The ID of the material in each layer, from the central layer outwards.

    A1_idx_layer : [int]
        The profile index of each boundary, from the central layer outwards.

    A1_r : [float]
        The profile radii, in increasing order (m).

    A1_P : [float]
        The pressure at each profile radius (Pa).
        
    A1_rho : [float]
        The density at each profile radius (kg m^-3).
        
    A1_T : [float]
        The temperature at each profile radius (K).
        
    A1_u : [float]
        The specific internal energy at each profile radius (J kg^-1).
        
    A1_mat_id : [int]
        The ID of the material at each profile radius.
    """

    def __init__(
        self,
        name=None,
        Fp_planet=None,
        A1_mat_layer=None,
        A1_T_rho_type=None,
        A1_R_layer=None,
        A1_M_layer=None,
        A1_idx_layer=None,
        M=None,
        P_0=None,
        T_0=None,
        rho_0=None,
        P_1=None,
        T_1=None,
        rho_1=None,
        P_2=None,
        T_2=None,
        rho_2=None,
        P_s=None,
        T_s=None,
        rho_s=None,
        I_MR2=None,
        M_max=None,
        R_min=None,
        R_max=None,
        rho_min=None,
        M_frac_tol=0.01,
        num_prof=10000,
        num_attempt=40,
        num_attempt_2=40,
    ):
        self.name = name
        self.Fp_planet = Fp_planet
        self.A1_mat_layer = A1_mat_layer
        self.A1_T_rho_type = A1_T_rho_type
        self.A1_R_layer = A1_R_layer
        self.A1_M_layer = A1_M_layer
        self.A1_idx_layer = A1_idx_layer
        self.M = M
        self.P_0 = P_0
        self.T_0 = T_0
        self.rho_0 = rho_0
        self.P_1 = P_1
        self.T_1 = T_1
        self.rho_1 = rho_1
        self.P_2 = P_2
        self.T_2 = T_2
        self.rho_2 = rho_2
        self.P_s = P_s
        self.T_s = T_s
        self.rho_s = rho_s
        self.I_MR2 = I_MR2
        self.M_max = M_max
        self.R_min = R_min
        self.R_max = R_max
        self.rho_min = rho_min
        self.M_frac_tol = M_frac_tol
        self.num_prof = num_prof
        self.num_attempt = num_attempt
        self.num_attempt_2 = num_attempt_2

        # Derived or default attributes

        # Number of layers
        if self.A1_mat_layer is not None:
            self.num_layer = len(self.A1_mat_layer)
            self.A1_mat_id_layer = [gv.Di_mat_id[mat] for mat in self.A1_mat_layer]
        else:
            # Placeholder
            self.num_layer = 1

        # Two of P, T, and rho must be provided at the surface to calculate the
        # third. If all three are provided then rho is overwritten.
        if self.P_s is not None and self.T_s is not None:
            self.rho_s = eos.rho_P_T(self.P_s, self.T_s, self.A1_mat_id_layer[-1])
        elif self.rho_s is not None and self.T_s is not None:
            self.P_s = eos.P_T_rho(self.T_s, self.rho_s, self.A1_mat_id_layer[-1])

        # Temperature--density relation
        if self.A1_T_rho_type is not None:
            self.A1_T_rho_type_id, self.A1_T_rho_args = T_rho_id_and_args_from_type(
                self.A1_T_rho_type
            )

        # Default filename and layer arrays
        if self.Fp_planet is None:
            self.Fp_planet = "data/%s.hdf5" % self.name  ###is that sensible?
        if self.A1_R_layer is None:
            self.A1_R_layer = [None] * self.num_layer
        if self.A1_M_layer is None:
            self.A1_M_layer = [None] * self.num_layer
        self.R = self.A1_R_layer[-1]

        # Force types for numba
        if self.A1_R_layer is not None:
            self.A1_R_layer = np.array(self.A1_R_layer, dtype="float")
        if self.A1_T_rho_args is not None:
            self.A1_T_rho_args = np.array(self.A1_T_rho_args, dtype="float")

    # ========
    # General
    # ========
    def escape_velocity(self):
        assert self.M is not None
        assert self.A1_R_layer[-1] is not None

        self.v_escape = np.sqrt(2 * gv.G * self.M / self.A1_R_layer[-1])

    def update_attributes(self):
        """ Set all planet information after making the profiles. """
        self.num_prof = len(self.A1_r)

        # Reverse profile arrays to be ordered by increasing radius
        if self.A1_r[-1] < self.A1_r[0]:
            self.A1_r = self.A1_r[::-1]
            self.A1_m_enc = self.A1_m_enc[::-1]
            self.A1_P = self.A1_P[::-1]
            self.A1_T = self.A1_T[::-1]
            self.A1_rho = self.A1_rho[::-1]
            self.A1_u = self.A1_u[::-1]
            self.A1_mat_id = self.A1_mat_id[::-1]

        # Index of the outer edge of each layer
        self.A1_idx_layer = np.append(
            np.where(np.diff(self.A1_mat_id) != 0)[0], self.num_prof - 1
        )

        # Boundary radii
        self.A1_R_layer = self.A1_r[self.A1_idx_layer]
        self.R = self.A1_R_layer[-1]

        # Layer masses
        self.A1_M_layer = self.A1_m_enc[self.A1_idx_layer]
        if self.num_layer > 1:
            self.A1_M_layer[1:] -= self.A1_M_layer[:-1]
        self.M = np.sum(self.A1_M_layer)

        # Moment of inertia factor
        self.I_MR2 = utils.moi(self.A1_r, self.A1_rho) / (self.M * self.R ** 2)

        # P, T, and rho at the centre and the outer boundary of each layer
        self.P_0 = self.A1_P[0]
        self.T_0 = self.A1_T[0]
        self.rho_0 = self.A1_rho[0]
        if self.num_layer > 1:
            self.P_1 = self.A1_P[self.A1_idx_layer[0]]
            self.T_1 = self.A1_T[self.A1_idx_layer[0]]
            self.rho_1 = self.A1_rho[self.A1_idx_layer[0]]
        if self.num_layer > 2:
            self.P_2 = self.A1_P[self.A1_idx_layer[1]]
            self.T_2 = self.A1_T[self.A1_idx_layer[1]]
            self.rho_2 = self.A1_rho[self.A1_idx_layer[1]]
        self.P_s = self.A1_P[-1]
        self.T_s = self.A1_T[-1]
        self.rho_s = self.A1_rho[-1]

        self.escape_velocity()

    def print_info(self):
        """ Print the main properties. """
        # Print and catch if any variables are None
        def print_try(string, variables):
            try:
                print(string % variables)
            except TypeError:
                print("    %s = None" % variables[0])

        space = 12
        print_try('Planet "%s": ', self.name)
        print_try(
            "    %s = %.5g  kg  = %.5g  M_earth",
            (utils.add_whitespace("M", space), self.M, self.M / gv.M_earth),
        )
        print_try(
            "    %s = %.5g  m  = %.5g  R_earth",
            (utils.add_whitespace("R", space), self.R, self.R / gv.R_earth),
        )
        print_try(
            "    %s = %s ",
            (
                utils.add_whitespace("mat", space),
                utils.format_array_string(self.A1_mat_layer, "string"),
            ),
        )
        print_try(
            "    %s = %s ",
            (
                utils.add_whitespace("mat_id", space),
                utils.format_array_string(self.A1_mat_id_layer, "%d"),
            ),
        )
        print_try(
            "    %s = %s ",
            (
                utils.add_whitespace("T_rho_type", space),
                utils.format_array_string(self.A1_T_rho_type, "string"),
            ),
        )
        print_try(
            "    %s = %s  R_earth",
            (
                utils.add_whitespace("R_layer", space),
                utils.format_array_string(self.A1_R_layer / gv.R_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s  M_earth",
            (
                utils.add_whitespace("M_layer", space),
                utils.format_array_string(self.A1_M_layer / gv.M_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s  M_tot",
            (
                utils.add_whitespace("M_frac_layer", space),
                utils.format_array_string(self.A1_M_layer / self.M, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s ",
            (
                utils.add_whitespace("idx_layer", space),
                utils.format_array_string(self.A1_idx_layer, "%d"),
            ),
        )
        print_try("    %s = %.5g  Pa", (utils.add_whitespace("P_s", space), self.P_s))
        print_try("    %s = %.5g  K", (utils.add_whitespace("T_s", space), self.T_s))
        print_try(
            "    %s = %.5g  kg m^-3", (utils.add_whitespace("rho_s", space), self.rho_s)
        )
        if self.num_layer > 2:
            print_try(
                "    %s = %.5g  Pa", (utils.add_whitespace("P_2", space), self.P_2)
            )
            print_try(
                "    %s = %.5g  K", (utils.add_whitespace("T_2", space), self.T_2)
            )
            print_try(
                "    %s = %.5g  kg m^-",
                (utils.add_whitespace("rho_2", space), self.rho_2),
            )
        if self.num_layer > 1:
            print_try(
                "    %s = %.5g  Pa", (utils.add_whitespace("P_1", space), self.P_1)
            )
            print_try(
                "    %s = %.5g  K", (utils.add_whitespace("T_1", space), self.T_1)
            )
            print_try(
                "    %s = %.5g  kg m^-3",
                (utils.add_whitespace("rho_1", space), self.rho_1),
            )
        print_try("    %s = %.5g  Pa", (utils.add_whitespace("P_0", space), self.P_0))
        print_try("    %s = %.5g  K", (utils.add_whitespace("T_0", space), self.T_0))
        print_try(
            "    %s = %.5g  kg m^-3", (utils.add_whitespace("rho_0", space), self.rho_0)
        )
        print_try(
            "    %s = %.5g  M_tot*R_tot^2",
            (utils.add_whitespace("I_MR2", space), self.I_MR2),
        )

    def load_planet_profiles(self, verbosity=1):
        """ Load the profiles arrays for an existing Planet object from a file. """
        Fp_planet = utils.check_end(self.Fp_planet, ".hdf5")

        if verbosity >= 1:
            print('Loading "%s"... ' % Fp_planet[-60:], end="")
            sys.stdout.flush()

        with h5py.File(Fp_planet, "r") as f:
            (
                self.A1_r,
                self.A1_m_enc,
                self.A1_rho,
                self.A1_T,
                self.A1_P,
                self.A1_u,
                self.A1_mat_id,
            ) = io.multi_get_planet_data(
                f, ["r", "m_enc", "rho", "T", "P", "u", "mat_id"]
            )

        if verbosity >= 1:
            print("Done")

    # ========
    # 1 Layer
    # ========
    def _1_layer_input(self):

        assert self.num_layer == 1
        assert self.P_s is not None
        assert self.T_s is not None
        assert self.rho_s is not None
        assert self.A1_mat_id_layer[0] is not None
        assert self.A1_T_rho_type_id[0] is not None

    def gen_prof_L1_find_R_given_M(self, verbosity=1):
        """ Compute the profile of a planet with 1 layer by finding the correct
            radius for a given mass.
        """
        # Check for necessary input
        assert self.R_max is not None
        assert self.M is not None
        self._1_layer_input()

        self.R = L1_spherical.L1_find_radius(
            self.num_prof,
            self.R_max,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.num_attempt,
            verbosity=verbosity,
        )
        self.A1_R_layer[-1] = self.R

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L1_spherical.L1_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
        )

        if verbosity >= 1:
            print("Done!")

        # Integrate the profiles
        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L1_spherical.L1_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
        )

        self.update_attributes()
        if verbosity >= 1:
            self.print_info()

    def gen_prof_L1_find_M_given_R(self, verbosity=1):
        # Check for necessary input
        assert self.R is not None or self.A1_R_layer[0] is not None
        assert self.M_max is not None
        assert len(self.A1_R_layer) == 1
        self._1_layer_input()
        if self.R is None:
            self.R = self.A1_R_layer[0]

        if verbosity >= 1:
            print("Finding M given R...")

        self.M = L1_spherical.L1_find_mass(
            self.num_prof,
            self.R,
            self.M_max,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
        )

        if verbosity >= 1:
            print("Done!")

        # Integrate the profiles
        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L1_spherical.L1_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
        )

        self.update_attributes()
        if verbosity >= 1:
            self.print_info()

    def gen_prof_L1_given_R_M(self, verbosity=1):
        # Check for necessary input
        assert self.R is not None
        assert self.M is not None
        self._1_layer_input()

        # Integrate the profiles
        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L1_spherical.L1_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
        )

        self.update_attributes()
        if verbosity >= 1:
            self.print_info()

    # ========
    # 2 Layers
    # ========
    def _2_layer_input(self):

        assert self.num_layer == 2
        assert self.P_s is not None
        assert self.T_s is not None
        assert self.rho_s is not None
        assert self.A1_mat_id_layer[0] is not None
        assert self.A1_T_rho_type_id[0] is not None
        assert self.A1_mat_id_layer[1] is not None
        assert self.A1_T_rho_type_id[1] is not None

    def gen_prof_L2_find_R1_given_R_M(self, verbosity=1):
        # Check for necessary input
        assert self.R is not None
        assert self.M is not None
        self._2_layer_input()

        self.A1_R_layer[0] = L2_spherical.L2_find_R1(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.num_attempt,
            verbosity=verbosity,
        )

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L2_spherical.L2_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L2_spherical.L2_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        self.update_attributes()

        if verbosity >= 1:
            print("Done!")
            self.print_info()

    def gen_prof_L2_find_M_given_R1_R(self, verbosity=1):
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        assert self.M_max is not None
        self._2_layer_input()

        if verbosity >= 1:
            print("Finding M given R1 and R...")

        self.M = L2_spherical.L2_find_mass(
            self.num_prof,
            self.R,
            self.M_max,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L2_spherical.L2_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        self.update_attributes()

        if verbosity >= 1:
            print("Done!")
            self.print_info()

    def gen_prof_L2_find_R_given_M_R1(self, verbosity=1):
        # Check for necessary input
        assert self.A1_R_layer[0] is not None
        assert self.R_max is not None
        assert self.M is not None
        self._2_layer_input()

        self.R = L2_spherical.L2_find_radius(
            self.num_prof,
            self.R_max,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.num_attempt,
            verbosity=verbosity,
        )
        self.A1_R_layer[-1] = self.R

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L2_spherical.L2_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        if verbosity >= 1:
            print("Done!")

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L2_spherical.L2_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        self.update_attributes()

        if verbosity >= 1:
            self.print_info()

    def gen_prof_L2_find_R1_R_given_M1_M2(self, verbosity=1):

        # Check for necessary input
        self._2_layer_input()
        assert self.R_max is not None
        assert self.A1_M_layer[0] is not None
        assert self.A1_M_layer[1] is not None
        # Check masses
        if self.M is not None:
            assert self.M == self.A1_M_layer[0] + self.A1_M_layer[1]
        else:
            self.M = self.A1_M_layer[0] + self.A1_M_layer[1]

        self.A1_R_layer[0], self.R = L2_spherical.L2_find_R1_R(
            self.num_prof,
            self.R_max,
            self.A1_M_layer[0],
            self.A1_M_layer[1],
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.num_attempt,
            verbosity=verbosity,
        )
        self.A1_R_layer[-1] = self.R

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L2_spherical.L2_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        if verbosity >= 1:
            print("Done!")

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L2_spherical.L2_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        self.update_attributes()

        if verbosity >= 1:
            self.print_info()

    def gen_prof_L2_given_R_M_R1(self, verbosity=1):
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        assert self.M is not None
        self._2_layer_input()

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L2_spherical.L2_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        self.update_attributes()

        if verbosity >= 1:
            self.print_info()

    def gen_prof_L2_given_prof_L1(self, mat, T_rho_type_id, T_rho_args, rho_min, verbosity=1):
        """ Add a second layer (atmosphere) on top of existing 1 layer profiles.
        
        ### Could probably be combined with gen_prof_L3_given_prof_L2()
            for any number of inner layers!

        Parameters
        ----------
        ...

        Set attributes
        --------------
        ...
        """
        # Check for necessary input
        assert self.num_layer == 1
        assert self.A1_R_layer[0] is not None
        assert self.M is not None
        assert self.P_s is not None
        assert self.T_s is not None
        assert self.rho_s is not None
        assert self.A1_mat_id_layer[0] is not None
        assert self.A1_T_rho_type_id[0] is not None

        self.num_layer = 2
        self.A1_mat_layer = np.append(self.A1_mat_layer, mat)
        self.A1_mat_id_layer = [gv.Di_mat_id[mat] for mat in self.A1_mat_layer]
        self.A1_T_rho_type_id = np.append(self.A1_T_rho_type_id, T_rho_type_id)
        A1_T_rho_args_aux = np.zeros((3, 2))
        A1_T_rho_args_aux[0:2] = self.A1_T_rho_args
        A1_T_rho_args_aux[2] = np.array(T_rho_args, dtype="float")
        self.A1_T_rho_args = A1_T_rho_args_aux
        self.rho_min = rho_min

        dr = self.A1_r[1]
        mat_id_L2 = self.A1_mat_id_layer[1]

        # Reverse profile arrays to be ordered by increasing radius
        if self.A1_r[-1] < self.A1_r[0]:
            self.A1_r = self.A1_r[::-1]
            self.A1_m_enc = self.A1_m_enc[::-1]
            self.A1_P = self.A1_P[::-1]
            self.A1_T = self.A1_T[::-1]
            self.A1_rho = self.A1_rho[::-1]
            self.A1_u = self.A1_u[::-1]
            self.A1_mat_id = self.A1_mat_id[::-1]

        # Initialise the new profiles
        A1_r = [self.A1_r[-1]]
        A1_m_enc = [self.A1_m_enc[-1]]
        A1_P = [self.A1_P[-1]]
        A1_T = [self.A1_T[-1]]
        A1_u = [self.A1_u[-1]]
        A1_mat_id = [mat_id_L2]
        A1_rho = [eos.rho_P_T(A1_P[0], A1_T[0], mat_id_L2)]

        # Set any T-rho relation variables
        self.A1_T_rho_args[1] = set_T_rho_args(
            A1_T[0],
            A1_rho[0],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            mat_id_L2,
        )

        # Integrate outwards until the minimum density (or zero pressure)
        while A1_rho[-1] > self.rho_min:
            A1_r.append(A1_r[-1] + dr)
            A1_m_enc.append(
                A1_m_enc[-1] + 4 * np.pi * A1_r[-1] * A1_r[-1] * A1_rho[-1] * dr
            )
            A1_P.append(
                A1_P[-1] - gv.G * A1_m_enc[-1] * A1_rho[-1] / (A1_r[-1] ** 2) * dr
            )
            if A1_P[-1] <= 0:
                # Add dummy values which will be removed along with the -ve P
                A1_rho.append(0)
                A1_T.append(0)
                A1_u.append(0)
                A1_mat_id.append(0)
                break
            # Update the T-rho parameters
            if self.A1_T_rho_type_id[-1] == gv.type_adb and mat_id_L2 == gv.id_HM80_HHe:
                self.A1_T_rho_args[-1] = set_T_rho_args(
                    A1_T[-1],
                    A1_rho[-1],
                    self.A1_T_rho_type_id[-1],
                    self.A1_T_rho_args[-1],
                    mat_id_L2,
                )
            rho = eos.find_rho(
                A1_P[-1],
                mat_id_L2,
                self.A1_T_rho_type_id[-1],
                self.A1_T_rho_args[-1],
                0.9 * A1_rho[-1],
                A1_rho[-1],
            )
            A1_rho.append(rho)
            A1_T.append(
                T_rho(rho, self.A1_T_rho_type_id[-1], self.A1_T_rho_args[-1], mat_id_L2)
            )
            A1_u.append(eos.u_rho_T(rho, A1_T[-1], mat_id_L2))
            A1_mat_id.append(mat_id_L2)

        # Apppend the new layer to the profiles, removing the final too-low
        # density or non-positive pressure step
        self.A1_r = np.append(self.A1_r, A1_r[1:-1])
        self.A1_m_enc = np.append(self.A1_m_enc, A1_m_enc[1:-1])
        self.A1_P = np.append(self.A1_P, A1_P[1:-1])
        self.A1_T = np.append(self.A1_T, A1_T[1:-1])
        self.A1_rho = np.append(self.A1_rho, A1_rho[1:-1])
        self.A1_u = np.append(self.A1_u, A1_u[1:-1])
        self.A1_mat_id = np.append(self.A1_mat_id, A1_mat_id[1:-1])

        self.update_attributes()

        if verbosity >= 1:
            self.print_info()

    def gen_prof_L2_find_R1_given_M1_add_L2(self, verbosity=1):
        """ Generate a 2 layer profile by first finding the inner 1 layer
            profile using the mass of that layer then add the third layer
            (atmosphere) on top.

        Note: the input T_s, P_s, rho_s here are used for the outer boundary
        of layer 1. They will then be overwritten with the final values
        after layer 2 is added.

        Parameters
        ----------
        ...

        Set attributes
        --------------
        ...
        """
        # Check for necessary input
        assert self.num_layer == 2
        assert self.A1_M_layer[0] is not None
        assert self.P_s is not None
        assert self.T_s is not None
        assert self.rho_s is not None
        assert self.A1_mat_id_layer[0] is not None
        assert self.A1_mat_id_layer[1] is not None
        assert self.A1_T_rho_type_id[0] is not None
        assert self.A1_T_rho_type_id[1] is not None

        # Store the layer 2 properties
        mat_L2 = self.A1_mat_layer[1]
        T_rho_type_L2 = self.A1_T_rho_type_id[1]
        T_rho_args_L2 = self.A1_T_rho_args[1]

        # Temporarily set self to be a 1 layer planet
        self.num_layer = 1
        self.A1_M_layer = self.A1_M_layer[:-1]
        self.A1_R_layer = self.A1_R_layer[:-1]
        self.A1_mat_layer = self.A1_mat_layer[:-1]
        self.A1_mat_id_layer = self.A1_mat_id_layer[:-1]
        self.A1_T_rho_type_id = self.A1_T_rho_type_id[:-1]
        self.A1_T_rho_args = self.A1_T_rho_args[:-1]
        self.rho_s = eos.rho_P_T(self.P_s, self.T_s, self.A1_mat_id_layer[-1])

        # Find the radius of the inner 1 layer in isolation
        self.M = np.sum(self.A1_M_layer)

        self.R = L1_spherical.L1_find_radius(
            self.num_prof,
            self.R_max,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.num_attempt,
            verbosity=verbosity,
        )
        self.A1_R_layer[-1] = self.R

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L1_spherical.L1_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
        )

        if verbosity >= 1:
            print("Done!")

        # Integrate the profiles
        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L1_spherical.L1_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
        )

        self.update_attributes()

        # Add the second layer
        if verbosity >= 1:
            print("Adding the second layer on top...")

        self.gen_prof_L2_given_prof_L1(
            mat=mat_L2,
            T_rho_type_id=T_rho_type_L2,
            T_rho_args=T_rho_args_L2,
            rho_min=self.rho_min,
        )

        if verbosity >= 1:
            print("Done!")

    def gen_prof_L2_find_M1_given_R1_add_L2(self, verbosity=1):
        """ Generate a 2 layer profile by first finding the inner 1 layer
            profile using the radius of that layer then add the third layer
            (atmosphere) on top.

            Note: the input T_s, P_s, rho_s here are used for the outer boundary
            of layer 1. They will then be overwritten with the final values
            after layer 2 is added.

        Parameters
        ----------
        ...

        Set attributes
        --------------
        ...
        """
        # Check for necessary input
        assert self.num_layer == 2
        assert self.A1_R_layer[0] is not None
        assert self.P_s is not None
        assert self.T_s is not None
        assert self.rho_s is not None
        assert self.A1_mat_id_layer[0] is not None
        assert self.A1_mat_id_layer[1] is not None
        assert self.A1_T_rho_type_id[0] is not None
        assert self.A1_T_rho_type_id[1] is not None

        # Store the layer 2 properties
        mat_L2 = self.A1_mat_layer[1]
        T_rho_type_L2 = self.A1_T_rho_type_id[1]
        T_rho_args_L2 = self.A1_T_rho_args[1]

        # Temporarily set self to be a 1 layer planet
        self.num_layer = 1
        self.A1_M_layer = self.A1_M_layer[:-1]
        self.A1_R_layer = self.A1_R_layer[:-1]
        self.A1_mat_layer = self.A1_mat_layer[:-1]
        self.A1_mat_id_layer = self.A1_mat_id_layer[:-1]
        self.A1_T_rho_type_id = self.A1_T_rho_type_id[:-1]
        self.A1_T_rho_args = self.A1_T_rho_args[:-1]
        self.rho_s = eos.rho_P_T(self.P_s, self.T_s, self.A1_mat_id_layer[-1])

        # Find the radius of the inner 1 layer in isolation
        self.M = np.sum(self.A1_M_layer)

        if self.R is None or self.R == 0:
            self.R = self.A1_R_layer[0]

        if verbosity >= 1:
            print("Finding M given R...")

        self.M = L1_spherical.L1_find_mass(
            self.num_prof,
            self.R,
            self.M_max,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
        )

        if verbosity >= 1:
            print("Done!")

        # Integrate the profiles
        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L1_spherical.L1_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
        )

        self.update_attributes()

        # Add the second layer
        if verbosity >= 1:
            print("Adding the second layer on top...")

        self.gen_prof_L2_given_prof_L1(
            mat=mat_L2,
            T_rho_type_id=T_rho_type_L2,
            T_rho_args=T_rho_args_L2,
            rho_min=self.rho_min,
        )

        if verbosity >= 1:
            print("Done!")

    # ========
    # 3 Layers
    # ========
    def _3_layer_input(self):

        assert self.num_layer == 3
        assert self.P_s is not None
        assert self.T_s is not None
        assert self.rho_s is not None
        assert self.A1_mat_id_layer[0] is not None
        assert self.A1_T_rho_type_id[0] is not None
        assert self.A1_mat_id_layer[1] is not None
        assert self.A1_T_rho_type_id[1] is not None
        assert self.A1_mat_id_layer[2] is not None
        assert self.A1_T_rho_type_id[2] is not None

    def gen_prof_L3_find_R1_R2_given_R_M_I(self, verbosity=1):  ### WIP
        # Check for necessary input
        assert self.R is not None
        assert self.M is not None
        assert self.I_MR2 is not None
        self._3_layer_input()

        self.A1_R_layer[0], self.A1_R_layer[1] = L3_spherical.L3_find_R1_R2(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.I_MR2,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
            self.num_attempt,
            self.num_attempt_2,
            verbosity=verbosity,
        )

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L3_spherical.L3_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        self.update_attributes()

        if verbosity >= 1:
            print("Done!")
            self.print_info()

    def gen_prof_L3_find_R_R1_R2_given_M_M1_M2(self):  ### WIP
        return None

    def gen_prof_L3_find_R2_given_R_M_R1(self, verbosity=1):
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        assert self.M is not None
        self._3_layer_input()

        self.A1_R_layer[1] = L3_spherical.L3_find_R2(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
            self.num_attempt,            
            verbosity=verbosity,
        )

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L3_spherical.L3_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        self.update_attributes()

        if verbosity >= 1:
            print("Done!")
            self.print_info()

    def gen_prof_L3_find_R1_given_R_M_R2(self, verbosity=1):
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[1] is not None
        assert self.M is not None
        self._3_layer_input()

        self.A1_R_layer[0] = L3_spherical.L3_find_R1(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
            self.num_attempt,
            verbosity=verbosity,
        )

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L3_spherical.L3_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        self.update_attributes()

        if verbosity >= 1:
            print("Done!")
            self.print_info()

    def gen_prof_L3_find_M_given_R_R1_R2(self, verbosity=1):
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        assert self.A1_R_layer[1] is not None
        assert self.M_max is not None
        self._3_layer_input()

        if verbosity >= 1:
            print("Finding M given R1, R2 and R...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof,
            self.R,
            self.M_max,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L3_spherical.L3_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        self.update_attributes()

        if verbosity >= 1:
            print("Done!")
            self.print_info()

    def gen_prof_L3_find_R_given_M_R1_R2(self, verbosity=1):
        # Check for necessary input
        assert self.R_max is not None
        assert self.A1_R_layer[0] is not None
        assert self.A1_R_layer[1] is not None
        assert self.M is not None
        self._3_layer_input()

        self.R = L3_spherical.L3_find_radius(
            self.num_prof,
            self.R_max,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
            self.num_attempt,
            verbosity=verbosity,
        )
        self.A1_R_layer[-1] = self.R

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L3_spherical.L3_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        self.update_attributes()

        if verbosity >= 1:
            print("Done!")
            self.print_info()

    def gen_prof_L3_given_R_M_R1_R2(self, verbosity=1):
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        assert self.A1_R_layer[1] is not None
        assert self.M is not None
        self._3_layer_input()

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L3_spherical.L3_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_R_layer[1],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.A1_mat_id_layer[2],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
        )

        self.update_attributes()

        if verbosity >= 1:
            self.print_info()

    def gen_prof_L3_given_prof_L2(
        self, mat=None, T_rho_type_id=None, T_rho_args=None, rho_min=None, verbosity=1
    ):
        """ Add a third layer (atmosphere) on top of existing 2 layer profiles.

        Parameters
        ----------
        ...

        Set attributes
        --------------
        ...
        """
        # Check for necessary input
        assert self.num_layer == 2
        assert self.A1_R_layer[0] is not None
        assert self.A1_R_layer[1] is not None
        assert self.M is not None
        assert self.P_s is not None
        assert self.T_s is not None
        assert self.rho_s is not None
        assert self.A1_mat_id_layer[0] is not None
        assert self.A1_T_rho_type_id[0] is not None
        assert self.A1_mat_id_layer[1] is not None
        assert self.A1_T_rho_type_id[1] is not None

        self.num_layer = 3
        if mat is not None:  ###else...?
            self.A1_mat_layer = np.append(self.A1_mat_layer, mat)
            self.A1_mat_id_layer = [gv.Di_mat_id[mat] for mat in self.A1_mat_layer]
        if T_rho_type_id is not None:
            self.A1_T_rho_type_id = np.append(self.A1_T_rho_type_id, T_rho_type_id)
        if T_rho_args is not None:
            A1_T_rho_args_aux = np.zeros((3, 2))
            A1_T_rho_args_aux[0:2] = self.A1_T_rho_args
            A1_T_rho_args_aux[2] = np.array(T_rho_args, dtype="float")
            self.A1_T_rho_args = A1_T_rho_args_aux
        if rho_min is not None:
            self.rho_min = rho_min

        dr = self.A1_r[1]
        mat_id_L3 = self.A1_mat_id_layer[2]

        # Reverse profile arrays to be ordered by increasing radius
        if self.A1_r[-1] < self.A1_r[0]:
            self.A1_r = self.A1_r[::-1]
            self.A1_m_enc = self.A1_m_enc[::-1]
            self.A1_P = self.A1_P[::-1]
            self.A1_T = self.A1_T[::-1]
            self.A1_rho = self.A1_rho[::-1]
            self.A1_u = self.A1_u[::-1]
            self.A1_mat_id = self.A1_mat_id[::-1]

        # Initialise the new profiles
        A1_r = [self.A1_r[-1]]
        A1_m_enc = [self.A1_m_enc[-1]]
        A1_P = [self.A1_P[-1]]
        A1_T = [self.A1_T[-1]]
        A1_u = [self.A1_u[-1]]
        A1_mat_id = [mat_id_L3]
        A1_rho = [eos.rho_P_T(A1_P[0], A1_T[0], mat_id_L3)]

        # Set any T-rho relation variables
        self.A1_T_rho_args[2] = set_T_rho_args(
            A1_T[0],
            A1_rho[0],
            self.A1_T_rho_type_id[2],
            self.A1_T_rho_args[2],
            mat_id_L3,
        )

        # Integrate outwards until the minimum density (or zero pressure)
        while A1_rho[-1] > self.rho_min:
            A1_r.append(A1_r[-1] + dr)
            A1_m_enc.append(
                A1_m_enc[-1] + 4 * np.pi * A1_r[-1] * A1_r[-1] * A1_rho[-1] * dr
            )
            A1_P.append(
                A1_P[-1] - gv.G * A1_m_enc[-1] * A1_rho[-1] / (A1_r[-1] ** 2) * dr
            )
            if A1_P[-1] <= 0:
                # Add dummy values which will be removed along with the -ve P
                A1_rho.append(0)
                A1_T.append(0)
                A1_u.append(0)
                A1_mat_id.append(0)
                break
            # Update the T-rho parameters
            if self.A1_T_rho_type_id[2] == gv.type_adb and mat_id_L3 == gv.id_HM80_HHe:
                self.A1_T_rho_args[2] = set_T_rho_args(
                    A1_T[-1],
                    A1_rho[-1],
                    self.A1_T_rho_type_id[2],
                    self.A1_T_rho_args[2],
                    mat_id_L3,
                )
            rho = eos.find_rho(
                A1_P[-1],
                mat_id_L3,
                self.A1_T_rho_type_id[2],
                self.A1_T_rho_args[2],
                0.9 * A1_rho[-1],
                A1_rho[-1],
            )
            A1_rho.append(rho)
            A1_T.append(
                T_rho(rho, self.A1_T_rho_type_id[2], self.A1_T_rho_args[2], mat_id_L3)
            )
            A1_u.append(eos.u_rho_T(rho, A1_T[-1], mat_id_L3))
            A1_mat_id.append(mat_id_L3)

        # Apppend the new layer to the profiles, removing the final too-low
        # density or non-positive pressure step
        self.A1_r = np.append(self.A1_r, A1_r[1:-1])
        self.A1_m_enc = np.append(self.A1_m_enc, A1_m_enc[1:-1])
        self.A1_P = np.append(self.A1_P, A1_P[1:-1])
        self.A1_T = np.append(self.A1_T, A1_T[1:-1])
        self.A1_rho = np.append(self.A1_rho, A1_rho[1:-1])
        self.A1_u = np.append(self.A1_u, A1_u[1:-1])
        self.A1_mat_id = np.append(self.A1_mat_id, A1_mat_id[1:-1])

        self.update_attributes()

        if verbosity >= 1:
            self.print_info()

    def gen_prof_L3_find_R1_R2_given_M1_M2_add_L3(
        self,
        M1=None,
        M2=None,
        R_min=None,
        R_max=None,
        M_frac_tol=None,
        rho_min=None,
        verbosity=1,
    ):
        """ Generate a 3 layer profile by first finding the inner 2 layer
            profile using the masses of each layer then add the third layer
            (atmosphere) on top.

            Note: the input T_s, P_s, rho_s here are used for the outer boundary
            of layer 2. They will then be overwritten with the final values
            after layer 3 is added.

        Parameters
        ----------
        ...

        Set attributes
        --------------
        ...
        """
        # Check for necessary input
        if M1 is not None:
            self.A1_M_layer[0] = M1
        if M2 is not None:
            self.A1_M_layer[1] = M2
        if R_min is not None:
            self.R_min = R_min
        if R_max is not None:
            self.R_max = R_max
        if M_frac_tol is not None:
            self.M_frac_tol = M_frac_tol
        if rho_min is not None:
            self.rho_min = rho_min
        assert self.num_layer == 3
        assert self.A1_M_layer[0] is not None
        assert self.A1_M_layer[1] is not None
        assert self.P_s is not None
        assert self.T_s is not None
        assert self.rho_s is not None
        assert self.A1_mat_id_layer[0] is not None
        assert self.A1_mat_id_layer[1] is not None
        assert self.A1_mat_id_layer[2] is not None
        assert self.A1_T_rho_type_id[0] is not None
        assert self.A1_T_rho_type_id[1] is not None
        assert self.A1_T_rho_type_id[2] is not None

        # Update R_min and R_max without changing the attributes
        R_min = self.R_min
        R_max = self.R_max

        # Store the layer 3 properties
        mat_L3 = self.A1_mat_layer[2]
        T_rho_type_L3 = self.A1_T_rho_type_id[2]
        T_rho_args_L3 = self.A1_T_rho_args[2]

        # Temporarily set self to be 2 layer planet
        self.num_layer = 2
        self.A1_M_layer = self.A1_M_layer[:2]
        self.A1_R_layer = self.A1_R_layer[:2]
        self.A1_mat_layer = self.A1_mat_layer[:2]
        self.A1_mat_id_layer = self.A1_mat_id_layer[:2]
        self.A1_T_rho_type_id = self.A1_T_rho_type_id[:2]
        self.A1_T_rho_args = self.A1_T_rho_args[:2]
        self.rho_s = eos.rho_P_T(self.P_s, self.T_s, self.A1_mat_id_layer[-1])
        ###what if T_s or P_s was derived instead?

        # Find the radii of the inner 2 layers in isolation
        self.M = np.sum(self.A1_M_layer)

        self.A1_R_layer[0], self.R = L2_spherical.L2_find_R1_R(
            self.num_prof,
            self.R_max,
            self.A1_M_layer[0],
            self.A1_M_layer[1],
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
            self.num_attempt,
            verbosity=verbosity,
        )
        self.A1_R_layer[-1] = self.R

        if verbosity >= 1:
            print("Tweaking M to avoid peaks at the center of the planet...")

        self.M = L2_spherical.L2_find_mass(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        (
            self.A1_r,
            self.A1_m_enc,
            self.A1_P,
            self.A1_T,
            self.A1_rho,
            self.A1_u,
            self.A1_mat_id,
        ) = L2_spherical.L2_integrate(
            self.num_prof,
            self.R,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_R_layer[0],
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            self.A1_mat_id_layer[1],
            self.A1_T_rho_type_id[1],
            self.A1_T_rho_args[1],
        )

        self.update_attributes()

        # Add the third layer
        if verbosity >= 1:
            print("Adding the third layer on top...")

        self.gen_prof_L3_given_prof_L2(
            mat=mat_L3,
            T_rho_type_id=T_rho_type_L3,
            T_rho_args=T_rho_args_L3,
            rho_min=self.rho_min,
        )

        if verbosity >= 1:
            print("Done!")


# ============================================================================ #
#                       Spining profile classes                                #
# ============================================================================ #


class SpinPlanet:
    """ Spinning planet class ...

    Parameters
    ----------        
    ...

    Attributes (in addition to the input parameters)
    ----------
    ...
    """

    def __init__(
        self,
        name=None,
        planet=None,
        period=None,
        num_prof=1000,
        R_e_max=None,
        R_p_max=None,
    ):
        self.name = name
        self.num_prof = num_prof
        self.R_e_max = R_e_max
        self.R_p_max = R_p_max
        self.period = period
        self.P_1 = None
        self.P_2 = None
        self.M = None

        if planet is not None:
            self.planet = planet
            self.num_layer = planet.num_layer
            self.A1_mat_layer = planet.A1_mat_layer
            self.A1_R_layer = planet.A1_R_layer
            self.A1_mat_id_layer = planet.A1_mat_id_layer
            self.A1_T_rho_type = planet.A1_T_rho_type
            self.A1_T_rho_type_id = planet.A1_T_rho_type_id
            self.A1_T_rho_args = planet.A1_T_rho_args
            self.A1_r = planet.A1_r
            self.A1_P = planet.A1_P
            self.A1_T = planet.A1_T
            self.A1_rho = planet.A1_rho
            self.M = planet.M

        # Set default R_e_max and R_p_max
        assert self.A1_r is not None
        if self.R_e_max is None:
            self.R_e_max = 1.5 * np.max(self.A1_r)

        if self.R_p_max is None:
            self.R_p_max = 1.2 * np.max(self.A1_r)

        assert self.num_layer in [1, 2, 3]
        assert self.R_e_max is not None
        assert self.R_p_max is not None

        # Initialize A1_rho_equator and A1_rho_pole with the spherical profile
        self.A1_r_equator = np.linspace(0, self.R_e_max, self.num_prof)
        self.A1_r_pole = np.linspace(0, self.R_p_max, self.num_prof)

        spherical_model = interp1d(
            self.A1_r, self.A1_rho, bounds_error=False, fill_value=0
        )

        self.A1_rho_equator = spherical_model(self.A1_r_equator)
        self.A1_rho_pole = spherical_model(self.A1_r_pole)

        # compute pressure and density at points of change of material
        self.P_0 = np.max(self.A1_P)
        self.P_s = np.min(self.A1_P)
        self.rho_0 = np.max(self.A1_rho)
        self.rho_s = np.min(self.A1_rho)

        self.P_1 = None  # pressure between core and mantle
        self.P_2 = None  # pressure between mantle and atmosphere

        if self.num_layer == 2:

            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[0]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[0]])
            P_boundary = 0.5 * (a + b)

            self.P_1 = P_boundary
            self.rho_1 = np.min(self.A1_rho[self.A1_r <= self.A1_R_layer[0]])
            self.T_1 = T_rho(
                self.rho_1,
                self.A1_T_rho_type_id[0],
                self.A1_T_rho_args[0],
                self.A1_mat_id_layer[0],
            )

        elif self.num_layer == 3:

            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[0]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[0]])
            P_boundary_12 = 0.5 * (a + b)

            self.P_1 = P_boundary_12
            self.rho_1 = np.min(self.A1_rho[self.A1_r <= self.A1_R_layer[0]])
            self.T_1 = T_rho(
                self.rho_1,
                self.A1_T_rho_type_id[0],
                self.A1_T_rho_args[0],
                self.A1_mat_id_layer[0],
            )

            a = np.min(self.A1_P[self.A1_r <= self.A1_R_layer[1]])
            b = np.max(self.A1_P[self.A1_r >= self.A1_R_layer[1]])
            P_boundary_23 = 0.5 * (a + b)

            self.P_2 = P_boundary_23
            self.rho_2 = np.min(self.A1_rho[self.A1_r <= self.A1_R_layer[1]])
            self.T_2 = T_rho(
                self.rho_1,
                self.A1_T_rho_type_id[1],
                self.A1_T_rho_args[1],
                self.A1_mat_id_layer[1],
            )

    def _check_input(self):

        if self.num_layer == 1:
            assert self.A1_mat_id_layer[0] is not None
            assert self.A1_T_rho_type_id[0] is not None
        elif self.num_layer == 2:
            assert self.A1_mat_id_layer[0] is not None
            assert self.A1_T_rho_type_id[0] is not None
            assert self.A1_mat_id_layer[1] is not None
            assert self.A1_T_rho_type_id[1] is not None
        elif self.num_layer == 3:
            assert self.A1_mat_id_layer[0] is not None
            assert self.A1_T_rho_type_id[0] is not None
            assert self.A1_mat_id_layer[1] is not None
            assert self.A1_T_rho_type_id[1] is not None
            assert self.A1_mat_id_layer[2] is not None
            assert self.A1_T_rho_type_id[2] is not None

    def update_attributes(self):
        # Compute mass of the planet
        self.M = us.compute_spin_planet_M(
            self.A1_r_equator, self.A1_rho_equator, self.A1_r_pole, self.A1_rho_pole
        )

        # Compute escape velocity
        v_esc_eq, v_esc_p = us.spin_escape_vel(
            self.A1_r_equator,
            self.A1_rho_equator,
            self.A1_r_pole,
            self.A1_rho_pole,
            self.period,
        )

        self.v_escape_pole = v_esc_p
        self.v_escape_equator = v_esc_eq

        # Compute equatorial and polar radius
        self.R_e = np.max(self.A1_r_equator[self.A1_rho_equator > 0.0])
        self.R_p = np.max(self.A1_r_pole[self.A1_rho_pole > 0.0])

        # Mass per layer, Equatorial and polar temperature and pressure
        if self.num_layer == 1:
            self.A1_R_layer_equator = np.array([self.R_e])
            self.A1_R_layer_pole = np.array([self.R_p])
            # Mass
            self.A1_M_layer = np.array([self.M])
            # Pressure and temperature
            self.A1_P_equator = np.zeros_like(self.A1_r_equator)
            self.A1_P_pole = np.zeros_like(self.A1_r_pole)
            self.A1_T_equator = np.zeros_like(self.A1_r_equator)
            self.A1_T_pole = np.zeros_like(self.A1_r_pole)
            for i, rho in enumerate(self.A1_rho_equator):
                if rho >= self.rho_s:
                    self.A1_T_equator[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_equator[i] = eos.P_T_rho(
                        self.A1_T_equator[i], rho, self.A1_mat_id_layer[0]
                    )
            for i, rho in enumerate(self.A1_rho_pole):
                if rho >= self.rho_s:
                    self.A1_T_pole[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_pole[i] = eos.P_T_rho(
                        self.A1_T_pole[i], rho, self.A1_mat_id_layer[0]
                    )
            # Mat_id
            self.A1_mat_id_equator = (
                np.ones(self.A1_r_equator.shape) * self.A1_mat_id_layer[0]
            )
            self.A1_mat_id_pole = (
                np.ones(self.A1_r_pole.shape) * self.A1_mat_id_layer[0]
            )

        elif self.num_layer == 2:
            self.R_1_equator = np.max(
                self.A1_r_equator[self.A1_rho_equator >= self.rho_1]
            )
            self.A1_R_layer_equator = np.array([self.R_1_equator, self.R_e])
            self.R_1_pole = np.max(self.A1_r_pole[self.A1_rho_pole >= self.rho_1])
            self.A1_R_layer_pole = np.array([self.R_1_pole, self.R_p])
            self.A1_mat_id_equator = (
                (self.A1_rho_equator >= self.rho_1) * self.A1_mat_id_layer[0]
                + (self.A1_rho_equator < self.rho_1) * self.A1_mat_id_layer[1]
            )
            self.A1_mat_id_pole = (
                (self.A1_rho_pole >= self.rho_1) * self.A1_mat_id_layer[0]
                + (self.A1_rho_pole < self.rho_1) * self.A1_mat_id_layer[1]
            )

            self.A1_mat_id_equator[self.A1_rho_equator <= 0] = -1
            self.A1_mat_id_pole[self.A1_rho_pole <= 0] = -1

            self.A1_P_equator = np.zeros_like(self.A1_r_equator)
            self.A1_P_pole = np.zeros_like(self.A1_r_pole)
            self.A1_T_equator = np.zeros_like(self.A1_r_equator)
            self.A1_T_pole = np.zeros_like(self.A1_r_pole)

            for i, rho in enumerate(self.A1_rho_equator):
                if rho >= self.rho_1:
                    self.A1_T_equator[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_equator[i] = eos.P_T_rho(
                        self.A1_T_equator[i], rho, self.A1_mat_id_layer[0]
                    )
                elif rho >= self.rho_s:
                    self.A1_T_equator[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[1],
                        self.A1_T_rho_args[1],
                        self.A1_mat_id_layer[1],
                    )
                    self.A1_P_equator[i] = eos.P_T_rho(
                        self.A1_T_equator[i], rho, self.A1_mat_id_layer[1]
                    )
            for i, rho in enumerate(self.A1_rho_pole):
                if rho >= self.rho_1:
                    self.A1_T_pole[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_pole[i] = eos.P_T_rho(
                        self.A1_T_pole[i], rho, self.A1_mat_id_layer[0]
                    )
                elif rho >= self.rho_s:
                    self.A1_T_pole[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[1],
                        self.A1_T_rho_args[1],
                        self.A1_mat_id_layer[1],
                    )
                    self.A1_P_pole[i] = eos.P_T_rho(
                        self.A1_T_pole[i], rho, self.A1_mat_id_layer[1]
                    )

            r_temp = np.copy(self.A1_r_equator)
            z_temp = np.copy(self.A1_r_pole)
            rho_r_temp = np.copy(self.A1_rho_equator)
            rho_z_temp = np.copy(self.A1_rho_pole)
            rho_r_temp[rho_r_temp < self.rho_1] = 0.0
            rho_z_temp[rho_z_temp < self.rho_1] = 0.0
            M1 = us.compute_spin_planet_M(r_temp, rho_r_temp, z_temp, rho_z_temp)

            M2 = self.M - M1

            self.A1_M_layer = np.array([M1, M2])

        elif self.num_layer == 3:
            self.R_1_equator = np.max(
                self.A1_r_equator[self.A1_rho_equator >= self.rho_1]
            )
            self.R_2_equator = np.max(
                self.A1_r_equator[self.A1_rho_equator >= self.rho_2]
            )
            self.A1_R_layer_equator = np.array(
                [self.R_1_equator, self.R_2_equator, self.R_e]
            )
            self.R_1_pole = np.max(self.A1_r_pole[self.A1_rho_pole >= self.rho_1])
            self.R_2_pole = np.max(self.A1_r_pole[self.A1_rho_pole >= self.rho_2])
            self.A1_R_layer_pole = np.array([self.R_1_pole, self.R_2_pole, self.R_p])
            self.A1_mat_id_equator = (
                (self.A1_rho_equator >= self.rho_1) * self.A1_mat_id_layer[0]
                + np.logical_and(
                    self.A1_rho_equator < self.rho_1, self.A1_rho_equator >= self.rho_2
                )
                * self.A1_mat_id_layer[1]
                + (self.A1_rho_equator < self.rho_2) * self.A1_mat_id_layer[2]
            )
            self.A1_mat_id_equator = (
                (self.A1_rho_pole >= self.rho_1) * self.A1_mat_id_layer[0]
                + np.logical_and(
                    self.A1_rho_pole < self.rho_1, self.A1_rho_pole >= self.rho_2
                )
                * self.A1_mat_id_layer[1]
                + (self.A1_rho_pole < self.rho_2) * self.A1_mat_id_layer[2]
            )

            self.A1_mat_id_equator[self.A1_rho_equator <= 0] = -1
            self.A1_mat_id_pole[self.A1_rho_pole <= 0] = -1

            self.A1_P_equator = np.zeros_like(self.A1_r_equator)
            self.A1_P_pole = np.zeros_like(self.A1_r_pole)
            self.A1_T_equator = np.zeros_like(self.A1_r_equator)
            self.A1_T_pole = np.zeros_like(self.A1_r_pole)

            for i, rho in enumerate(self.A1_rho_equator):
                if rho >= self.rho_1:
                    self.A1_T_equator[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_equator[i] = eos.P_T_rho(
                        self.A1_T_equator[i], rho, self.A1_mat_id_layer[0]
                    )
                elif rho >= self.rho_2:
                    self.A1_T_equator[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[1],
                        self.A1_T_rho_args[1],
                        self.A1_mat_id_layer[1],
                    )
                    self.A1_P_equator[i] = eos.P_T_rho(
                        self.A1_T_equator[i], rho, self.A1_mat_id_layer[1]
                    )
                elif rho >= self.rho_s:
                    self.A1_T_equator[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[2],
                        self.A1_T_rho_args[2],
                        self.A1_mat_id_layer[2],
                    )
                    self.A1_P_equator[i] = eos.P_T_rho(
                        self.A1_T_equator[i], rho, self.A1_mat_id_layer[2]
                    )
            for i, rho in enumerate(self.A1_rho_pole):
                if rho >= self.rho_1:
                    self.A1_T_pole[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_pole[i] = eos.P_T_rho(
                        self.A1_T_pole[i], rho, self.A1_mat_id_layer[0]
                    )
                elif rho >= self.rho_2:
                    self.A1_T_pole[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[1],
                        self.A1_T_rho_args[1],
                        self.A1_mat_id_layer[1],
                    )
                    self.A1_P_pole[i] = eos.P_T_rho(
                        self.A1_T_pole[i], rho, self.A1_mat_id_layer[1]
                    )
                elif rho >= self.rho_s:
                    self.A1_T_pole[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[2],
                        self.A1_T_rho_args[2],
                        self.A1_mat_id_layer[2],
                    )
                    self.A1_P_pole[i] = eos.P_T_rho(
                        self.A1_T_pole[i], rho, self.A1_mat_id_layer[2]
                    )

            r_temp = np.copy(self.A1_r_equator)
            z_temp = np.copy(self.A1_r_pole)
            rho_r_temp = np.copy(self.A1_rho_equator)
            rho_z_temp = np.copy(self.A1_rho_pole)
            rho_r_temp[rho_r_temp < self.rho_1] = 0.0
            rho_z_temp[rho_z_temp < self.rho_1] = 0.0
            M1 = us.compute_spin_planet_M(r_temp, rho_r_temp, z_temp, rho_z_temp)

            rho_r_temp = np.copy(self.A1_rho_equator)
            rho_z_temp = np.copy(self.A1_rho_pole)
            rho_r_temp[rho_r_temp < self.rho_2] = 0.0
            rho_z_temp[rho_z_temp < self.rho_2] = 0.0
            M2 = us.compute_spin_planet_M(r_temp, rho_r_temp, z_temp, rho_z_temp)
            M2 = M2 - M1

            M3 = self.M - M2 - M1

            self.A1_M_layer = np.array([M1, M2, M3])

        self.T_0 = self.A1_T_equator[0]
        self.T_s = self.A1_T_equator[self.A1_T_equator > 0][-1]

    def print_info(self):
        """ Print the main properties. """
        # Print and catch if any variables are None
        def print_try(string, variables):
            try:
                print(string % variables)
            except TypeError:
                print("    %s = None" % variables[0])

        space = 12
        print_try('Planet "%s": ', self.name)
        print_try(
            "    %s = %.5g  kg  = %.5g  M_earth",
            (utils.add_whitespace("M", space), self.M, self.M / gv.M_earth),
        )
        print_try(
            "    %s = %.5g  m  = %.5g  R_earth",
            (utils.add_whitespace("R_equator", space), self.R_e, self.R_e / gv.R_earth),
        )
        print_try(
            "    %s = %.5g  m  = %.5g  R_earth",
            (utils.add_whitespace("R_pole", space), self.R_p, self.R_p / gv.R_earth),
        )
        print_try(
            "    %s = %s ",
            (
                utils.add_whitespace("mat", space),
                utils.format_array_string(self.A1_mat_layer, "string"),
            ),
        )
        print_try(
            "    %s = %s ",
            (
                utils.add_whitespace("mat_id", space),
                utils.format_array_string(self.A1_mat_id_layer, "%d"),
            ),
        )
        print_try(
            "    %s = %s  R_earth",
            (
                utils.add_whitespace("R_layer_eq", space),
                utils.format_array_string(self.A1_R_layer_equator / gv.R_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s  R_earth",
            (
                utils.add_whitespace("R_layer_pole", space),
                utils.format_array_string(self.A1_R_layer_pole / gv.R_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s  M_earth",
            (
                utils.add_whitespace("M_layer", space),
                utils.format_array_string(self.A1_M_layer / gv.M_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s  M_tot",
            (
                utils.add_whitespace("M_frac_layer", space),
                utils.format_array_string(self.A1_M_layer / self.M, "%.5g"),
            ),
        )
        print_try("    %s = %.5g  Pa", (utils.add_whitespace("P_s", space), self.P_s))
        print_try("    %s = %.5g  K", (utils.add_whitespace("T_s", space), self.T_s))
        print_try(
            "    %s = %.5g  kg m^-3", (utils.add_whitespace("rho_s", space), self.rho_s)
        )
        if self.num_layer > 2:
            print_try(
                "    %s = %.5g  Pa", (utils.add_whitespace("P_2", space), self.P_2)
            )
            print_try(
                "    %s = %.5g  K", (utils.add_whitespace("T_2", space), self.T_2)
            )
            print_try(
                "    %s = %.5g  kg m^-3",
                (utils.add_whitespace("rho_2", space), self.rho_2),
            )
        if self.num_layer > 1:
            print_try(
                "    %s = %.5g  Pa", (utils.add_whitespace("P_1", space), self.P_1)
            )
            print_try(
                "    %s = %.5g  K", (utils.add_whitespace("T_1", space), self.T_1)
            )
            print_try(
                "    %s = %.5g  kg m^-3",
                (utils.add_whitespace("rho_1", space), self.rho_1),
            )
        print_try("    %s = %.5g  Pa", (utils.add_whitespace("P_0", space), self.P_0))
        print_try("    %s = %.5g  K", (utils.add_whitespace("T_0", space), self.T_0))
        print_try(
            "    %s = %.5g  kg m^-3", (utils.add_whitespace("rho_0", space), self.rho_0)
        )

    def find_min_period(self, max_period=10, max_iter=20, verbosity=1):

        min_period = us.find_min_period(
            self.num_layer,
            self.A1_r_equator,
            self.A1_rho_equator,
            self.A1_r_pole,
            self.A1_rho_pole,
            self.P_0,
            self.P_s,
            self.rho_0,
            self.rho_s,
            self.A1_mat_id_layer,
            self.A1_T_rho_type_id,
            self.A1_T_rho_args,
            self.P_1,
            self.P_2,
            max_period,
            max_iter,
            verbosity=verbosity,
        )

        self.min_period = min_period

    def spin(
        self,
        max_iter_1=12,
        max_iter_2=20,
        tol=0.001,
        check_min_period=True,
        verbosity=1,
    ):
        # Check for necessary input
        self._check_input()

        for i in tqdm(
            range(max_iter_1), desc="Computing spinning profile", disable=verbosity == 0
        ):
            # compute min_period
            if check_min_period:
                self.find_min_period(max_iter=max_iter_2, verbosity=0)

                # select period for this iteration
                if self.period >= self.min_period:
                    period_iter = self.period
                else:
                    period_iter = self.min_period
            else:
                period_iter = self.period

            # compute profile
            A1_rho_equator, A1_rho_pole = us.spin_iteration(
                period_iter,
                self.num_layer,
                self.A1_r_equator,
                self.A1_rho_equator,
                self.A1_r_pole,
                self.A1_rho_pole,
                self.P_0,
                self.P_s,
                self.rho_0,
                self.rho_s,
                self.A1_mat_id_layer,
                self.A1_T_rho_type_id,
                self.A1_T_rho_args,
                self.P_1,
                self.P_2,
            )

            # convergence criterion
            criterion = np.mean(
                np.abs(A1_rho_equator - self.A1_rho_equator) / self.rho_s
            )

            # save results
            self.A1_rho_equator = A1_rho_equator
            self.A1_rho_pole = A1_rho_pole

            # check if there is convergence
            if criterion < tol:
                if verbosity >= 1:
                    print("Convergence criterion reached.")
                break

        if self.period < period_iter:
            if verbosity >= 1:
                print("")
                print("Minimum period found at", period_iter, "h")
            self.period = period_iter

        self.update_attributes()

        if verbosity >= 1:
            self.print_info()


def _L1_spin_planet_fix_M(
    planet,
    period,
    num_prof=1000,
    R_e_max=None,
    R_p_max=None,
    check_min_period=False,
    max_iter_1=20,
    tol=0.001,
    verbosity=1,
):
    """ Create a spinning planet from a spherical one, keeping the same layer masses.
    
    For a 1 layer planet.

    Parameters
    ----------
    planet : woma.Planet
        The spherical planet object. Must have 1 layer.

    period : float
        Period (h).
        
    num_prof : int
        Number of grid points used in the 1D equatorial and polar profiles.

    R_e_max : float
        Maximum equatorial radius (m). Defaults to 4 times the spherical radius.
        
    R_p_max : float
        Maximum polar radius (m). Defaults to 2 times the spherical radius.
        
    check_min_period : bool
        Checks if period provided is lees than the minimum physically allowed.
        Use True only for extreme high spin.
        
    max_iter_1: int
        Maximum number of iterations allowed.
        
    tol : int
        Tolerance level. The iterative search will end when the fractional 
        difference between the mass of the spinning planet and the spherical one
        is less than tol.

    Returns
    -------
    spin_planet : woma.SpinPlanet
        The spinning planet object.
    """

    assert isinstance(planet, Planet)
    assert planet.num_layer == 1

    # Default max radii
    if R_e_max is None:
        R_e_max = 2 * planet.R
    if R_p_max is None:
        R_p_max = 1.2 * planet.R

    f_min = 0.0
    f_max = 1.0

    for i in tqdm(
        range(max_iter_1), desc="Computing spinning profile", disable=verbosity == 0
    ):

        f = np.mean([f_min, f_max])

        # create copy of planet
        new_planet = copy.deepcopy(planet)

        # shrink it
        new_planet.A1_R_layer = f * new_planet.A1_R_layer
        new_planet.R = f * new_planet.R

        # make new profile
        new_planet.M_max = new_planet.M
        new_planet.gen_prof_L1_find_M_given_R(verbosity=0)

        spin_planet = SpinPlanet(
            planet=new_planet, period=period, R_e_max=R_e_max, R_p_max=R_p_max
        )

        spin_planet.spin(check_min_period=check_min_period, verbosity=0)

        criterion = np.abs(planet.M - spin_planet.M) / planet.M < tol

        if criterion:
            if verbosity >= 1:
                print("Tolerance level criteria reached.")
                spin_planet.print_info()
            return spin_planet

        if spin_planet.M > planet.M:
            f_max = f
        else:
            f_min = f

    if verbosity >= 1:
        spin_planet.print_info()

    return spin_planet


def _L2_spin_planet_fix_M(
    planet,
    period,
    num_prof=1000,
    R_e_max=None,
    R_p_max=None,
    check_min_period=False,
    max_iter_1=20,
    max_iter_2=5,
    tol=0.01,
    verbosity=1,
):
    """ Create a spinning planet from a spherical one, keeping the same layer masses.
    
    For a 2 layer planet.

    Parameters
    ----------
    planet : woma.Planet
        The spherical planet object. Must have 2 layers.

    period : float
        Period (h).
        
    num_prof : int
        Number of grid points used in the 1D equatorial and polar profiles.

    R_e_max : float
        Maximum equatorial radius (m). Defaults to 2 times the spherical radius.
        
    R_p_max : float
        Maximum polar radius (m). Defaults to 1.2 times the spherical radius.
        
    check_min_period : bool
        Checks if period provided is lees than the minimum physically allowed.
        Use True only for extreme high spin.
        
    max_iter_1: int
        Maximum number of iterations allowed. Inner loop.
        
    max_iter_2: int
        Maximum number of iterations allowed. Outer loop.
        
    tol : int
        Tolerance level. The iterative search will end when the fractional 
        differences between the layer masses of the spinning planet and the 
        spherical one are less than tol.

    Returns
    -------
    spin_planet : woma.SpinPlanet
        The spinning planet object.
    """

    assert isinstance(planet, Planet)
    assert planet.num_layer == 2

    # Default max radii
    if R_e_max is None:
        R_e_max = 2 * planet.R
    if R_p_max is None:
        R_p_max = 1.2 * planet.R

    M = planet.M

    f_M_core = planet.A1_M_layer[0] / M

    new_planet = copy.deepcopy(planet)

    spin_planet = SpinPlanet(
        planet=new_planet,
        period=period,
        num_prof=num_prof,
        R_e_max=R_e_max,
        R_p_max=R_p_max,
    )

    spin_planet.spin(check_min_period=check_min_period, verbosity=0)

    for k in tqdm(
        range(max_iter_2), desc="Computing spinning profile", disable=verbosity == 0
    ):

        if spin_planet.M > M:
            R_mantle_min = new_planet.A1_R_layer[0]
            R_mantle_max = new_planet.A1_R_layer[1]
        else:
            R_mantle_min = new_planet.A1_R_layer[1]
            R_mantle_max = 1.1 * new_planet.A1_R_layer[1]

        for i in tqdm(range(max_iter_1), desc="Adjusting outer edge", disable=True):

            # R_core   = np.mean([R_core_min, R_core_max])
            R_mantle = np.mean([R_mantle_min, R_mantle_max])

            # create copy of planet
            new_planet = copy.deepcopy(planet)

            # modify boundaries
            new_planet.A1_R_layer[1] = R_mantle
            new_planet.R = R_mantle

            # make new profile
            new_planet.M_max = 1.2 * planet.M
            new_planet.gen_prof_L2_find_M_given_R1_R(verbosity=0)

            spin_planet = SpinPlanet(
                planet=new_planet,
                period=period,
                num_prof=num_prof,
                R_e_max=R_e_max,
                R_p_max=R_p_max,
            )

            spin_planet.spin(check_min_period=check_min_period, verbosity=0)

            criterion_1 = np.abs(planet.M - spin_planet.M) / planet.M < tol
            criterion_2 = (
                np.abs(
                    planet.A1_M_layer[0] / planet.M
                    - spin_planet.A1_M_layer[0] / spin_planet.M
                )
                < tol
            )

            if criterion_1 and criterion_2:
                if verbosity >= 1:
                    print("Tolerance level criteria reached.")
                    spin_planet.print_info()
                return spin_planet

            if criterion_1:
                break

            if spin_planet.M > planet.M:
                R_mantle_max = R_mantle
            else:
                R_mantle_min = R_mantle

        if spin_planet.A1_M_layer[0] / spin_planet.M > f_M_core:
            R_core_min = 0
            R_core_max = new_planet.A1_R_layer[0]
        else:
            R_core_min = new_planet.A1_R_layer[0]
            R_core_max = new_planet.A1_R_layer[1]

        for i in tqdm(
            range(max_iter_1), desc="Adjusting core-mantle boundary", disable=True
        ):

            R_core = np.mean([R_core_min, R_core_max])

            # create copy of planet
            new_planet = copy.deepcopy(planet)

            # modify boundaries
            new_planet.A1_R_layer[0] = R_core
            new_planet.A1_R_layer[1] = R_mantle
            new_planet.R = R_mantle

            # make new profile
            new_planet.M_max = 1.2 * planet.M
            new_planet.gen_prof_L2_find_M_given_R1_R(verbosity=0)

            spin_planet = SpinPlanet(
                planet=new_planet,
                period=period,
                num_prof=num_prof,
                R_e_max=R_e_max,
                R_p_max=R_p_max,
            )

            spin_planet.spin(verbosity=0, check_min_period=check_min_period)

            criterion_1 = np.abs(planet.M - spin_planet.M) / planet.M < tol
            criterion_2 = (
                np.abs(
                    planet.A1_M_layer[0] / planet.M
                    - spin_planet.A1_M_layer[0] / spin_planet.M
                )
                < tol
            )

            if criterion_1 and criterion_2:
                if verbosity >= 1:
                    print("Tolerance level criteria reached.")
                    spin_planet.print_info()
                return spin_planet

            if criterion_2:
                break

            new_f_M_core = spin_planet.A1_M_layer[0] / spin_planet.M

            if new_f_M_core > f_M_core:
                R_core_max = R_core
            else:
                R_core_min = R_core

    if verbosity >= 1:
        spin_planet.print_info()

    return spin_planet


def spin_planet_fix_M(
    planet,
    period,
    num_prof=1000,
    R_e_max=None,
    R_p_max=None,
    check_min_period=False,
    max_iter_1=8,
    max_iter_2=8,
    tol=0.01,
):
    """ Create a spinning planet from a spherical one, keeping the same layer masses.

    Parameters
    ----------
    planet : woma.Planet
        The spherical planet object.

    period : float
        Period (h).
        
    num_prof : int
        Number of grid points used in the 1D equatorial and polar profiles.

    R_e_max : float
        Maximum equatorial radius (m). Defaults to 2 times the spherical radius.
        
    R_p_max : float
        Maximum polar radius (m). Defaults to 1.2 times the spherical radius.
        
    check_min_period : bool
        Checks if period provided is lees than the minimum physically allowed.
        Use True only for extreme high spin.
        
    max_iter_1: int
        Maximum number of iterations allowed. Inner loop.
        
    max_iter_2: int
        Maximum number of iterations allowed. Outer loop.
        
    tol : int
        Tolerance level. The iterative search will end when the fractional 
        differences between the layer masses of the spinning planet and the 
        spherical one are less than tol.

    Returns
    -------
    spin_planet : woma.SpinPlanet
        The spinning planet object.
    """

    # Default max radii
    if R_e_max is None:
        R_e_max = 2 * planet.R
    if R_p_max is None:
        R_p_max = 1.2 * planet.R

    if planet.num_layer == 1:

        spin_planet = _L1_spin_planet_fix_M(
            planet,
            period,
            num_prof,
            R_e_max,
            R_p_max,
            check_min_period,
            max_iter_1,
            tol,
        )

    elif planet.num_layer == 2:

        spin_planet = _L2_spin_planet_fix_M(
            planet,
            period,
            num_prof,
            R_e_max,
            R_p_max,
            check_min_period,
            max_iter_1,
            max_iter_2,
            tol,
        )

    elif planet.num_layer == 3:

        raise ValueError("3 layers not implemented yet")

    else:

        raise ValueError("planet.num_layer must be 1, 2, or 3")

    return spin_planet


class ParticleSet:
    """ Arrange particles to precisely match a spinning or spherical planetary profile.

    Parameters
    ----------
    planet : woma.Planet or woma.SpinPlanet
        The planet profile object.

    N_particles : int
        The number of particles to place.
        
    N_ngb : int
        The number of neighbours used to estimate the SPH smoothing lengths and 
        densities.

    verbosity : int
        The verbosity to control printed output:
        0       None
        1       Standard (default)
        2       Extra

    Attributes (in addition to the input parameters)
    ----------
    A2_pos : [[float]]
        Array of [x, y, z] positions for all particles (m).
    
    A2_vel : [[float]]
        Array of [vx, vy, vz] velocities for all particles (m).
    
    A1_x : [float]
        Array of x positions for all particles (m).
        
    A1_y : [float]
        Array of y positions for all particles (m).
        
    A1_z : [float]
        Array of z positions for all particles (m).
        
    A1_vx : [float]
        Array of x velocities for all particles (m s^-1).
        
    A1_vy : [float]
        Array of y velocities for all particles (m s^-1).
        
    A1_vz : [float]
        Array of z velocities for all particles (m s^-1).
        
    A1_m : [float]
        Array of masses for all particles (kg).
        
    A1_rho : [float]
        Array of densities for all particles (kg m^-3).
        
    A1_u : [float]
        Array of specific internal energies for all particles (J kg^-1).
        
    A1_T : [float]
        Array of temperatures for all particles (K).
        
    A1_P : [float]
        Array of pressures for all particles (Pa).
    
    A1_h : [float]
        Array of smoothing lengths for all particles (m).
        
    A1_mat_id : [int]
        Array of material ids for all particles. See glob_vars.py
        
    A1_id : [int]
        Array of ids for all particles.        
    """

    def __init__(self, planet=None, N_particles=None, N_ngb=48, verbosity=1):
        self.N_particles = N_particles
        self.N_ngb = N_ngb

        assert isinstance(planet, Planet) or isinstance(planet, SpinPlanet)
        assert self.N_particles is not None

        if isinstance(planet, Planet):
            particles = seagen.GenSphere(
                self.N_particles,
                planet.A1_r[1:],
                planet.A1_rho[1:],
                planet.A1_mat_id[1:],
                planet.A1_u[1:],
                planet.A1_T[1:],
                planet.A1_P[1:],
                verbosity=verbosity,
            )

            self.A1_x = particles.A1_x
            self.A1_y = particles.A1_y
            self.A1_z = particles.A1_z
            self.A1_vx = np.zeros_like(particles.A1_x)
            self.A1_vy = np.zeros_like(particles.A1_x)
            self.A1_vz = np.zeros_like(particles.A1_x)
            self.A1_m = particles.A1_m
            self.A1_rho = particles.A1_rho
            self.A1_u = particles.A1_u
            self.A1_T = particles.A1_T
            self.A1_P = particles.A1_P
            self.A1_mat_id = particles.A1_mat
            self.A1_id = np.arange(self.A1_m.shape[0])

            # Smoothing lengths, crudely estimated from the densities
            w_edge = 2  # r/h at which the kernel goes to zero
            self.A1_h = (
                np.cbrt(self.N_ngb * self.A1_m / (4 / 3 * np.pi * self.A1_rho)) / w_edge
            )

            self.N_particles = particles.A1_x.shape[0]

        if isinstance(planet, SpinPlanet):
            if self.num_layer == 1:

                (
                    self.A1_x,
                    self.A1_y,
                    self.A1_z,
                    self.A1_vx,
                    self.A1_vy,
                    self.A1_vz,
                    self.A1_m,
                    self.A1_rho,
                    self.A1_u,
                    self.A1_P,
                    self.A1_h,
                    self.A1_mat_id,
                    self.A1_id,
                ) = L1_spin.picle_placement_L1(
                    planet.A1_r_equator,
                    planet.A1_rho_equator,
                    planet.A1_r_pole,
                    planet.A1_rho_pole,
                    planet.period,
                    self.N_particles,
                    planet.A1_mat_id_layer[0],
                    planet.A1_T_rho_type_id[0],
                    planet.A1_T_rho_args[0],
                    self.N_ngb,
                    verbosity=verbosity,
                )

                self.N_particles = self.A1_x.shape[0]

            elif self.num_layer == 2:

                rho_P_model = interp1d(planet.A1_P, planet.A1_rho)
                rho_1 = rho_P_model(planet.P_1)

                (
                    self.A1_x,
                    self.A1_y,
                    self.A1_z,
                    self.A1_vx,
                    self.A1_vy,
                    self.A1_vz,
                    self.A1_m,
                    self.A1_rho,
                    self.A1_u,
                    self.A1_P,
                    self.A1_h,
                    self.A1_mat_id,
                    self.A1_id,
                ) = L2_spin.picle_placement_L2(
                    planet.A1_r_equator,
                    planet.A1_rho_equator,
                    planet.A1_r_pole,
                    planet.A1_rho_pole,
                    planet.period,
                    self.N_particles,
                    rho_1,
                    planet.A1_mat_id_layer[0],
                    planet.A1_T_rho_type_id[0],
                    planet.A1_T_rho_args[0],
                    planet.A1_mat_id_layer[1],
                    planet.A1_T_rho_type_id[1],
                    planet.A1_T_rho_args[1],
                    self.N_ngb,
                    verbosity=verbosity,
                )

                self.N_particles = self.A1_x.shape[0]

            elif self.num_layer == 3:

                rho_P_model = interp1d(planet.A1_P, planet.A1_rho)
                rho_1 = rho_P_model(planet.P_1)
                rho_2 = rho_P_model(planet.P_2)

                (
                    self.A1_x,
                    self.A1_y,
                    self.A1_z,
                    self.A1_vx,
                    self.A1_vy,
                    self.A1_vz,
                    self.A1_m,
                    self.A1_rho,
                    self.A1_u,
                    self.A1_P,
                    self.A1_h,
                    self.A1_mat_id,
                    self.A1_id,
                ) = L3_spin.picle_placement_L3(
                    planet.A1_r_equator,
                    planet.A1_rho_equator,
                    planet.A1_r_pole,
                    planet.A1_rho_pole,
                    planet.period,
                    planet.N_particles,
                    rho_1,
                    rho_2,
                    planet.A1_mat_id_layer[0],
                    planet.A1_T_rho_type_id[0],
                    planet.A1_T_rho_args[0],
                    planet.A1_mat_id_layer[1],
                    planet.A1_T_rho_type_id[1],
                    planet.A1_T_rho_args[1],
                    planet.A1_mat_id_layer[2],
                    planet.A1_T_rho_type_id[2],
                    planet.A1_T_rho_args[2],
                    self.N_ngb,
                    verbosity=verbosity,
                )

                self.N_particles = self.A1_x.shape[0]

        # 2D position and velocity arrays
        self.A2_pos = np.transpose([self.A1_x, self.A1_y, self.A1_z])
        self.A2_vel = np.transpose([self.A1_vx, self.A1_vy, self.A1_vz])
