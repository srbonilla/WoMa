"""
WoMa (World Maker)
====

Create models of rotating (and non-rotating) planets (or stars etc.) by solving 
the differential equations for hydrostatic equilibrium, and/or create initial 
conditions for smoothed particle hydrodynamics (SPH) or any other particle-based 
methods by placing particles to precisely match the planet's profiles.

See README.md and tutorial.ipynb for general documentation and examples.

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


class Planet:
    """ Planet class ...

    Parameters
    ----------
    name : str
        The name of the planet object.

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

    P_s, T_s, rho_s : float
        The pressure, temperature, and density at the surface. Only two of the 
        three need be provided (Pa, K, kg m^-3).

    M : float
        The total mass (kg).

    R : float
        The total radius (m).

    A1_M_layer : [float]
        The mass within each layer, starting from the from the central layer 
        outwards (kg).

    A1_R_layer : [float]
        The outer radii of each layer, from the central layer outwards (m).

    I_MR2 : float
        The moment of inertia factor.

    num_prof : int
        The number of profile integration steps.
        

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
        A1_mat_layer=None,
        A1_T_rho_type=None,
        P_s=None,
        T_s=None,
        rho_s=None,
        M=None,
        R=None,
        A1_M_layer=None,
        A1_R_layer=None,
        A1_idx_layer=None,
        P_0=None,
        T_0=None,
        rho_0=None,
        P_1=None,
        T_1=None,
        rho_1=None,
        P_2=None,
        T_2=None,
        rho_2=None,
        I_MR2=None,
        num_prof=10000,
    ):
        self.name = name
        self.A1_mat_layer = A1_mat_layer
        self.A1_T_rho_type = A1_T_rho_type
        self.P_s = P_s
        self.T_s = T_s
        self.rho_s = rho_s
        self.M = M
        self.R = R
        self.A1_M_layer = A1_M_layer
        self.A1_R_layer = A1_R_layer
        self.A1_idx_layer = A1_idx_layer
        self.P_0 = P_0
        self.T_0 = T_0
        self.rho_0 = rho_0
        self.P_1 = P_1
        self.T_1 = T_1
        self.rho_1 = rho_1
        self.P_2 = P_2
        self.T_2 = T_2
        self.rho_2 = rho_2
        self.I_MR2 = I_MR2
        self.num_prof = num_prof

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
        if self.A1_M_layer is None:
            self.A1_M_layer = [None] * self.num_layer
        if self.A1_R_layer is None:
            self.A1_R_layer = [None] * self.num_layer
        if self.M is None:
            self.M = self.A1_M_layer[-1]
        if self.R is None:
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

        self.v_esc = np.sqrt(2 * gv.G * self.M / self.A1_R_layer[-1])

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

    def load_planet_profiles(self, Fp_planet, verbosity=1):
        """ Load the profiles arrays for an existing Planet object from a file. 

        Parameters
        ----------
        Fp_planet : str
            The object data file path.
        """
        Fp_planet = utils.check_end(Fp_planet, ".hdf5")

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

    def gen_prof_L1_find_R_given_M(self, R_max, tol=0.001, num_attempt=40, verbosity=1):
        """ 
        Compute the profile of a planet with 1 layer to find the radius given 
        the mass.
            
        Parameters
        ----------
        self.M : float
            The total mass (kg).

        R_max : float
            The maximum radius to try (m).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """
        # Check for necessary input
        assert self.M is not None
        self._1_layer_input()

        self.R = L1_spherical.L1_find_radius(
            self.num_prof,
            R_max,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            tol=tol,
            num_attempt=num_attempt,
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
            tol=tol,
            num_attempt=num_attempt,
            verbosity=verbosity,
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

    def gen_prof_L1_find_M_given_R(self, M_max, tol=0.001, num_attempt=40, verbosity=1):
        """ 
        Compute the profile of a planet with 1 layer to find the mass given the 
        radius.
            
        Parameters
        ----------
        self.R or self.A1_R_layer[0] : float
            The total radius (m).

        M_max : float
            The maximum mass to try (kg).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """
        # Check for necessary input
        assert self.R is not None or self.A1_R_layer[0] is not None
        assert len(self.A1_R_layer) == 1
        self._1_layer_input()
        if self.R is None:
            self.R = self.A1_R_layer[0]

        self.M = L1_spherical.L1_find_mass(
            self.num_prof,
            self.R,
            M_max,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            tol=tol,
            num_attempt=num_attempt,
            verbosity=verbosity,
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
        """ 
        Compute the profile of a planet with 1 layer given the mass and radius.
            
        Parameters
        ----------
        self.R : float
            The total radius (m).
            
        self.M : float
            The total mass (kg).            
        """

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

    def gen_prof_L2_find_R1_given_R_M(self, tol=0.001, num_attempt=40, verbosity=1):
        """ 
        Compute the profile of a planet with 2 layers to find the outer radius 
        of the first layer, given the total mass and total radius.
            
        Parameters
        ----------
        self.R : float
            The total radius (m).
        
        self.M : float
            The total mass (kg).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """
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
            tol=tol,
            num_attempt=num_attempt,
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
            tol=tol,
            num_attempt=num_attempt,
            verbosity=verbosity,
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

    def gen_prof_L2_find_M_given_R1_R(
        self, M_max, tol=0.001, num_attempt=40, verbosity=1
    ):
        """ 
        Compute the profile of a planet with 2 layers to find the total mass
        given the outer radii of both layers.
            
        Parameters
        ----------
        self.A1_R_layer : [float]
            The radii of each layer (m).
        
        M_max : float
            The maximum mass to try (kg).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        self._2_layer_input()

        self.M = L2_spherical.L2_find_mass(
            self.num_prof,
            self.R,
            M_max,
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
            tol=tol,
            num_attempt=num_attempt,
            verbosity=verbosity,
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

    def gen_prof_L2_find_R_given_M_R1(
        self, R_max, tol=0.001, num_attempt=40, verbosity=1
    ):
        """ 
        Compute the profile of a planet with 2 layers to find the total radius
        given the total mass and the outer radius of the first layer.
            
        Parameters
        ----------
        self.M : float
            The total mass (kg).
        
        self.A1_R_layer[0] : [float]
            The outer radius of the first layer (m).
        
        R_max : float
            The maximum radius to try (m).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """
        # Check for necessary input
        assert self.A1_R_layer[0] is not None
        assert self.M is not None
        self._2_layer_input()

        self.R = L2_spherical.L2_find_radius(
            self.num_prof,
            R_max,
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
            tol=tol,
            num_attempt=num_attempt,
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
            tol=tol,
            num_attempt=num_attempt,
            verbosity=verbosity,
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

    def gen_prof_L2_find_R1_R_given_M1_M2(
        self, R_max, tol=0.001, num_attempt=40, verbosity=1
    ):
        """ 
        Compute the profile of a planet with 2 layers to find outer radii of 
        both layers given the masses of both layers.
            
        Parameters
        ----------
        self.A1_M_layer : [float]
            The masses of each layer (kg).
        
        R_max : float
            The maximum radius to try (m).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """

        # Check for necessary input
        self._2_layer_input()
        assert self.A1_M_layer[0] is not None
        assert self.A1_M_layer[1] is not None
        # Check masses
        if self.M is not None:
            assert self.M == self.A1_M_layer[0] + self.A1_M_layer[1]
        else:
            self.M = self.A1_M_layer[0] + self.A1_M_layer[1]

        self.A1_R_layer[0], self.R = L2_spherical.L2_find_R1_R(
            self.num_prof,
            R_max,
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
            tol=tol,
            num_attempt=num_attempt,
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
            tol=tol,
            num_attempt=num_attempt,
            verbosity=verbosity,
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
        """ 
        Compute the profile of a planet with 2 layers given the total mass and 
        the outer radii of both layers.
            
        Parameters
        ----------
        self.M : float
            The total mass (kg).
            
        self.A1_R_layer : [float]
            The outer radii of each layer (m).       
        """
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

    def gen_prof_L2_given_prof_L1(  # this function should go in L2_spherical.py
        self, mat, T_rho_type_id, T_rho_args, rho_min, verbosity=1
    ):
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

    def gen_prof_L2_find_R1_given_M1_add_L2(
        self, verbosity=1
    ):  # this should go to L2_spherical.py
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
        assert self.rho_min is not None

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
            R_max,
            self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            num_attempt=num_attempt,
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
        assert self.rho_min is not None

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
            M_max,
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
            num_attempt=num_attempt,
            num_attempt_2=num_attempt_2,
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

    def gen_prof_L3_find_R2_given_R_M_R1(self, tol=0.001, num_attempt=40, verbosity=1):
        """ 
        Compute the profile of a planet with 3 layers to find the outer radius 
        of the second layer, given the total mass, total radius, and outer 
        radius of the first layer.
            
        Parameters
        ----------
        self.R : float
            The total radius (m).
        
        self.A1_R_layer[0] : [float]
            The outer radius of the first layer (m).
        
        self.M : float
            The total mass (kg).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """
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
            num_attempt=num_attempt,
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

    def gen_prof_L3_find_R1_given_R_M_R2(self, tol=0.001, num_attempt=40, verbosity=1):
        """ 
        Compute the profile of a planet with 3 layers to find the outer radius 
        of the first layer, given the total mass, total radius, and outer 
        radius of the second layer.
            
        Parameters
        ----------
        self.R : float
            The total radius (m).
        
        self.A1_R_layer[1] : [float]
            The outer radius of the second layer (m).
        
        self.M : float
            The total mass (kg).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """
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
            num_attempt=num_attempt,
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

    def gen_prof_L3_find_M_given_R_R1_R2(
        self, M_max, tol=0.001, num_attempt=40, verbosity=1
    ):
        """ 
        Compute the profile of a planet with 3 layers to find the total mass
        given the outer radii of all three layers.
            
        Parameters
        ----------
        self.A1_R_layer : [float]
            The radii of each layer (m).
        
        M_max : float
            The maximum mass to try (kg).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        assert self.A1_R_layer[1] is not None
        self._3_layer_input()

        if verbosity >= 1:
            print("Finding M given R1, R2 and R...")

        self.M = L3_spherical.L3_find_mass(
            self.num_prof,
            self.R,
            M_max,
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

    def gen_prof_L3_find_R_given_M_R1_R2(
        self, R_max, tol=0.001, num_attempt=40, verbosity=1
    ):
        """ 
        Compute the profile of a planet with 3 layers to find the total radius
        given the total mass and the outer radii of the firsta and second 
        layers.
            
        Parameters
        ----------
        self.A1_R_layer[0] and [1] : float
            The radii of the first and second layers (m).
        
        self.M : float
            The total mass (kg).
        
        R_max : float
            The maximum radius to try (m).

        tol : float
            The tolerance for finding unknown parameters as a fractional 
            difference between two consecutive iterations.

        num_attempt : int
            The maximum number of iteration attempts.        
        """
        # Check for necessary input
        assert self.A1_R_layer[0] is not None
        assert self.A1_R_layer[1] is not None
        assert self.M is not None
        self._3_layer_input()

        self.R = L3_spherical.L3_find_radius(
            self.num_prof,
            R_max,
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
            num_attempt=num_attempt,
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
        """ 
        Compute the profile of a planet with 3 layers given the total mass and 
        the outer radii of all three layers.
            
        Parameters
        ----------
        self.M : float
            The total mass (kg).
            
        self.A1_R_layer : [float]
            The outer radii of each layer (m).       
        """
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
        R_max=None,
        tol_M_frac=None,
        rho_min=None,
        num_attempt=40,
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
        if R_max is not None:
            R_max = R_max
        if tol_M_frac is not None:
            tol_M_frac = tol_M_frac
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
        assert self.rho_min is not None

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
            R_max,
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
            num_attempt=num_attempt,
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


class SpinPlanet:
    """ Spinning planet class ...

    Parameters
    ----------
    name : str
        The name of the spinning planet object.
        
    planet : woma.Planet
        The spherical planet object from which to generate the spinning planet.
        
    period : float
        The rotation period for the planet (hours).

    num_prof : int
        The number of grid points used in the 1D equatorial and polar profiles, 
        i.e. the number of nested spheroids used to model the spinning planet.
    """

    def __init__(
        self, name=None, planet=None, period=None, num_prof=1000,
    ):
        self.name = name
        self.planet = planet
        self.period = period
        self.num_prof = num_prof

        # Spherical planet attributes
        self.num_layer = planet.num_layer
        self.A1_mat_layer = planet.A1_mat_layer
        self.A1_mat_id_layer = planet.A1_mat_id_layer
        self.A1_T_rho_type = planet.A1_T_rho_type
        self.A1_T_rho_type_id = planet.A1_T_rho_type_id
        self.A1_T_rho_args = planet.A1_T_rho_args
        self.A1_M_layer = planet.A1_M_layer
        self.A1_R_layer = planet.A1_R_layer
        self.A1_r = planet.A1_r
        self.A1_P = planet.A1_P
        self.A1_T = planet.A1_T
        self.A1_rho = planet.A1_rho
        self.M = planet.M
        self.P_0 = planet.P_0
        self.T_0 = planet.T_0
        self.rho_0 = planet.rho_0
        if self.num_layer > 1:
            self.P_1 = planet.P_1
            self.T_1 = planet.T_1
            self.rho_1 = planet.rho_1
        else:
            self.P_1 = None
            self.T_1 = None
            self.rho_1 = None
        if self.num_layer > 2:
            self.P_2 = planet.P_2
            self.T_2 = planet.T_2
            self.rho_2 = planet.rho_2
        else:
            self.P_2 = None
            self.T_2 = None
            self.rho_2 = None
        self.P_s = planet.P_s
        self.T_s = planet.T_s
        self.rho_s = planet.rho_s

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
        self.M = us.M_spin_planet(
            self.A1_r_eq, self.A1_rho_eq, self.A1_r_po, self.A1_rho_po
        )

        # Compute escape velocity
        v_esc_eq, v_esc_po = us.spin_escape_vel(
            self.A1_r_eq, self.A1_rho_eq, self.A1_r_po, self.A1_rho_po, self.period,
        )

        self.v_esc_po = v_esc_po
        self.v_esc_eq = v_esc_eq

        # Compute equatorial and polar radius
        self.R_e = np.max(self.A1_r_eq[self.A1_rho_eq > 0.0])
        self.R_p = np.max(self.A1_r_po[self.A1_rho_po > 0.0])

        # Mass per layer, Equatorial and polar temperature and pressure
        if self.num_layer == 1:
            self.A1_R_layer_eq = np.array([self.R_e])
            self.A1_R_layer_po = np.array([self.R_p])
            # Mass
            self.A1_M_layer = np.array([self.M])
            # Pressure and temperature
            self.A1_P_eq = np.zeros_like(self.A1_r_eq)
            self.A1_P_po = np.zeros_like(self.A1_r_po)
            self.A1_T_eq = np.zeros_like(self.A1_r_eq)
            self.A1_T_po = np.zeros_like(self.A1_r_po)
            for i, rho in enumerate(self.A1_rho_eq):
                if rho >= self.rho_s:
                    self.A1_T_eq[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_eq[i] = eos.P_T_rho(
                        self.A1_T_eq[i], rho, self.A1_mat_id_layer[0]
                    )
            for i, rho in enumerate(self.A1_rho_po):
                if rho >= self.rho_s:
                    self.A1_T_po[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_po[i] = eos.P_T_rho(
                        self.A1_T_po[i], rho, self.A1_mat_id_layer[0]
                    )
            # Mat_id
            self.A1_mat_id_eq = np.ones(self.A1_r_eq.shape) * self.A1_mat_id_layer[0]
            self.A1_mat_id_po = np.ones(self.A1_r_po.shape) * self.A1_mat_id_layer[0]

        elif self.num_layer == 2:
            self.R_1_eq = np.max(self.A1_r_eq[self.A1_rho_eq >= self.rho_1])
            self.A1_R_layer_eq = np.array([self.R_1_eq, self.R_e])
            self.R_1_po = np.max(self.A1_r_po[self.A1_rho_po >= self.rho_1])
            self.A1_R_layer_po = np.array([self.R_1_po, self.R_p])
            self.A1_mat_id_eq = (self.A1_rho_eq >= self.rho_1) * self.A1_mat_id_layer[
                0
            ] + (self.A1_rho_eq < self.rho_1) * self.A1_mat_id_layer[1]
            self.A1_mat_id_po = (self.A1_rho_po >= self.rho_1) * self.A1_mat_id_layer[
                0
            ] + (self.A1_rho_po < self.rho_1) * self.A1_mat_id_layer[1]

            self.A1_mat_id_eq[self.A1_rho_eq <= 0] = -1
            self.A1_mat_id_po[self.A1_rho_po <= 0] = -1

            self.A1_P_eq = np.zeros_like(self.A1_r_eq)
            self.A1_P_po = np.zeros_like(self.A1_r_po)
            self.A1_T_eq = np.zeros_like(self.A1_r_eq)
            self.A1_T_po = np.zeros_like(self.A1_r_po)

            for i, rho in enumerate(self.A1_rho_eq):
                if rho >= self.rho_1:
                    self.A1_T_eq[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_eq[i] = eos.P_T_rho(
                        self.A1_T_eq[i], rho, self.A1_mat_id_layer[0]
                    )
                elif rho >= self.rho_s:
                    self.A1_T_eq[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[1],
                        self.A1_T_rho_args[1],
                        self.A1_mat_id_layer[1],
                    )
                    self.A1_P_eq[i] = eos.P_T_rho(
                        self.A1_T_eq[i], rho, self.A1_mat_id_layer[1]
                    )
            for i, rho in enumerate(self.A1_rho_po):
                if rho >= self.rho_1:
                    self.A1_T_po[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_po[i] = eos.P_T_rho(
                        self.A1_T_po[i], rho, self.A1_mat_id_layer[0]
                    )
                elif rho >= self.rho_s:
                    self.A1_T_po[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[1],
                        self.A1_T_rho_args[1],
                        self.A1_mat_id_layer[1],
                    )
                    self.A1_P_po[i] = eos.P_T_rho(
                        self.A1_T_po[i], rho, self.A1_mat_id_layer[1]
                    )

            r_temp = np.copy(self.A1_r_eq)
            z_temp = np.copy(self.A1_r_po)
            rho_r_temp = np.copy(self.A1_rho_eq)
            rho_z_temp = np.copy(self.A1_rho_po)
            rho_r_temp[rho_r_temp < self.rho_1] = 0.0
            rho_z_temp[rho_z_temp < self.rho_1] = 0.0
            M1 = us.M_spin_planet(r_temp, rho_r_temp, z_temp, rho_z_temp)

            M2 = self.M - M1

            self.A1_M_layer = np.array([M1, M2])

        elif self.num_layer == 3:
            self.R_1_eq = np.max(self.A1_r_eq[self.A1_rho_eq >= self.rho_1])
            self.R_2_eq = np.max(self.A1_r_eq[self.A1_rho_eq >= self.rho_2])
            self.A1_R_layer_eq = np.array([self.R_1_eq, self.R_2_eq, self.R_e])
            self.R_1_po = np.max(self.A1_r_po[self.A1_rho_po >= self.rho_1])
            self.R_2_po = np.max(self.A1_r_po[self.A1_rho_po >= self.rho_2])
            self.A1_R_layer_po = np.array([self.R_1_po, self.R_2_po, self.R_p])
            self.A1_mat_id_eq = (
                (self.A1_rho_eq >= self.rho_1) * self.A1_mat_id_layer[0]
                + np.logical_and(
                    self.A1_rho_eq < self.rho_1, self.A1_rho_eq >= self.rho_2
                )
                * self.A1_mat_id_layer[1]
                + (self.A1_rho_eq < self.rho_2) * self.A1_mat_id_layer[2]
            )
            self.A1_mat_id_eq = (
                (self.A1_rho_po >= self.rho_1) * self.A1_mat_id_layer[0]
                + np.logical_and(
                    self.A1_rho_po < self.rho_1, self.A1_rho_po >= self.rho_2
                )
                * self.A1_mat_id_layer[1]
                + (self.A1_rho_po < self.rho_2) * self.A1_mat_id_layer[2]
            )

            self.A1_mat_id_eq[self.A1_rho_eq <= 0] = -1
            self.A1_mat_id_po[self.A1_rho_po <= 0] = -1

            self.A1_P_eq = np.zeros_like(self.A1_r_eq)
            self.A1_P_po = np.zeros_like(self.A1_r_po)
            self.A1_T_eq = np.zeros_like(self.A1_r_eq)
            self.A1_T_po = np.zeros_like(self.A1_r_po)

            for i, rho in enumerate(self.A1_rho_eq):
                if rho >= self.rho_1:
                    self.A1_T_eq[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_eq[i] = eos.P_T_rho(
                        self.A1_T_eq[i], rho, self.A1_mat_id_layer[0]
                    )
                elif rho >= self.rho_2:
                    self.A1_T_eq[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[1],
                        self.A1_T_rho_args[1],
                        self.A1_mat_id_layer[1],
                    )
                    self.A1_P_eq[i] = eos.P_T_rho(
                        self.A1_T_eq[i], rho, self.A1_mat_id_layer[1]
                    )
                elif rho >= self.rho_s:
                    self.A1_T_eq[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[2],
                        self.A1_T_rho_args[2],
                        self.A1_mat_id_layer[2],
                    )
                    self.A1_P_eq[i] = eos.P_T_rho(
                        self.A1_T_eq[i], rho, self.A1_mat_id_layer[2]
                    )
            for i, rho in enumerate(self.A1_rho_po):
                if rho >= self.rho_1:
                    self.A1_T_po[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[0],
                        self.A1_T_rho_args[0],
                        self.A1_mat_id_layer[0],
                    )
                    self.A1_P_po[i] = eos.P_T_rho(
                        self.A1_T_po[i], rho, self.A1_mat_id_layer[0]
                    )
                elif rho >= self.rho_2:
                    self.A1_T_po[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[1],
                        self.A1_T_rho_args[1],
                        self.A1_mat_id_layer[1],
                    )
                    self.A1_P_po[i] = eos.P_T_rho(
                        self.A1_T_po[i], rho, self.A1_mat_id_layer[1]
                    )
                elif rho >= self.rho_s:
                    self.A1_T_po[i] = T_rho(
                        rho,
                        self.A1_T_rho_type_id[2],
                        self.A1_T_rho_args[2],
                        self.A1_mat_id_layer[2],
                    )
                    self.A1_P_po[i] = eos.P_T_rho(
                        self.A1_T_po[i], rho, self.A1_mat_id_layer[2]
                    )

            r_temp = np.copy(self.A1_r_eq)
            z_temp = np.copy(self.A1_r_po)
            rho_r_temp = np.copy(self.A1_rho_eq)
            rho_z_temp = np.copy(self.A1_rho_po)
            rho_r_temp[rho_r_temp < self.rho_1] = 0.0
            rho_z_temp[rho_z_temp < self.rho_1] = 0.0
            M1 = us.M_spin_planet(r_temp, rho_r_temp, z_temp, rho_z_temp)

            rho_r_temp = np.copy(self.A1_rho_eq)
            rho_z_temp = np.copy(self.A1_rho_po)
            rho_r_temp[rho_r_temp < self.rho_2] = 0.0
            rho_z_temp[rho_z_temp < self.rho_2] = 0.0
            M2 = us.M_spin_planet(r_temp, rho_r_temp, z_temp, rho_z_temp)
            M2 = M2 - M1

            M3 = self.M - M2 - M1

            self.A1_M_layer = np.array([M1, M2, M3])

        self.T_0 = self.A1_T_eq[0]
        self.T_s = self.A1_T_eq[self.A1_T_eq > 0][-1]

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
            "    %s = %.5g  h", (utils.add_whitespace("period", space), self.period)
        )
        print_try(
            "    %s = %.5g  kg  = %.5g  M_earth",
            (utils.add_whitespace("M", space), self.M, self.M / gv.M_earth),
        )
        print_try(
            "    %s = %.5g  m  = %.5g  R_earth",
            (utils.add_whitespace("R_eq", space), self.R_e, self.R_e / gv.R_earth),
        )
        print_try(
            "    %s = %.5g  m  = %.5g  R_earth",
            (utils.add_whitespace("R_po", space), self.R_p, self.R_p / gv.R_earth),
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
                utils.format_array_string(self.A1_R_layer_eq / gv.R_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s  R_earth",
            (
                utils.add_whitespace("R_layer_po", space),
                utils.format_array_string(self.A1_R_layer_po / gv.R_earth, "%.5g"),
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

    def _prep_spin_profile_arrays(self, R_max_eq, R_max_po):
        # Initialize A1_rho_eq and A1_rho_po with the spherical profile
        self.A1_r_eq = np.linspace(0, R_max_eq, self.num_prof)
        self.A1_r_po = np.linspace(0, R_max_po, self.num_prof)

        rho_model = interp1d(self.A1_r, self.A1_rho, bounds_error=False, fill_value=0)

        self.A1_rho_eq = rho_model(self.A1_r_eq)
        self.A1_rho_po = rho_model(self.A1_r_po)

    def find_min_period(self, max_period=10, max_iter=20, verbosity=1):

        min_period = us.find_min_period(
            self.num_layer,
            self.A1_r_eq,
            self.A1_rho_eq,
            self.A1_r_po,
            self.A1_rho_po,
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

    def _spin_planet_simple(
        self, 
        R_max_eq, 
        R_max_po, 
        check_min_period, 
        tol_density_profile, 
        max_iter_1,
        max_iter_2,
        verbosity=1,
    ):
        """ 
        Create a spinning planet from a spherical one.
        """
        # Set up the spheroid equatorial and polar arrays
        self._prep_spin_profile_arrays(R_max_eq, R_max_po)

        for i in tqdm(
            range(max_iter_1),
            desc="Computing spinning profile",
            disable=verbosity == 0,
        ):
            # Compute min_period
            if check_min_period:
                self.find_min_period(max_iter=max_iter_2, verbosity=0)

                # Select period for this iteration
                if self.period >= self.min_period:
                    period_iter = self.period
                else:
                    period_iter = self.min_period
            else:
                period_iter = self.period

            # Compute profile
            A1_rho_eq, A1_rho_po = us.spin_iteration(
                period_iter,
                self.num_layer,
                self.A1_r_eq,
                self.A1_rho_eq,
                self.A1_r_po,
                self.A1_rho_po,
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

            # Convergence criterion
            criterion = np.mean(np.abs(A1_rho_eq - self.A1_rho_eq) / self.rho_s)

            # Save results
            self.A1_rho_eq = A1_rho_eq
            self.A1_rho_po = A1_rho_po

            # Check if there is convergence
            if criterion < tol_density_profile:
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
        self,
        R_max_eq,
        R_max_po,
        check_min_period,
        tol_layer_masses,
        tol_density_profile,
        max_iter_1,
        max_iter_2,
        verbosity=1,
    ):
        """ 
        Create a spinning planet from a spherical one, keeping the same layer 
        masses, for a 1 layer planet.
        """

        assert self.num_layer == 1

        # Default max radii
        if R_max_eq is None:
            R_max_eq = 2 * self.R
        if R_max_po is None:
            R_max_po = 1.2 * self.R

        f_min = 0.0
        f_max = 1.0

        M_fixed = self.M

        for i in tqdm(
            range(max_iter_1), desc="Computing spinning profile", disable=verbosity == 0
        ):

            f = np.mean([f_min, f_max])

            # Shrink the input spherical planet
            self.planet.A1_R_layer = f * self.planet.A1_R_layer
            self.planet.R = f * self.planet.R

            # Make the new spherical profiles
            self.planet.M_max = self.planet.M
            self.planet.gen_prof_L1_find_M_given_R(verbosity=0)

            # Create the spinning profiles
            self._spin_planet_simple(
                R_max_eq, 
                R_max_po, 
                check_min_period, 
                tol_density_profile, 
                max_iter_1,
                max_iter_2,
                verbosity=0,
            )

            criterion = np.abs(self.M - M_fixed) / M_fixed < tol_layer_masses

            if criterion:
                if verbosity >= 1:
                    print("Tolerance level criteria reached.")
                    self.print_info()
                return

            if spin_planet.M > M_fixed:
                f_max = f
            else:
                f_min = f

        if verbosity >= 1:
            self.print_info()

    def _L2_spin_planet_fix_M(
        self,
        R_max_eq,
        R_max_po,
        check_min_period,
        tol_layer_masses,
        tol_density_profile,
        max_iter_1,
        max_iter_2,
        verbosity=1,
    ):
        """ 
        Create a spinning planet from a spherical one, keeping the same layer 
        masses, for a 2 layer planet.
        """

        assert self.num_layer == 2

        # Default max radii
        if R_max_eq is None:
            R_max_eq = 2 * self.R
        if R_max_po is None:
            R_max_po = 1.2 * self.R

        M_fixed = self.M
        M0_fixed = self.A1_M_layer[0]

        # Create the spinning profiles
        self._spin_planet_simple(
            R_max_eq, 
            R_max_po, 
            check_min_period, 
            tol_density_profile, 
            max_iter_1,
            max_iter_2,
            verbosity=0,
        )

        for k in tqdm(
            range(max_iter_2), desc="Computing spinning profile", disable=verbosity == 0
        ):

            if self.M > M_fixed:
                R_mantle_min = self.planet.A1_R_layer[0]
                R_mantle_max = self.planet.A1_R_layer[1]
            else:
                R_mantle_min = self.planet.A1_R_layer[1]
                R_mantle_max = 1.1 * self.planet.A1_R_layer[1]

            for i in tqdm(range(max_iter_1), desc="Adjusting outer edge", disable=True):

                R_mantle = np.mean([R_mantle_min, R_mantle_max])

                # Modify the input spherical planet boundaries
                self.planet.A1_R_layer[1] = R_mantle
                self.planet.R = R_mantle

                # Make the new spherical profiles
                self.planet.gen_prof_L2_find_M_given_R1_R(M_max=1.2 * self.M, verbosity=0)

                # Create the spinning profiles
                self._spin_planet_simple(
                    R_max_eq, 
                    R_max_po, 
                    check_min_period, 
                    tol_density_profile, 
                    max_iter_1,
                    max_iter_2,
                    verbosity=0,
                )

                criterion_1 = np.abs(self.M - M_fixed) / M_fixed < tol_layer_masses
                criterion_2 = (
                    np.abs(self.A1_M_layer[0] - M0_fixed) / M0_fixed < tol_layer_masses
                )

                if criterion_1 and criterion_2:
                    if verbosity >= 1:
                        print("Tolerance level criteria reached.")
                        self.print_info()
                    return

                if criterion_1:
                    break

                if self.M > M_fixed:
                    R_mantle_max = R_mantle
                else:
                    R_mantle_min = R_mantle

            if self.A1_M_layer[0] / self.M > M0_fixed / M_fixed:
                R_core_min = 0
                R_core_max = self.planet.A1_R_layer[0]
            else:
                R_core_min = self.planet.A1_R_layer[0]
                R_core_max = self.planet.A1_R_layer[1]

            for i in tqdm(
                range(max_iter_1), desc="Adjusting layer 1 and 2 boundary", disable=True
            ):

                R_core = np.mean([R_core_min, R_core_max])

                # Modify the input spherical planet boundaries
                self.planet.A1_R_layer[0] = R_core
                self.planet.A1_R_layer[1] = R_mantle
                self.planet.R = R_mantle

                # Make the new spherical profiles
                self.planet.gen_prof_L2_find_M_given_R1_R(M_max=1.2 * self.M, verbosity=0)

                # Create the spinning profiles
                self._spin_planet_simple(
                    R_max_eq, 
                    R_max_po, 
                    check_min_period, 
                    tol_density_profile, 
                    max_iter_1,
                    max_iter_2,
                    verbosity=0,
                )

                criterion_1 = np.abs(self.M - M_fixed) / M_fixed < tol_layer_masses
                criterion_2 = (
                    np.abs(self.A1_M_layer[0] - M0_fixed) / M0_fixed < tol_layer_masses
                )

                if criterion_1 and criterion_2:
                    if verbosity >= 1:
                        print("Tolerance level criteria reached.")
                        self.print_info()

                if criterion_2:
                    break

                if self.A1_M_layer[0] / self.M > M0_fixed / M_fixed:
                    R_core_max = R_core
                else:
                    R_core_min = R_core

        if verbosity >= 1:
            self.print_info()

    def spin(
        self,
        fix_mass=True,
        R_max_eq=None,
        R_max_po=None,
        check_min_period=True,
        tol_density_profile=0.001,
        tol_layer_masses=0.01,
        max_iter_1=15,
        max_iter_2=15,
        verbosity=1,
    ):
        """ Create the spinning planet from the spherical one.

        Parameters
        ----------        
        fix_mass : bool
            If True (default), then iterate the input mass to ensure that the  
            final spinning mass is the same. If False, then more quickly create 
            the spinning profiles directly from the spherical one.

        R_max_eq : float
            Maximum equatorial radius (m). Defaults to 2 times the spherical 
            radius.
            
        R_max_po : float
            Maximum polar radius (m). Defaults to 1.2 times the spherical radius.
            
        check_min_period : bool
            Checks if the period provided is less than the minimum physically 
            allowed. Slow -- set True only if required for extremely high spin.
            
        tol_density_profile : float
            The iterative search will end when the fractional differences 
            between the density profiles in successive iterations is less than 
            this tolerance.
            
        tol_layer_masses : float
            The iterative search will end when the fractional differences 
            between the layer masses of the spinning planet and the spherical 
            one are less than this tolerance.
            
        max_iter_1: int
            Maximum number of iterations allowed. Inner loop.
            
        max_iter_2: int
            Maximum number of iterations allowed. Outer loop.
        """
        # Check for necessary input
        self._check_input()

        if R_max_eq is None:
            R_max_eq = 2 * self.planet.R
        if R_max_po is None:
            R_max_po = 1.2 * self.planet.R

        if fix_mass:

            if self.planet.num_layer == 1:

                self._L1_spin_planet_fix_M(
                    R_max_eq,
                    R_max_po,
                    check_min_period,
                    tol_layer_masses,
                    tol_density_profile,
                    max_iter_1,
                    verbosity=verbosity,
                )

            elif self.planet.num_layer == 2:

                self._L2_spin_planet_fix_M(
                    R_max_eq,
                    R_max_po,
                    check_min_period,
                    tol_layer_masses,
                    tol_density_profile,
                    max_iter_1,
                    max_iter_2,
                    verbosity=verbosity,
                )

            elif self.planet.num_layer == 3:

                raise ValueError("3 layers not implemented yet")

            else:

                raise ValueError("self.planet.num_layer must be 1, 2, or 3")

        else:
            self._spin_planet_simple(
                R_max_eq,
                R_max_po,
                check_min_period,
                tol_density_profile,
                max_iter_1,
                max_iter_2,
                verbosity=verbosity,
            )


class ParticlePlanet:
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
        Array of material IDs for all particles. See the README.md documentation.
        
    A1_id : [int]
        Array of IDs for all particles.        
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
                ) = L1_spin.L1_place_particles(
                    planet.A1_r_eq,
                    planet.A1_rho_eq,
                    planet.A1_r_po,
                    planet.A1_rho_po,
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
                ) = L2_spin.L2_place_particles(
                    planet.A1_r_eq,
                    planet.A1_rho_eq,
                    planet.A1_r_po,
                    planet.A1_rho_po,
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
                ) = L3_spin.L3_place_particles(
                    planet.A1_r_eq,
                    planet.A1_rho_eq,
                    planet.A1_r_po,
                    planet.A1_rho_po,
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
