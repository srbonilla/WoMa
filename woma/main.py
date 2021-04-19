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
import h5py
from scipy.interpolate import interp1d
from copy import deepcopy
import seagen

from woma.spherical_funcs import L1_spherical, L2_spherical, L3_spherical
import woma.spin_funcs.utils_spin as us
from woma.misc import glob_vars as gv
from woma.misc import utils, io
from woma.eos import eos
from woma.eos.T_rho import T_rho, T_rho_id_and_args_from_type


class Planet:
    """Create model profiles of a spherical body in hydrostatic equilibrium.

    See also README.md and tutorial.ipynb.

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

        "power=alpha"   T ~ rho^alpha. Set alpha = 0 for isothermal.
        "adiabatic"     Adiabatic.
        "entropy=s"     Fixed specific entropy s (J kg^-1 K^-1).

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

    load_file : str (opt.)
        If provided, then load the attributes and profiles from an HDF5 file.
        See woma.Planet.save() and .load().

    Attributes (in addition to the input parameters)
    ----------
    num_layer : int
        The number of planetary layers.

    P_0, P_1, ... P_s;  T_0, ..., T_s;  rho_0, ..., rho_s : float
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
        num_prof=1000,
        load_file=None,
        verbosity=1,
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

        # Load eos tables
        utils.load_eos_tables(self.A1_mat_layer)

        # Load from file
        if load_file is not None:
            self.load(load_file, verbosity=verbosity)
            return

        # Derived or default attributes
        # Number of layers
        if self.A1_mat_layer is not None:
            self.num_layer = len(self.A1_mat_layer)
            self.A1_mat_id_layer = [gv.Di_mat_id[mat] for mat in self.A1_mat_layer]
        else:
            # Placeholder
            self.num_layer = 1

        # Temperature--density relation
        if self.A1_T_rho_type is not None:
            self.A1_T_rho_type_id, self.A1_T_rho_args = T_rho_id_and_args_from_type(
                self.A1_T_rho_type
            )
            # Whether to calculate the surface temperature from a fixed entropy
            if self.A1_T_rho_type_id[-1] == gv.type_ent:
                do_T_from_fixed_s = True
            else:
                do_T_from_fixed_s = False
        else:
            do_T_from_fixed_s = False

        # T and P or rho must be provided at the surface to calculate the third,
        # or a fixed entropy to calculate P and rho
        if self.P_s is not None and self.P_s <= 0:
            e = "Pressure at surface must be > 0."
            raise ValueError(e)
        if self.T_s is not None and self.T_s <= 0:
            e = "Temperature at surface must be > 0."
            raise ValueError(e)
        if self.rho_s is not None and self.rho_s <= 0:
            e = "Density at surface must be > 0."
            raise ValueError(e)
        if self.T_s is None and not do_T_from_fixed_s:
            e = "Temperature at surface must be provided."
            raise ValueError(e)
        if self.P_s is not None:
            self.rho_s = eos.rho_P_T(self.P_s, self.T_s, self.A1_mat_id_layer[-1])
        elif self.rho_s is not None:
            if do_T_from_fixed_s:
                self.T_s = eos.sesame.T_rho_s(
                    self.rho_s, self.A1_T_rho_args[-1][0], self.A1_mat_id_layer[-1]
                )
            self.P_s = eos.P_T_rho(self.T_s, self.rho_s, self.A1_mat_id_layer[-1])
            if self.P_s <= 0:
                e = (
                    "Pressure at surface computed is not positive.\n"
                    "Please modify temperature and/or density at surface."
                )
                raise ValueError(e)
        else:
            e = "Temperature and pressure or density at surface must be provided."
            raise ValueError(e)

        # Default filename and layer arrays
        if self.A1_M_layer is None:
            self.A1_M_layer = [None] * self.num_layer
        if self.A1_R_layer is None:
            self.A1_R_layer = [None] * self.num_layer
        if self.R is None and (self.A1_R_layer[-1] is not None):
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
        if self.num_layer >= 2:
            self.A1_M_layer[1:] -= self.A1_M_layer[:-1]
        self.M = np.sum(self.A1_M_layer)

        # Moment of inertia factor
        self.I_MR2 = utils.moi(self.A1_r, self.A1_rho) / (self.M * self.R ** 2)

        # P, T, and rho at the centre and the outer boundary of each layer
        self.P_0 = self.A1_P[0]
        self.T_0 = self.A1_T[0]
        self.rho_0 = self.A1_rho[0]
        if self.num_layer >= 2:
            self.P_1 = self.A1_P[self.A1_idx_layer[0]]
            self.T_1 = self.A1_T[self.A1_idx_layer[0]]
            self.rho_1 = self.A1_rho[self.A1_idx_layer[0]]
        if self.num_layer >= 3:
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
        if self.num_layer >= 3:
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
        if self.num_layer >= 2:
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

    def save(self, filename, verbosity=1):
        """Save the attributes and profiles to an HDF5 file.

        Parameters
        ----------
        filename : str
            The data file path.
        """
        filename = utils.check_end(filename, ".hdf5")

        if verbosity >= 1:
            print('Saving "%s"...' % filename[-60:], end=" ", flush=True)

        with h5py.File(filename, "w") as f:
            # Group
            grp = f.create_group("/planet")
            # Attributes
            grp.attrs[io.Di_hdf5_planet_label["num_layer"]] = self.num_layer
            grp.attrs[io.Di_hdf5_planet_label["mat_layer"]] = self.A1_mat_layer
            grp.attrs[io.Di_hdf5_planet_label["mat_id_layer"]] = self.A1_mat_id_layer
            grp.attrs[io.Di_hdf5_planet_label["T_rho_type"]] = self.A1_T_rho_type
            grp.attrs[io.Di_hdf5_planet_label["T_rho_type_id"]] = self.A1_T_rho_type_id
            grp.attrs[io.Di_hdf5_planet_label["T_rho_args"]] = self.A1_T_rho_args
            grp.attrs[io.Di_hdf5_planet_label["R_layer"]] = self.A1_R_layer
            grp.attrs[io.Di_hdf5_planet_label["M_layer"]] = self.A1_M_layer
            grp.attrs[io.Di_hdf5_planet_label["M"]] = self.M
            grp.attrs[io.Di_hdf5_planet_label["R"]] = self.R
            grp.attrs[io.Di_hdf5_planet_label["idx_layer"]] = self.A1_idx_layer
            grp.attrs[io.Di_hdf5_planet_label["P_s"]] = self.P_s
            grp.attrs[io.Di_hdf5_planet_label["T_s"]] = self.T_s
            grp.attrs[io.Di_hdf5_planet_label["rho_s"]] = self.rho_s
            # Profiles
            grp.create_dataset(io.Di_hdf5_planet_label["r"], data=self.A1_r, dtype="d")
            grp.create_dataset(
                io.Di_hdf5_planet_label["m_enc"], data=self.A1_m_enc, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_planet_label["rho"], data=self.A1_rho, dtype="d"
            )
            grp.create_dataset(io.Di_hdf5_planet_label["T"], data=self.A1_T, dtype="d")
            grp.create_dataset(io.Di_hdf5_planet_label["P"], data=self.A1_P, dtype="d")
            grp.create_dataset(io.Di_hdf5_planet_label["u"], data=self.A1_u, dtype="d")
            grp.create_dataset(
                io.Di_hdf5_planet_label["mat_id"], data=self.A1_mat_id, dtype="i"
            )

        if verbosity >= 1:
            print("Done")

    def load(self, filename, verbosity=1):
        """Load the attributes and profiles from an HDF5 file.

        Parameters
        ----------
        filename : str
            The data file path.
        """
        filename = utils.check_end(filename, ".hdf5")

        if verbosity >= 1:
            print('Loading "%s"...' % filename[-60:], end=" ", flush=True)

        with h5py.File(filename, "r") as f:
            (
                # Attributes
                self.num_layer,
                self.A1_mat_layer,
                self.A1_mat_id_layer,
                self.A1_T_rho_type,
                self.A1_T_rho_type_id,
                self.A1_T_rho_args,
                self.A1_R_layer,
                self.A1_M_layer,
                self.M,
                self.R,
                self.A1_idx_layer,
                self.P_s,
                self.T_s,
                self.rho_s,
                # Profiles
                self.A1_r,
                self.A1_m_enc,
                self.A1_rho,
                self.A1_T,
                self.A1_P,
                self.A1_u,
                self.A1_mat_id,
            ) = io.multi_get_planet_data(
                f,
                [
                    # Attributes
                    "num_layer",
                    "mat_layer",
                    "mat_id_layer",
                    "T_rho_type",
                    "T_rho_type_id",
                    "T_rho_args",
                    "R_layer",
                    "M_layer",
                    "M",
                    "R",
                    "idx_layer",
                    "P_s",
                    "T_s",
                    "rho_s",
                    # Profiles
                    "r",
                    "m_enc",
                    "rho",
                    "T",
                    "P",
                    "u",
                    "mat_id",
                ],
            )

        self.update_attributes()

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

    def gen_prof_L1_find_R_given_M(
        self, R_max, tol=0.001, tol_M_tweak=1e-7, num_attempt=40, verbosity=1
    ):
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

        tol_M_tweak : float
            The tolerance for tweaking the mass to avoid density peaks at the
            center; the relative difference between consecutive masses.

        num_attempt : int
            The maximum number of iteration attempts.
        """
        # Check for necessary input
        assert self.M is not None
        self._1_layer_input()

        self.R = L1_spherical.L1_find_R_given_M(
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

        if verbosity < 2:
            verbosity_2 = 0
        else:
            verbosity_2 = verbosity
        if verbosity == 1:
            print(
                "Tweaking M to avoid density peaks at the center of the planet...",
                end=" ",
                flush=True,
            )
        if verbosity >= 2:
            print("Tweaking M to avoid density peaks at the center of the planet...")

        self.M = L1_spherical.L1_find_M_given_R(
            self.num_prof,
            self.R,
            1.05 * self.M,
            self.P_s,
            self.T_s,
            self.rho_s,
            self.A1_mat_id_layer[0],
            self.A1_T_rho_type_id[0],
            self.A1_T_rho_args[0],
            tol=tol_M_tweak,
            num_attempt=num_attempt,
            verbosity=verbosity_2,
        )

        if verbosity >= 1:
            print("Done")

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

    def gen_prof_L1_find_M_given_R(self, M_max, tol=1e-7, num_attempt=40, verbosity=1):
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

        self.M = L1_spherical.L1_find_M_given_R(
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

    def gen_prof_given_inner_prof(
        self, mat, T_rho_type, rho_min=0, P_min=0, verbosity=1
    ):
        """Add a new layer on top of existing profiles by integrating outwards.

        i.e. The self Planet object must already have valid profiles as
        generated by another WoMa function, then this function will increase
        the number of layers by one.

        In addition to this function's arguments, try changing the surface
        conditions (P_s, T_s, rho_s) and remaking the initial planet to control
        this new layer, as these set the conditions at its base.

        Parameters
        ----------
        mat : str
            The name of the material in the new layer. See Di_mat_id in
            `eos/eos.py`.

        T_rho_type : int
            The type of temperature-density relation in the new layer. See
            Di_mat_id in `eos/eos.py`.

            "power=alpha"   T = K * rho^alpha.
            "adiabatic"     Adiabatic.

        rho_min : float
            The minimum density (must be >= 0) at which the new layer will stop.

        P_min : float
            The minimum pressure (must be >= 0) at which the new layer will stop.
        """
        # Append the new layer info to the existing profiles
        self.num_layer += 1
        self.A1_mat_layer = np.append(self.A1_mat_layer, mat)
        mat_id = gv.Di_mat_id[mat]
        utils.load_eos_tables(self.A1_mat_layer)  # load new material table
        self.A1_mat_id_layer = np.append(self.A1_mat_id_layer, mat_id)

        T_rho_type_id, T_rho_args = T_rho_id_and_args_from_type([T_rho_type])
        self.A1_T_rho_type_id = np.append(self.A1_T_rho_type_id, T_rho_type_id)
        self.A1_T_rho_args = np.array(
            np.append(self.A1_T_rho_args, T_rho_args, axis=0), dtype="float"
        )

        assert rho_min >= 0
        assert P_min >= 0

        self.rho_min = rho_min
        self.P_min = P_min

        # Make sure the profile arrays are ordered by increasing radius
        if self.A1_r[-1] < self.A1_r[0]:
            self.A1_r = self.A1_r[::-1]
            self.A1_m_enc = self.A1_m_enc[::-1]
            self.A1_P = self.A1_P[::-1]
            self.A1_T = self.A1_T[::-1]
            self.A1_rho = self.A1_rho[::-1]
            self.A1_u = self.A1_u[::-1]
            self.A1_mat_id = self.A1_mat_id[::-1]

        # Integrate the profiles outwards
        (
            A1_r,
            A1_m_enc,
            A1_P,
            A1_T,
            A1_rho,
            A1_u,
            A1_mat_id,
        ) = L1_spherical.L1_integrate_out(
            self.A1_r[-1],
            self.A1_r[1],
            self.A1_m_enc[-1],
            self.A1_P[-1],
            self.A1_T[-1],
            self.A1_u[-1],
            self.A1_mat_id_layer[-1],
            self.A1_T_rho_type_id[-1],
            self.A1_T_rho_args[-1],
            rho_min=self.rho_min,
            P_min=self.P_min,
        )

        # Apppend the new layer to the full profiles
        self.A1_r = np.append(self.A1_r, A1_r)
        self.A1_m_enc = np.append(self.A1_m_enc, A1_m_enc)
        self.A1_P = np.append(self.A1_P, A1_P)
        self.A1_T = np.append(self.A1_T, A1_T)
        self.A1_rho = np.append(self.A1_rho, A1_rho)
        self.A1_u = np.append(self.A1_u, A1_u)
        self.A1_mat_id = np.append(self.A1_mat_id, A1_mat_id)

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

    def gen_prof_L2_find_R1_given_M_R(
        self, tol=0.001, tol_M_tweak=1e-7, num_attempt=40, verbosity=1
    ):
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

        tol_M_tweak : float
            The tolerance for tweaking the mass to avoid density peaks at the
            center; the relative difference between consecutive masses.

        num_attempt : int
            The maximum number of iteration attempts.
        """
        # Check for necessary input
        assert self.R is not None
        assert self.M is not None
        self._2_layer_input()

        self.A1_R_layer[0] = L2_spherical.L2_find_R1_given_M_R(
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

        if verbosity < 2:
            verbosity_2 = 0
        else:
            verbosity_2 = verbosity
        if verbosity == 1:
            print(
                "Tweaking M to avoid density peaks at the center of the planet...",
                end=" ",
                flush=True,
            )
        if verbosity >= 2:
            print("Tweaking M to avoid density peaks at the center of the planet...")

        self.M = L2_spherical.L2_find_M_given_R_R1(
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
            tol=tol_M_tweak,
            num_attempt=num_attempt,
            verbosity=verbosity_2,
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
            print("Done")
            self.print_info()

    def gen_prof_L2_find_M_given_R_R1(
        self, M_max, tol=1e-7, num_attempt=40, verbosity=1
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

        self.M = L2_spherical.L2_find_M_given_R_R1(
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
            self.print_info()

    def gen_prof_L2_find_R_given_M_R1(
        self, R_max, tol=0.001, tol_M_tweak=1e-7, num_attempt=40, verbosity=1
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

        tol_M_tweak : float
            The tolerance for tweaking the mass to avoid density peaks at the
            center; the relative difference between consecutive masses.

        num_attempt : int
            The maximum number of iteration attempts.
        """
        # Check for necessary input
        assert self.A1_R_layer[0] is not None
        assert self.M is not None
        self._2_layer_input()

        self.R = L2_spherical.L2_find_R_given_M_R1(
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

        if verbosity < 2:
            verbosity_2 = 0
        else:
            verbosity_2 = verbosity
        if verbosity == 1:
            print(
                "Tweaking M to avoid density peaks at the center of the planet...",
                end="  ",
                flush=True,
            )
        if verbosity >= 2:
            print("Tweaking M to avoid density peaks at the center of the planet...")

        self.M = L2_spherical.L2_find_M_given_R_R1(
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
            tol=tol_M_tweak,
            num_attempt=num_attempt,
            verbosity=verbosity_2,
        )

        if verbosity >= 1:
            print("Done")

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

    def gen_prof_L2_find_R_R1_given_M1_M2(
        self, R_min, R_max, tol=0.001, tol_M_tweak=1e-7, num_attempt=40, verbosity=1
    ):
        """
        Compute the profile of a planet with 2 layers to find outer radii of
        both layers given the masses of both layers.

        Parameters
        ----------
        self.A1_M_layer : [float]
            The masses of each layer (kg).

        R_min : float
            The minimum radius to try (m).

        R_max : float
            The maximum radius to try (m).

        tol : float
            The tolerance for finding unknown parameters as a fractional
            difference between two consecutive iterations.

        tol_M_tweak : float
            The tolerance for tweaking the mass to avoid density peaks at the
            center; the relative difference between consecutive masses.

        num_attempt : int
            The maximum number of iteration attempts.
        """

        # Check for necessary input
        self._2_layer_input()
        assert self.A1_M_layer[0] is not None
        assert self.A1_M_layer[1] is not None
        self.M = self.A1_M_layer[0] + self.A1_M_layer[1]

        self.A1_R_layer[0], self.R = L2_spherical.L2_find_R_R1_given_M1_M2(
            self.num_prof,
            R_min,
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

        if verbosity < 2:
            verbosity_2 = 0
        else:
            verbosity_2 = verbosity
        if verbosity == 1:
            print(
                "Tweaking M to avoid density peaks at the center of the planet...",
                end=" ",
                flush=True,
            )
        if verbosity >= 2:
            print("Tweaking M to avoid density peaks at the center of the planet...")

        self.M = L2_spherical.L2_find_M_given_R_R1(
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
            tol=tol_M_tweak,
            num_attempt=num_attempt,
            verbosity=verbosity_2,
        )

        if verbosity >= 1:
            print("Done")

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

    def gen_prof_L3_find_M_given_R_R1_R2(
        self, M_max, tol=1e-7, num_attempt=40, verbosity=1
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

        self.M = L3_spherical.L3_find_M_given_R_R1_R2(
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
            num_attempt=num_attempt,
            tol=tol,
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

    def gen_prof_L3_find_R1_given_M_R_R2(
        self, tol=0.001, tol_M_tweak=1e-7, num_attempt=40, verbosity=1
    ):
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

        tol_M_tweak : float
            The tolerance for tweaking the mass to avoid density peaks at the
            center; the relative difference between consecutive masses.

        num_attempt : int
            The maximum number of iteration attempts.
        """
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer is not None
        assert self.A1_R_layer[1] is not None
        assert self.M is not None
        self._3_layer_input()

        self.A1_R_layer[0] = L3_spherical.L3_find_R1_given_M_R_R2(
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
            tol=tol,
            verbosity=verbosity,
        )

        if verbosity < 2:
            verbosity_2 = 0
        else:
            verbosity_2 = verbosity
        if verbosity == 1:
            print(
                "Tweaking M to avoid density peaks at the center of the planet...",
                end=" ",
                flush=True,
            )
        if verbosity >= 2:
            print("Tweaking M to avoid density peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_M_given_R_R1_R2(
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
            num_attempt=num_attempt,
            tol=tol_M_tweak,
            verbosity=verbosity_2,
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
            print("Done")
            self.print_info()

    def gen_prof_L3_find_R2_given_M_R_R1(
        self, tol=0.001, tol_M_tweak=1e-7, num_attempt=40, verbosity=1
    ):
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

        tol_M_tweak : float
            The tolerance for tweaking the mass to avoid density peaks at the
            center; the relative difference between consecutive masses.

        num_attempt : int
            The maximum number of iteration attempts.
        """
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        assert self.M is not None
        self._3_layer_input()

        self.A1_R_layer[1] = L3_spherical.L3_find_R2_given_M_R_R1(
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
            tol=tol,
            verbosity=verbosity,
        )

        if verbosity < 2:
            verbosity_2 = 0
        else:
            verbosity_2 = verbosity
        if verbosity == 1:
            print(
                "Tweaking M to avoid density peaks at the center of the planet...",
                end=" ",
                flush=True,
            )
        if verbosity >= 2:
            print("Tweaking M to avoid density peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_M_given_R_R1_R2(
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
            num_attempt=num_attempt,
            tol=tol_M_tweak,
            verbosity=verbosity_2,
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
            print("Done")
            self.print_info()

    def gen_prof_L3_find_R_given_M_R1_R2(
        self, R_max, tol=0.001, tol_M_tweak=1e-7, num_attempt=40, verbosity=1
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

        tol_M_tweak : float
            The tolerance for tweaking the mass to avoid density peaks at the
            center; the relative difference between consecutive masses.

        num_attempt : int
            The maximum number of iteration attempts.
        """
        # Check for necessary input
        assert self.A1_R_layer[0] is not None
        assert self.A1_R_layer[1] is not None
        assert self.M is not None
        self._3_layer_input()

        self.R = L3_spherical.L3_find_R_given_M_R1_R2(
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
            tol=tol,
            verbosity=verbosity,
        )
        self.A1_R_layer[-1] = self.R

        if verbosity < 2:
            verbosity_2 = 0
        else:
            verbosity_2 = verbosity
        if verbosity == 1:
            print(
                "Tweaking M to avoid density peaks at the center of the planet...",
                end="  ",
                flush=True,
            )
        if verbosity >= 2:
            print("Tweaking M to avoid density peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_M_given_R_R1_R2(
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
            num_attempt=num_attempt,
            tol=tol_M_tweak,
            verbosity=verbosity_2,
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
            print("Done")
            self.print_info()

    def gen_prof_L3_find_R1_R2_given_M_R_I(
        self,
        R_1_min,
        R_1_max,
        tol=0.001,
        tol_M_tweak=1e-7,
        num_attempt=40,
        num_attempt_2=40,
        verbosity=1,
    ):
        """
        Compute the profile of a planet with 3 layers to find the boundary core-mantle,
        and mantle-atmosphere given the total mass, the outer radii, and
        the moment of inertia factor.

        Parameters
        ----------
        self.R : float
            The radius of the planet (m).

        self.M : float
            The total mass (kg).

        R_1_min : float
            The minimum boundary core-mantle to try (m).

        R_1_max : float
            The maximum boundary core-mantle to try (m).

        tol : float
            The tolerance for finding unknown parameters as a fractional
            difference between two consecutive iterations.

        tol_M_tweak : float
            The tolerance for tweaking the mass to avoid density peaks at the
            center; the relative difference between consecutive masses.

        num_attempt : int
            The maximum number of iteration attempts. Outer loop.

        num_attempt_2 : int
            The maximum number of iteration attempts. Inner loop.
        """
        # Check for necessary input
        assert self.R is not None
        assert self.M is not None
        assert self.I_MR2 is not None
        self._3_layer_input()

        self.A1_R_layer[0], self.A1_R_layer[1] = L3_spherical.L3_find_R1_R2_given_M_R_I(
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
            R_1_min,
            R_1_max,
            num_attempt=num_attempt,
            num_attempt_2=num_attempt_2,
            tol=tol,
            verbosity=verbosity,
        )

        if verbosity < 2:
            verbosity_2 = 0
        else:
            verbosity_2 = verbosity
        if verbosity == 1:
            print(
                "Tweaking M to avoid density peaks at the center of the planet...",
                end=" ",
                flush=True,
            )
        if verbosity >= 2:
            print("Tweaking M to avoid density peaks at the center of the planet...")

        self.M = L3_spherical.L3_find_M_given_R_R1_R2(
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
            num_attempt=num_attempt,
            tol=tol_M_tweak,
            verbosity=verbosity_2,
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
            print("Done")
            self.print_info()

    def gen_prof_L3_find_R_R1_R2_given_M1_M2_M3(self):  ### WIP
        return None


class SpinPlanet:
    """Create nested-spheroid profiles of a spinning body in equilibrium.
    The planet spins with positive z-axis angular velocity.

    See also README.md and tutorial.ipynb.

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

    R_max_eq : float
        Maximum equatorial radius (m). Defaults to 2 times the spherical
        radius.

    R_max_po : float
        Maximum polar radius (m). Defaults to 1.2 times the spherical radius.

    f_iter : float
        Fractional radius to proceed with iterative method of finding
        spinning planet given a fixed mass.

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

    num_attempt_1: int
        Maximum number of iterations allowed. Inner loop.

    num_attempt_2: int
        Maximum number of iterations allowed. Outer loop.

    load_file : str (opt.)
        If provided, then load the attributes and profiles from an HDF5 file.
        See woma.SpinPlanet.save() and .load().


    Attributes (in addition to the input parameters)
    ----------
    M : float
        The total mass (kg).

    R_eq, R_po : float
        The equatorial and polar radii (m).

    A1_M_layer : [float]
        The mass within each layer, starting from the from the central layer
        outwards (kg).

    A1_R, A1_Z : [float]
        The semi-major (equatorial) and semi-minor (polar) radii of the
        nested spheroids (m).

    A1_m : [float]
        The mass of each spheroid (kg).

    A1_rho, A1_P, A1_T, A1_u : [float]
        The pressure (Pa), density (kg m^-3), temperature (K), and specific
        internal energy (J kg^-1) of each spheroid.

    A1_mat_id : [int]
        The material ID of each spheroid. (See the README.md documentation.)

    P_0, P_1, ... P_s;  T_0, ..., T_s;  rho_0, ..., rho_s : float
        The pressure (Pa), temperature (K), and density (kg m^-3) at each layer
        boundary, from the centre (_0) up to the surface (_s).

    I_MR2 : float
        The moment of inertia factor using the equatorial radius.

    L : float
        The rotational angular momentum (kg m^2 s^-1).
    """

    def __init__(
        self,
        planet=None,
        period=None,
        name=None,
        num_prof=1000,
        fix_mass=True,
        R_max_eq=None,
        R_max_po=None,
        f_iter=None,
        check_min_period=False,
        tol_density_profile=0.001,
        tol_layer_masses=0.01,
        num_attempt_1=15,
        num_attempt_2=15,
        load_file=None,
        verbosity=1,
    ):
        # Load from file
        if load_file is not None:
            self.load(load_file)
            return
        # Otherwise, require planet and period inputs
        else:
            assert isinstance(planet, Planet)
            assert period is not None

        self.name = name
        self.planet = deepcopy(planet)
        self.period = period
        self.num_prof = num_prof
        self.fix_mass = fix_mass
        self.R_max_eq = R_max_eq
        self.R_max_po = R_max_po
        self.f_iter = f_iter
        self.check_min_period = check_min_period
        self.tol_density_profile = tol_density_profile
        self.tol_layer_masses = tol_layer_masses
        self.num_attempt_1 = num_attempt_1
        self.num_attempt_2 = num_attempt_2
        self.verbosity = verbosity

        # Inherit and initialise attributes from the spherical planet
        self.num_layer = planet.num_layer
        self.A1_mat_layer = planet.A1_mat_layer
        self.A1_mat_id_layer = planet.A1_mat_id_layer
        self.A1_T_rho_type = planet.A1_T_rho_type
        self.A1_T_rho_type_id = planet.A1_T_rho_type_id
        self.A1_T_rho_args = planet.A1_T_rho_args
        self.P_s = self.planet.P_s
        self.T_s = self.planet.T_s
        self.rho_s = self.planet.rho_s
        self.P_0 = self.planet.P_0
        self.T_0 = self.planet.T_0
        self.rho_0 = self.planet.rho_0
        if self.num_layer >= 2:
            self.P_1 = self.planet.P_1
            self.T_1 = self.planet.T_1
            self.rho_1 = self.planet.rho_1
        if self.num_layer >= 3:
            self.P_2 = self.planet.P_2
            self.T_2 = self.planet.T_2
            self.rho_2 = self.planet.rho_2

        # Load eos tables
        utils.load_eos_tables(self.A1_mat_layer)

        # Make the spinning profiles!
        self.spin(
            R_max_eq=self.R_max_eq,
            R_max_po=self.R_max_po,
            f_iter=self.f_iter,
            fix_mass=self.fix_mass,
            check_min_period=self.check_min_period,
            tol_density_profile=self.tol_density_profile,
            tol_layer_masses=self.tol_layer_masses,
            num_attempt_1=self.num_attempt_1,
            num_attempt_2=self.num_attempt_2,
            verbosity=self.verbosity,
        )

        if self.verbosity >= 1:
            self.print_info()

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

    def _find_boundary_indices(self):
        """ Find the indices of the outermost elements in each layer. """
        # First index above each layer
        A1_idx_layer_eq = np.array([np.argmax(self.A1_rho_eq == 0)])
        A1_idx_layer_po = np.array([np.argmax(self.A1_rho_po == 0)])
        if self.num_layer >= 2:
            A1_idx_layer_eq = np.append(
                np.argmax(self.A1_rho_eq <= self.planet.rho_1), A1_idx_layer_eq
            )
            A1_idx_layer_po = np.append(
                np.argmax(self.A1_rho_po <= self.planet.rho_1), A1_idx_layer_po
            )
        if self.num_layer >= 3:
            A1_idx_layer_eq = np.append(
                np.argmax(self.A1_rho_eq <= self.planet.rho_2), A1_idx_layer_eq
            )
            A1_idx_layer_po = np.append(
                np.argmax(self.A1_rho_po <= self.planet.rho_2), A1_idx_layer_po
            )

        return A1_idx_layer_eq - 1, A1_idx_layer_po - 1

    def _update_internal_attributes(self):
        """ Update the attributes required for the internal iterations. """
        self.A1_idx_layer_eq, self.A1_idx_layer_po = self._find_boundary_indices()

        # Nested spheroid properties
        self.A1_rho = self.A1_rho_eq[: self.A1_idx_layer_eq[-1] + 1]
        self.A1_R = self.A1_r_eq[: self.A1_idx_layer_eq[-1] + 1]
        # Find polar radii by interpolating between the polar densities
        rho_model_po_inv = interp1d(self.A1_rho_po, self.A1_r_po)
        self.A1_Z = rho_model_po_inv(self.A1_rho)

        # Enclosed, total, and layer masses
        try:
            self.A1_m = us.spheroid_masses(self.A1_R, self.A1_Z, self.A1_rho)
        except:
            e = (
                "Period too low. Please consider the following:\n"
                "increase period, increase R_max_eq, "
                "or enable check_min_period=True."
            )
            raise ValueError(e)
        self.M = np.sum(self.A1_m)

        self.A1_M_layer = np.array(
            [np.sum(self.A1_m[: idx + 1]) for idx in self.A1_idx_layer_eq]
        )
        if self.num_layer >= 2:
            self.A1_M_layer[1:] -= self.A1_M_layer[0]
        if self.num_layer >= 3:
            self.A1_M_layer[2:] -= self.A1_M_layer[1]

    def update_attributes(self):
        """ Update the attributes for the final output planet. """
        self._update_internal_attributes()

        # Update A1_T_rho_args
        self.A1_T_rho_args = self.planet.A1_T_rho_args

        # Escape speed
        self.v_esc_eq, self.v_esc_po = us.spin_escape_vel(
            self.A1_r_eq,
            self.A1_rho_eq,
            self.A1_r_po,
            self.A1_rho_po,
            self.period,
        )

        # Equatorial and polar radii
        self.R_eq = self.A1_r_eq[self.A1_idx_layer_eq[-1]]
        self.R_po = self.A1_r_po[self.A1_idx_layer_po[-1]]

        # Nested spheroid properties (removing massless outside "spheroids")
        self.A1_m = self.A1_m[: self.A1_idx_layer_eq[-1] + 1]
        self.A1_rho = self.A1_rho_eq[: self.A1_idx_layer_eq[-1] + 1]
        self.A1_R = self.A1_r_eq[: self.A1_idx_layer_eq[-1] + 1]
        self.A1_R_layer = np.array([self.A1_R[idx] for idx in self.A1_idx_layer_eq])

        # Find polar radii by interpolating between the polar densities
        rho_model_po_inv = interp1d(self.A1_rho_po, self.A1_r_po)
        self.A1_Z = rho_model_po_inv(self.A1_rho)
        self.A1_Z_layer = np.array([self.A1_Z[idx] for idx in self.A1_idx_layer_eq])

        self.A1_mat_id = np.ones_like(self.A1_R)
        self.A1_T = np.ones_like(self.A1_R)
        self.A1_P = np.ones_like(self.A1_R)
        self.A1_u = np.ones_like(self.A1_R)

        self.A1_mat_id[:] = self.A1_mat_id_layer[0]
        if self.num_layer >= 2:
            self.A1_mat_id[(self.A1_idx_layer_eq[0] + 1) :] = self.A1_mat_id_layer[1]
        if self.num_layer >= 3:
            self.A1_mat_id[(self.A1_idx_layer_eq[1] + 1) :] = self.A1_mat_id_layer[2]

        # Set values through each layer
        for i, rho in enumerate(self.A1_rho[: self.A1_idx_layer_eq[0] + 1]):
            self.A1_T[i] = T_rho(
                rho,
                self.A1_T_rho_type_id[0],
                self.A1_T_rho_args[0],
                self.A1_mat_id_layer[0],
            )
            self.A1_u[i] = eos.u_rho_T(rho, self.A1_T[i], self.A1_mat_id[i])
            # self.A1_P[i] = eos.P_u_rho(self.A1_u[i], rho, self.A1_mat_id[i])
            self.A1_P[i] = eos.P_T_rho(self.A1_T[i], rho, self.A1_mat_id[i])

        if self.num_layer >= 2:
            for i, rho in enumerate(
                self.A1_rho[self.A1_idx_layer_eq[0] + 1 : self.A1_idx_layer_eq[1] + 1]
            ):
                j = self.A1_idx_layer_eq[0] + 1 + i
                self.A1_T[j] = T_rho(
                    rho,
                    self.A1_T_rho_type_id[1],
                    self.A1_T_rho_args[1],
                    self.A1_mat_id_layer[1],
                )
                self.A1_u[j] = eos.u_rho_T(rho, self.A1_T[j], self.A1_mat_id[j])
                # self.A1_P[j] = eos.P_u_rho(self.A1_u[j], rho, self.A1_mat_id[j])
                self.A1_P[j] = eos.P_T_rho(self.A1_T[j], rho, self.A1_mat_id[j])
        if self.num_layer >= 3:
            for i, rho in enumerate(
                self.A1_rho[self.A1_idx_layer_eq[1] + 1 : self.A1_idx_layer_eq[2] + 1]
            ):
                j = self.A1_idx_layer_eq[1] + 1 + i
                self.A1_T[j] = T_rho(
                    rho,
                    self.A1_T_rho_type_id[2],
                    self.A1_T_rho_args[2],
                    self.A1_mat_id_layer[2],
                )
                self.A1_u[j] = eos.u_rho_T(rho, self.A1_T[j], self.A1_mat_id[j])
                # self.A1_P[j] = eos.P_u_rho(self.A1_u[j], rho, self.A1_mat_id[j])
                self.A1_P[j] = eos.P_T_rho(self.A1_T[j], rho, self.A1_mat_id[j])

        # Boundary values
        self.P_0 = self.A1_P[0]
        self.T_0 = self.A1_T[0]
        self.rho_0 = self.A1_rho[0]
        self.P_s = self.A1_P[-1]
        self.T_s = self.A1_T[-1]
        self.rho_s = self.A1_rho[-1]
        if self.num_layer >= 2:
            self.P_1 = self.A1_P[self.A1_idx_layer_eq[0]]
            self.T_1 = self.A1_T[self.A1_idx_layer_eq[0]]
            self.rho_1 = self.A1_rho[self.A1_idx_layer_eq[0]]
        if self.num_layer >= 3:
            self.P_2 = self.A1_P[self.A1_idx_layer_eq[1]]
            self.T_2 = self.A1_T[self.A1_idx_layer_eq[1]]
            self.rho_2 = self.A1_rho[self.A1_idx_layer_eq[1]]

        # Moment of inertia
        # I_z = int dm*(x^2 + y^2) = int rho r_{xy}^3 dr_{xy} dphi dz
        # compute for constant-density spheroids with density d_rho,
        # then sum for every spheroid
        # int r_{xy}^3 dr_{xy} dz = 8/15 pi R^4 Z
        A1_drho = np.append(self.A1_rho, 0)
        A1_drho = A1_drho[:-1] - A1_drho[1:]

        self.I_MR2 = np.sum(8 / 15 * np.pi * A1_drho * self.A1_R ** 4 * self.A1_Z)
        self.I_MR2 /= self.M * self.R_eq ** 2

        # Angular momentum
        hours_to_sec = 60 * 60
        w = 2 * np.pi / self.period / hours_to_sec
        self.L = self.I_MR2 * self.M * self.R_eq ** 2 * w

    def print_info(self):
        """ Print the main properties. """
        # Print and catch if any variables are None
        def print_try(string, variables):
            try:
                print(string % variables)
            except TypeError:
                print("    %s = None" % variables[0])

        self.update_attributes()

        space = 12
        print_try('SpinPlanet "%s": ', self.name)
        print_try(
            '    %s = "%s"', (utils.add_whitespace("planet", space), self.planet.name)
        )
        print_try(
            "    %s = %.5g  h", (utils.add_whitespace("period", space), self.period)
        )
        print_try(
            "    %s = %.5g  kg  = %.5g  M_earth",
            (utils.add_whitespace("M", space), self.M, self.M / gv.M_earth),
        )
        print_try(
            "    %s = %.5g  m  = %.5g  R_earth",
            (utils.add_whitespace("R_eq", space), self.R_eq, self.R_eq / gv.R_earth),
        )
        print_try(
            "    %s = %.5g  m  = %.5g  R_earth",
            (utils.add_whitespace("R_po", space), self.R_po, self.R_po / gv.R_earth),
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
                utils.add_whitespace("R_layer", space),
                utils.format_array_string(self.A1_R_layer / gv.R_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s  R_earth",
            (
                utils.add_whitespace("Z_layer", space),
                utils.format_array_string(self.A1_Z_layer / gv.R_earth, "%.5g"),
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
        if self.num_layer >= 3:
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
        if self.num_layer >= 2:
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
            "    %s = %.5g  M_tot R_eq^2",
            (utils.add_whitespace("I_MR2", space), self.I_MR2),
        )
        print_try(
            "    %s = %.5g  kg m^2 s^-1",
            (utils.add_whitespace("L", space), self.L),
        )

    def save(self, filename, verbosity=1):
        """Save the attributes and profiles to an HDF5 file.

        Parameters
        ----------
        filename : str
            The data file path.
        """
        filename = utils.check_end(filename, ".hdf5")

        if verbosity >= 1:
            print('Saving "%s"...' % filename[-60:], end=" ", flush=True)

        with h5py.File(filename, "w") as f:
            # Spherical planet group
            grp = f.create_group("/planet")
            # Attributes
            grp.attrs[io.Di_hdf5_planet_label["num_layer"]] = self.planet.num_layer
            grp.attrs[io.Di_hdf5_planet_label["mat_layer"]] = self.planet.A1_mat_layer
            grp.attrs[
                io.Di_hdf5_planet_label["mat_id_layer"]
            ] = self.planet.A1_mat_id_layer
            grp.attrs[io.Di_hdf5_planet_label["T_rho_type"]] = self.planet.A1_T_rho_type
            grp.attrs[
                io.Di_hdf5_planet_label["T_rho_type_id"]
            ] = self.planet.A1_T_rho_type_id
            grp.attrs[io.Di_hdf5_planet_label["T_rho_args"]] = self.planet.A1_T_rho_args
            grp.attrs[io.Di_hdf5_planet_label["R_layer"]] = self.planet.A1_R_layer
            grp.attrs[io.Di_hdf5_planet_label["M_layer"]] = self.planet.A1_M_layer
            grp.attrs[io.Di_hdf5_planet_label["M"]] = self.planet.M
            grp.attrs[io.Di_hdf5_planet_label["R"]] = self.planet.R
            grp.attrs[io.Di_hdf5_planet_label["idx_layer"]] = self.planet.A1_idx_layer
            grp.attrs[io.Di_hdf5_planet_label["P_s"]] = self.planet.P_s
            grp.attrs[io.Di_hdf5_planet_label["T_s"]] = self.planet.T_s
            grp.attrs[io.Di_hdf5_planet_label["rho_s"]] = self.planet.rho_s
            # Profiles
            grp.create_dataset(
                io.Di_hdf5_planet_label["r"], data=self.planet.A1_r, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_planet_label["m_enc"], data=self.planet.A1_m_enc, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_planet_label["rho"], data=self.planet.A1_rho, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_planet_label["T"], data=self.planet.A1_T, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_planet_label["P"], data=self.planet.A1_P, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_planet_label["u"], data=self.planet.A1_u, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_planet_label["mat_id"], data=self.planet.A1_mat_id, dtype="i"
            )

            # Spinning planet group
            grp = f.create_group("/spin_planet")
            # Attributes
            grp.attrs[io.Di_hdf5_spin_label["period"]] = self.period
            grp.attrs[io.Di_hdf5_spin_label["num_layer"]] = self.num_layer
            grp.attrs[io.Di_hdf5_spin_label["mat_layer"]] = self.A1_mat_layer
            grp.attrs[io.Di_hdf5_spin_label["mat_id_layer"]] = self.A1_mat_id_layer
            grp.attrs[io.Di_hdf5_spin_label["T_rho_type"]] = self.A1_T_rho_type
            grp.attrs[io.Di_hdf5_spin_label["T_rho_type_id"]] = self.A1_T_rho_type_id
            grp.attrs[io.Di_hdf5_spin_label["T_rho_args"]] = self.A1_T_rho_args
            grp.attrs[io.Di_hdf5_spin_label["R_layer"]] = self.A1_R_layer
            grp.attrs[io.Di_hdf5_spin_label["Z_layer"]] = self.A1_Z_layer
            grp.attrs[io.Di_hdf5_spin_label["M_layer"]] = self.A1_M_layer
            grp.attrs[io.Di_hdf5_spin_label["M"]] = self.M
            grp.attrs[io.Di_hdf5_spin_label["R_eq"]] = self.R_eq
            grp.attrs[io.Di_hdf5_spin_label["R_po"]] = self.R_po
            grp.attrs[io.Di_hdf5_spin_label["idx_layer"]] = self.A1_idx_layer_eq
            grp.attrs[io.Di_hdf5_spin_label["P_s"]] = self.P_s
            grp.attrs[io.Di_hdf5_spin_label["T_s"]] = self.T_s
            grp.attrs[io.Di_hdf5_spin_label["rho_s"]] = self.rho_s
            # Spheroid profiles
            grp.create_dataset(
                io.Di_hdf5_spin_label["r_eq"], data=self.A1_r_eq, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_spin_label["r_po"], data=self.A1_r_po, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_spin_label["rho_eq"], data=self.A1_rho_eq, dtype="d"
            )
            grp.create_dataset(
                io.Di_hdf5_spin_label["rho_po"], data=self.A1_rho_po, dtype="d"
            )
            grp.create_dataset(io.Di_hdf5_spin_label["R"], data=self.A1_R, dtype="d")
            grp.create_dataset(io.Di_hdf5_spin_label["Z"], data=self.A1_Z, dtype="d")
            grp.create_dataset(io.Di_hdf5_spin_label["m"], data=self.A1_m, dtype="d")
            grp.create_dataset(
                io.Di_hdf5_spin_label["rho"], data=self.A1_rho, dtype="d"
            )
            grp.create_dataset(io.Di_hdf5_spin_label["T"], data=self.A1_T, dtype="d")
            grp.create_dataset(io.Di_hdf5_spin_label["P"], data=self.A1_P, dtype="d")
            grp.create_dataset(io.Di_hdf5_spin_label["u"], data=self.A1_u, dtype="d")
            grp.create_dataset(
                io.Di_hdf5_spin_label["mat_id"], data=self.A1_mat_id, dtype="i"
            )

        if verbosity >= 1:
            print("Done")

    def load(self, filename, verbosity=1):
        """Load the attributes and profiles from an HDF5 file.

        Parameters
        ----------
        filename : str
            The data file path.
        """
        filename = utils.check_end(filename, ".hdf5")

        if verbosity >= 1:
            print('Loading "%s"...' % filename[-60:], end=" ", flush=True)

        # Spherical planet
        self.planet = Planet(load_file=filename, verbosity=0)

        # Spinning planet
        with h5py.File(filename, "r") as f:
            (
                # Attributes
                self.period,
                self.num_layer,
                self.A1_mat_layer,
                self.A1_mat_id_layer,
                self.A1_T_rho_type,
                self.A1_T_rho_type_id,
                self.A1_T_rho_args,
                self.A1_R_layer,
                self.A1_Z_layer,
                self.A1_M_layer,
                self.M,
                self.R_eq,
                self.R_po,
                self.A1_idx_layer_eq,
                self.P_s,
                self.T_s,
                self.rho_s,
                # Spheroid profiles
                self.A1_r_eq,
                self.A1_r_po,
                self.A1_rho_eq,
                self.A1_rho_po,
                self.A1_R,
                self.A1_Z,
                self.A1_m,
                self.A1_rho,
                self.A1_T,
                self.A1_P,
                self.A1_u,
                self.A1_mat_id,
            ) = io.multi_get_spin_planet_data(
                f,
                [
                    # Attributes
                    "period",
                    "num_layer",
                    "mat_layer",
                    "mat_id_layer",
                    "T_rho_type",
                    "T_rho_type_id",
                    "T_rho_args",
                    "R_layer",
                    "Z_layer",
                    "M_layer",
                    "M",
                    "R_eq",
                    "R_po",
                    "idx_layer",
                    "P_s",
                    "T_s",
                    "rho_s",
                    # Profiles
                    "r_eq",
                    "r_po",
                    "rho_eq",
                    "rho_po",
                    "R",
                    "Z",
                    "m",
                    "rho",
                    "T",
                    "P",
                    "u",
                    "mat_id",
                ],
            )

        self.update_attributes()

        if verbosity >= 1:
            print("Done")

    def _prep_spin_profile_arrays(self, R_max_eq, R_max_po):
        # Initialize A1_rho_eq and A1_rho_po with the spherical profile
        self.A1_r_eq = np.linspace(0, R_max_eq, self.num_prof)
        self.A1_r_po = np.linspace(0, R_max_po, self.num_prof)

        rho_model = interp1d(
            self.planet.A1_r, self.planet.A1_rho, bounds_error=False, fill_value=0
        )

        self.A1_rho_eq = rho_model(self.A1_r_eq)
        self.A1_rho_po = rho_model(self.A1_r_po)

    def find_min_period(self, max_period=10, tol=0.01, num_attempt=20, verbosity=1):
        ### Need to pass arguments on from spin functions
        min_period = us.find_min_period(
            self.num_layer,
            self.A1_r_eq,
            self.A1_rho_eq,
            self.A1_r_po,
            self.A1_rho_po,
            self.planet.P_0,
            self.planet.P_s,
            self.planet.rho_0,
            self.planet.rho_s,
            self.planet.A1_mat_id_layer,
            self.planet.A1_T_rho_type_id,
            self.planet.A1_T_rho_args,
            self.planet.P_1,
            self.planet.P_2,
            max_period,
            num_attempt=num_attempt,
            tol=tol,
            verbosity=verbosity,
        )

        self.min_period = min_period

    def _spin_planet_simple(
        self,
        R_max_eq,
        R_max_po,
        tol_density_profile=0.001,
        check_min_period=False,
        verbosity=1,
        num_attempt=20,
        num_attempt_find_min_period=20,
    ):
        """
        Create a spinning planet from a spherical one.
        """
        # Set up the spheroid equatorial and polar arrays
        self._prep_spin_profile_arrays(R_max_eq, R_max_po)

        # Iterate updating the density profiles and the effective potential
        for i in range(num_attempt):
            # Check the current period is not below the minimum
            if check_min_period:
                self.find_min_period(
                    num_attempt=num_attempt_find_min_period, verbosity=verbosity
                )

                # Select period for this iteration
                if self.period_input >= self.min_period:
                    period_iter = self.period_input
                else:
                    period_iter = self.min_period
            # Assume we're above the minimum period
            else:
                period_iter = self.period_input

            # Compute the spinning profiles
            A1_rho_eq, A1_rho_po = us.spin_iteration(
                period_iter,
                self.num_layer,
                self.A1_r_eq,
                self.A1_rho_eq,
                self.A1_r_po,
                self.A1_rho_po,
                self.planet.P_0,
                self.planet.P_s,
                self.planet.rho_0,
                self.planet.rho_s,
                self.planet.A1_mat_id_layer,
                self.planet.A1_T_rho_type_id,
                self.planet.A1_T_rho_args,
                self.planet.P_1,
                self.planet.P_2,
            )

            if np.any(np.isnan(A1_rho_eq)):
                raise ValueError(
                    "Equatorial profile has nan values. Please report this."
                )

            # Check the fractional change in the density profile for convergence
            tol_reached = np.mean(np.abs(A1_rho_eq - self.A1_rho_eq) / self.rho_s)

            # Print progress
            if verbosity >= 1:
                print(
                    "\rIter %d(%d): tol=%.2g(%.2g)"
                    % (i + 1, num_attempt, tol_reached, tol_density_profile),
                    end="  ",
                    flush=True,
                )

            # Save results
            self.A1_rho_eq = A1_rho_eq
            self.A1_rho_po = A1_rho_po

            # Stop once converged
            if tol_reached < tol_density_profile:
                if verbosity >= 1:
                    print("")
                break

        if self.period_input < period_iter:
            if verbosity >= 1:
                print("")
                print("Minimum period found at", period_iter, "h")
            self.period = period_iter

        self._update_internal_attributes()

        if verbosity >= 2:
            self.print_info()

    def _L1_spin_planet_fix_M_bisection(
        self,
        R_max_eq,
        R_max_po,
        check_min_period=False,
        tol_layer_masses=0.01,
        tol_density_profile=0.001,
        num_attempt=15,
        verbosity=1,
    ):
        """
        Create a spinning planet from a spherical one, keeping the same layer
        masses, for a 1 layer planet.
        """

        assert self.num_layer == 1

        # Initialise bisection over the spherical planet radius
        f_min = 0.0
        f_max = 1.0

        # Desired mass
        M_fixed = self.planet.M

        if check_min_period:
            verbosity_2 = verbosity - 1
        else:
            verbosity_2 = 0

        # Create the spinning profiles
        self._spin_planet_simple(
            R_max_eq,
            R_max_po,
            tol_density_profile=tol_density_profile,
            check_min_period=check_min_period,
            verbosity=verbosity_2,
        )

        # Check the fractional error in the mass for convergence
        M_discrep = np.abs(self.M - M_fixed) / M_fixed

        # No need to search if simple spin does the job (e.g. very high periods)
        if M_discrep < tol_layer_masses:
            if verbosity >= 2:
                self.print_info()
            return

        # Vary the spherical planet radius to fix the spinning planet mass
        for i in range(num_attempt):

            # Bisection
            f = np.mean([f_min, f_max])

            # Shrink the input spherical planet
            self.planet.A1_R_layer = f * self.A1_R_layer_original
            self.planet.R = f * self.R_original

            # Make the new spherical profiles, set a large max mass if needed
            try:
                self.planet.gen_prof_L1_find_M_given_R(M_max=1.2 * M_fixed, verbosity=0)
            except ValueError:
                M_max = 10 * np.pi * self.planet.R ** 3 * self.planet.rho_0
                self.planet.gen_prof_L1_find_M_given_R(M_max=M_max, verbosity=0)

            # Create the spinning profiles
            self._spin_planet_simple(
                R_max_eq,
                R_max_po,
                tol_density_profile=tol_density_profile,
                check_min_period=check_min_period,
                verbosity=verbosity_2,
            )

            # Check the fractional error in the mass for convergence
            M_discrep = np.abs(self.M - M_fixed) / M_fixed

            # Print progress
            if verbosity >= 1:
                print(
                    "\rIter %d(%d): R=%.5gR_E: tol=%.2g(%.2g)"
                    % (
                        i + 1,
                        num_attempt,
                        self.planet.R / gv.R_earth,
                        M_discrep,
                        tol_layer_masses,
                    ),
                    end="  ",
                    flush=True,
                )

            # Stop once converged
            if M_discrep < tol_layer_masses:
                if verbosity >= 1:
                    print("")
                break

            # Update the bounds
            if self.M > M_fixed:
                f_max = f
            else:
                f_min = f

    def _L1_spin_planet_fix_M_steps(
        self,
        R_max_eq,
        R_max_po,
        f_iter,
        check_min_period=False,
        tol_layer_masses=0.01,
        tol_density_profile=0.001,
        num_attempt=15,
        verbosity=1,
    ):
        """
        Create a spinning planet from a spherical one, keeping the same layer
        masses, for a 1 layer planet.
        """

        assert self.num_layer == 1

        # Desired masses
        M_fixed = self.planet.M

        if check_min_period:
            verbosity_2 = verbosity - 1
        else:
            verbosity_2 = 0

        # Create the spinning profiles
        self._spin_planet_simple(
            R_max_eq,
            R_max_po,
            tol_density_profile=tol_density_profile,
            check_min_period=check_min_period,
            verbosity=verbosity_2,
        )

        # Check the fractional error in the mass for convergence
        M_discrep = np.abs(self.M - M_fixed) / M_fixed

        # No need to search if simple spin does the job (e.g. very high periods)
        if M_discrep < tol_layer_masses:
            if verbosity >= 1:
                print("Simple spin method sufficient")

            if verbosity >= 2:
                self.print_info()
            return

        # Define dr
        dr_default = f_iter * self.planet.R

        for i in range(num_attempt):
            # Reduce dr as mass discrepancy decreases
            dr = (
                (M_discrep - tol_layer_masses) * 5 * dr_default / 100 / tol_layer_masses
            )
            dr += dr_default
            dr = np.abs(dr)

            if self.M > M_fixed:
                self.planet.A1_R_layer[0] = self.planet.R - dr
                self.planet.R = self.planet.R - dr
            else:
                self.planet.A1_R_layer[0] = self.planet.R + dr
                self.planet.R = self.planet.R + dr

            # Make the new spherical profiles
            self.planet.gen_prof_L1_find_M_given_R(M_max=1.2 * M_fixed, verbosity=0)

            self._spin_planet_simple(
                R_max_eq,
                R_max_po,
                tol_density_profile=tol_density_profile,
                check_min_period=check_min_period,
                verbosity=verbosity_2,
            )

            # Check the fractional error in the mass for convergence
            M_discrep = (self.M - M_fixed) / M_fixed

            # Print progress
            if verbosity >= 1:
                print(
                    "Iter %d(%d): R=%.4gR_E: tol=%.2g(%.2g)"
                    % (
                        i + 1,
                        num_attempt,
                        self.planet.R / gv.R_earth,
                        M_discrep,
                        tol_layer_masses,
                    ),
                    end="\n",
                    flush=True,
                )

            # Stop once converged
            if np.abs(M_discrep) < tol_layer_masses:
                if verbosity >= 1:
                    print("")
                if verbosity >= 2:
                    self.print_info()

                return

    def _L2_spin_planet_fix_M_bisection(
        self,
        R_max_eq,
        R_max_po,
        check_min_period=False,
        tol_layer_masses=0.01,
        tol_density_profile=0.001,
        num_attempt_1=15,
        num_attempt_2=15,
        verbosity=1,
    ):
        """
        Create a spinning planet from a spherical one, keeping the same layer
        masses, for a 2 layer planet.
        """

        assert self.num_layer == 2

        # Desired masses
        M_fixed = self.planet.M
        M_0_fixed = self.planet.A1_M_layer[0]

        if check_min_period:
            verbosity_2 = verbosity - 1
        else:
            verbosity_2 = 0

        # Create the spinning profiles
        self._spin_planet_simple(
            R_max_eq,
            R_max_po,
            tol_density_profile=tol_density_profile,
            check_min_period=check_min_period,
            verbosity=verbosity_2,
        )

        # Check the fractional error in the masses for convergence
        M_discrep = np.abs(self.M - M_fixed) / M_fixed
        M_0_discrep = np.abs(self.A1_M_layer[0] - M_0_fixed) / M_0_fixed

        # No need to search if simple spin does the job (e.g. very high periods)
        if M_discrep < tol_layer_masses and M_0_discrep < tol_layer_masses:
            if verbosity >= 1:
                print("Simple spin method sufficient")

            if verbosity >= 2:
                self.print_info()
            return

        # Vary the spherical planet radii to fix the spinning planet masses
        for i in range(num_attempt_1):

            # Initialise the bounds
            if self.M > M_fixed:
                R_min = self.planet.A1_R_layer[0]
                R_max = self.planet.A1_R_layer[1]
            else:
                R_min = self.planet.A1_R_layer[1]
                R_max = 1.2 * self.planet.A1_R_layer[1]

            # Vary the outer radius to fix the spinning total mass
            for j in range(num_attempt_2):

                # Bisect
                R = np.mean([R_min, R_max])

                # Modify the input spherical planet boundaries
                self.planet.A1_R_layer[1] = R
                self.planet.R = R

                # Make the new spherical profiles
                self.planet.gen_prof_L2_find_M_given_R_R1(
                    M_max=1.2 * M_fixed, verbosity=0
                )

                # Create the spinning profiles
                self._spin_planet_simple(
                    R_max_eq,
                    R_max_po,
                    tol_density_profile=tol_density_profile,
                    check_min_period=check_min_period,
                    verbosity=verbosity_2,
                )

                # Check the fractional error in the masses for convergence
                M_discrep = np.abs(self.M - M_fixed) / M_fixed
                M_0_discrep = np.abs(self.A1_M_layer[0] - M_0_fixed) / M_0_fixed

                # Print progress
                if verbosity >= 1:
                    print(
                        "\rIter %d(%d),%d(%d): R=%.3gR_E R1=%.3gR_E: tol=%.2g(%.2g),%.2g(%.2g)"
                        % (
                            i + 1,
                            num_attempt_1,
                            j + 1,
                            2 * num_attempt_2,
                            self.planet.R / gv.R_earth,
                            self.planet.A1_R_layer[0] / gv.R_earth,
                            M_discrep,
                            tol_layer_masses,
                            M_0_discrep,
                            tol_layer_masses,
                        ),
                        end="  ",
                        flush=True,
                    )

                # Stop once converged
                if M_discrep < tol_layer_masses and M_0_discrep < tol_layer_masses:
                    if verbosity >= 1:
                        print("")
                    if verbosity >= 2:
                        self.print_info()

                    return

                elif M_discrep < tol_layer_masses:
                    break

                # Update the bounds
                if self.M > M_fixed:
                    R_max = R
                else:
                    R_min = R

            # Initialise the bounds
            if self.A1_M_layer[0] / self.M > M_0_fixed / M_fixed:
                R_1_min = 0
                R_1_max = self.planet.A1_R_layer[0]
            else:
                R_1_min = self.planet.A1_R_layer[0]
                R_1_max = self.planet.A1_R_layer[1]

            # Vary the inner radius to fix the spinning inner layer mass
            for j in range(num_attempt_2):

                # Bisect
                R_1 = np.mean([R_1_min, R_1_max])

                # Modify the input spherical planet boundaries
                self.planet.A1_R_layer[0] = R_1
                self.planet.A1_R_layer[1] = R
                self.planet.R = R

                # Make the new spherical profiles, set a large max mass if needed
                try:
                    self.planet.gen_prof_L2_find_M_given_R_R1(
                        M_max=1.2 * M_fixed, verbosity=0
                    )
                except ValueError:
                    M_max = 10 * np.pi * self.planet.R ** 3 * self.planet.rho_0
                    self.planet.gen_prof_L2_find_M_given_R_R1(M_max=M_max, verbosity=0)

                # Create the spinning profiles
                self._spin_planet_simple(
                    R_max_eq,
                    R_max_po,
                    tol_density_profile=tol_density_profile,
                    check_min_period=check_min_period,
                    verbosity=verbosity_2,
                )

                # Check the fractional error in the masses for convergence
                M_discrep = np.abs(self.M - M_fixed) / M_fixed
                M_0_discrep = np.abs(self.A1_M_layer[0] - M_0_fixed) / M_0_fixed

                # Print progress
                if verbosity >= 1:
                    print(
                        "\rIter %d(%d),%d(%d): R=%.3gR_E R1=%.3gR_E: tol=%.2g(%.2g),%.2g(%.2g)"
                        % (
                            i + 1,
                            num_attempt_1,
                            num_attempt_1 + j + 1,
                            2 * num_attempt_2,
                            self.planet.R / gv.R_earth,
                            self.planet.A1_R_layer[0] / gv.R_earth,
                            M_discrep,
                            tol_layer_masses,
                            M_0_discrep,
                            tol_layer_masses,
                        ),
                        end="  ",
                        flush=True,
                    )

                # Stop once converged
                if M_discrep < tol_layer_masses and M_0_discrep < tol_layer_masses:
                    if verbosity >= 1:
                        print("")
                    if verbosity >= 2:
                        self.print_info()

                    return

                elif M_0_discrep < tol_layer_masses:
                    break

                # Update the bounds
                if self.A1_M_layer[0] / self.M > M_0_fixed / M_fixed:
                    R_1_max = R_1
                else:
                    R_1_min = R_1

    def _L2_spin_planet_fix_M_steps(
        self,
        R_max_eq,
        R_max_po,
        f_iter,
        check_min_period=False,
        tol_layer_masses=0.01,
        tol_density_profile=0.001,
        num_attempt_1=15,
        num_attempt_2=15,
        verbosity=1,
    ):
        """
        Create a spinning planet from a spherical one, keeping the same layer
        masses, for a 2 layer planet.
        """

        assert self.num_layer == 2

        # Desired masses
        M_fixed = self.planet.M
        M_0_fixed = self.planet.A1_M_layer[0]
        M_1_fixed = self.planet.A1_M_layer[1]

        # Create the spinning profiles
        if check_min_period:
            verbosity_2 = verbosity - 1
        else:
            verbosity_2 = 0
        self._spin_planet_simple(
            R_max_eq,
            R_max_po,
            tol_density_profile=tol_density_profile,
            check_min_period=check_min_period,
            verbosity=verbosity_2,
        )

        # Check the fractional error in the masses for convergence
        M_0_discrep = np.abs(self.A1_M_layer[0] - M_0_fixed) / M_0_fixed
        M_1_discrep = np.abs(self.A1_M_layer[1] - M_1_fixed) / M_1_fixed

        # No need to search if simple spin does the job (e.g. very high periods)
        if M_1_discrep < tol_layer_masses and M_0_discrep < tol_layer_masses:
            if verbosity >= 1:
                print("Simple spin method sufficient")

            if verbosity >= 2:
                self.print_info()
            return

        # Define dr
        dr_default = f_iter * self.planet.R

        for i in range(num_attempt_1):
            # First iteration (fix M_1)
            for j in range(num_attempt_2):
                # Reduce dr as mass discrepancy decreases
                dr = (
                    (M_1_discrep - tol_layer_masses)
                    * 5
                    * dr_default
                    / 100
                    / tol_layer_masses
                )
                dr += dr_default
                dr = np.abs(dr)

                if self.A1_M_layer[1] > M_1_fixed:
                    self.planet.A1_R_layer[1] = self.planet.R - dr
                    self.planet.R = self.planet.R - dr
                else:
                    self.planet.A1_R_layer[1] = self.planet.R + dr
                    self.planet.R = self.planet.R + dr

                # Make the new spherical profiles
                self.planet.gen_prof_L2_find_M_given_R_R1(
                    M_max=1.2 * M_fixed, verbosity=0
                )

                self._spin_planet_simple(
                    R_max_eq,
                    R_max_po,
                    tol_density_profile=tol_density_profile,
                    check_min_period=check_min_period,
                    verbosity=verbosity_2,
                )

                # Check the fractional error in the masses for convergence
                M_0_discrep = (self.A1_M_layer[0] - M_0_fixed) / M_0_fixed
                M_1_discrep = (self.A1_M_layer[1] - M_1_fixed) / M_1_fixed

                # Print progress
                if verbosity >= 1:
                    print(
                        "Iter %d(%d),%d(%d): R0=%.4gR_E R1=%.4gR_E: tol=%.2g(%.2g), %.2g(%.2g)"
                        % (
                            i + 1,
                            num_attempt_1,
                            j + 1,
                            2 * num_attempt_2,
                            self.planet.A1_R_layer[0] / gv.R_earth,
                            self.planet.A1_R_layer[1] / gv.R_earth,
                            M_0_discrep,
                            tol_layer_masses,
                            M_1_discrep,
                            tol_layer_masses,
                        ),
                        end="\n",
                        flush=True,
                    )

                # End if M_1 and M_0 are satisfied
                if (
                    np.abs(M_0_discrep) < tol_layer_masses
                    and np.abs(M_1_discrep) < tol_layer_masses
                ):
                    if verbosity >= 1:
                        print("")
                    if verbosity >= 2:
                        self.print_info()

                    return

                # Break if M_1 is satisfied
                if np.abs(M_1_discrep) < tol_layer_masses:
                    break

            # Second iteration (fix M_0)
            for j in range(num_attempt_2):
                # Reduce dr as mass discrepancy decreases
                dr = (
                    (M_0_discrep - tol_layer_masses)
                    * 5
                    * dr_default
                    / 100
                    / tol_layer_masses
                )
                dr += dr_default
                dr = np.abs(dr)

                if self.A1_M_layer[0] > M_0_fixed:
                    self.planet.A1_R_layer[0] = self.planet.A1_R_layer[0] - dr
                else:
                    self.planet.A1_R_layer[0] = self.planet.A1_R_layer[0] + dr

                # Make the new spherical profiles
                self.planet.gen_prof_L2_find_M_given_R_R1(
                    M_max=1.2 * M_fixed, verbosity=0
                )

                self._spin_planet_simple(
                    R_max_eq,
                    R_max_po,
                    tol_density_profile=tol_density_profile,
                    check_min_period=check_min_period,
                    verbosity=verbosity_2,
                )

                # Check the fractional error in the masses for convergence
                M_0_discrep = (self.A1_M_layer[0] - M_0_fixed) / M_0_fixed
                M_1_discrep = (self.A1_M_layer[1] - M_1_fixed) / M_1_fixed

                # Print progress
                if verbosity >= 1:
                    print(
                        "Iter %d(%d),%d(%d): R0=%.3gR_E R1=%.3gR_E: tol=%.2g(%.2g), %.2g(%.2g)"
                        % (
                            i + 1,
                            num_attempt_1,
                            num_attempt_2 + j + 1,
                            2 * num_attempt_2,
                            self.planet.A1_R_layer[0] / gv.R_earth,
                            self.planet.A1_R_layer[1] / gv.R_earth,
                            M_0_discrep,
                            tol_layer_masses,
                            M_1_discrep,
                            tol_layer_masses,
                        ),
                        end="\n",
                        flush=True,
                    )

                # End if M_1 and M_0 are satisfied
                if (
                    np.abs(M_0_discrep) < tol_layer_masses
                    and np.abs(M_1_discrep) < tol_layer_masses
                ):
                    if verbosity >= 1:
                        print("")
                    if verbosity >= 2:
                        self.print_info()

                    return

                # Break if M_0 is satisfied
                if np.abs(M_0_discrep) < tol_layer_masses:
                    break

    def spin(
        self,
        R_max_eq=None,
        R_max_po=None,
        f_iter=None,
        fix_mass=True,
        check_min_period=False,
        tol_density_profile=0.001,
        tol_layer_masses=0.01,
        num_attempt_1=15,
        num_attempt_2=5,
        verbosity=1,
    ):
        """Create the spinning planet from the spherical one.

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
            allowed. Slow, set True only if required for extremely high spin.

        tol_density_profile : float
            The iterative search will end when the fractional differences
            between the density profiles in successive iterations is less than
            this tolerance.

        tol_layer_masses : float
            The iterative search will end when the fractional differences
            between the layer masses of the spinning planet and the spherical
            one are less than this tolerance.

        num_attempt_1 : int
            Maximum number of iterations allowed. Inner loop.

        num_attempt_2 : int
            Maximum number of iterations allowed. Outer loop.
        """
        # Initial values from the spherical planet
        self.A1_R_layer_original = self.planet.A1_R_layer
        self.R_original = self.planet.R
        self.period_input = self.period
        self.planet.num_prof = self.num_prof

        # Check for necessary input
        self._check_input()

        # Default maximum radii
        if R_max_eq is None:
            R_max_eq = 2 * self.planet.R
        if R_max_po is None:
            R_max_po = 1.2 * self.planet.R

        # Default f_iter
        if f_iter is None:
            f_iter = 0.005

        # f_iter should be < 1%
        if f_iter > 0.01:
            e = (
                "f_iter should be lower than 0.01.\n"
                "But I'm just a message, so feel free to come and mute me."
            )
            raise ValueError(e)

        if fix_mass:
            if self.planet.num_layer == 1:
                self._L1_spin_planet_fix_M_steps(
                    R_max_eq,
                    R_max_po,
                    f_iter,
                    check_min_period=check_min_period,
                    tol_layer_masses=tol_layer_masses,
                    tol_density_profile=tol_density_profile,
                    num_attempt=num_attempt_1,
                    verbosity=verbosity,
                )

            elif self.planet.num_layer == 2:
                self._L2_spin_planet_fix_M_steps(
                    R_max_eq,
                    R_max_po,
                    f_iter,
                    check_min_period=check_min_period,
                    tol_layer_masses=tol_layer_masses,
                    tol_density_profile=tol_density_profile,
                    num_attempt_1=num_attempt_1,
                    num_attempt_2=num_attempt_2,
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
                tol_density_profile=tol_density_profile,
                check_min_period=check_min_period,
                verbosity=verbosity,
            )

        self.update_attributes()


class ParticlePlanet:
    """Place particles to precisely match spinning or spherical body profiles.

    See also README.md and tutorial.ipynb.

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
        The [x, y, z] positions of each particle (m).

    A2_vel : [[float]]
        The [v_x, v_y, v_z] velocities of each particle (m s^-1).

    A1_m : [float]
        The mass of each particle (kg).

    A1_rho : [float]
        The density of each particle (kg m^-3).

    A1_u : [float]
        The specific internal energy of each particle (J kg^-1).

    A1_T : [float]
        The temperature of each particle (K).

    A1_P : [float]
        The pressure of each particle (Pa).

    A1_h : [float]
        The approximate smoothing length of each particle (m).

    A1_mat_id : [int]
        The material ID of each particle. (See the README.md documentation.)
    """

    def __init__(self, planet, N_particles, N_ngb=48, verbosity=1):
        self.N_particles = N_particles
        self.N_ngb = N_ngb

        assert isinstance(planet, Planet) or isinstance(planet, SpinPlanet)
        assert self.N_particles is not None

        utils.load_eos_tables(planet.A1_mat_layer)

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

            (
                self.A1_x,
                self.A1_y,
                self.A1_z,
                self.A1_vx,
                self.A1_vy,
                self.A1_vz,
                self.A1_m,
                self.A1_rho,
                self.A1_P,
                self.A1_T,
                self.A1_u,
                self.A1_h,
                self.A1_mat_id,
                self.A1_id,
            ) = us.place_particles(
                planet.A1_R,
                planet.A1_Z,
                planet.A1_rho,
                planet.A1_mat_id,
                planet.A1_P,
                planet.A1_T,
                planet.A1_u,
                N_particles,
                planet.period,
                N_ngb=self.N_ngb,
                verbosity=verbosity,
            )

            self.N_particles = self.A1_x.shape[0]

        # 2D position and velocity arrays
        self.A2_pos = np.transpose([self.A1_x, self.A1_y, self.A1_z])
        self.A2_vel = np.transpose([self.A1_vx, self.A1_vy, self.A1_vz])

    def set_material_entropies(self, A1_mat, A1_s_mat):
        """Set specific entropies for all particles of each material.

        Not applicable for all equations of state.

        Parameters
        ----------
        A1_mat : [str]
            The name of the material in each layer, from the central layer
            outwards. See Di_mat_id in `eos/eos.py`.

        A1_s_mat : [float]
            The specific entropy for particles of each material (J K^-1 kg^-1).

        Set attributes
        --------------
        A1_s : [float]
            The specific entropy of each particle (J K^-1 kg^-1).
        """
        A1_mat_id = [gv.Di_mat_id[mat] for mat in A1_mat]
        self.A1_s = np.zeros_like(self.A1_m)

        for mat_id, s in zip(A1_mat_id, A1_s_mat):
            self.A1_s[self.A1_mat_id == mat_id] = s

    def calculate_entropies(self):
        """
        Calculate the particles' specific entropies from their densities and
        temperatures.

        Not available for all equations of state.

        Currently requires all particles to be of materials that have entropy
        implemented.

        Set attributes
        --------------
        A1_s : [float]
            The specific entropy of each particle (J K^-1 kg^-1).
        """
        self.A1_s = eos.A1_s_rho_T(self.A1_rho, self.A1_T, self.A1_mat_id)

    def save(
        self,
        filename,
        boxsize=0,
        file_to_SI=utils.SI_to_SI,
        do_entropies=False,
        verbosity=1,
    ):
        """Save the particle configuration to an HDF5 file.

        Uses the same format as the SWIFT simulation code (www.swiftsim.com).

        Parameters
        ----------
        filename : str
            The data file path.

        boxsize : float (opt.)
            The simulation box side length (m). If provided, then the origin
            will be shifted to the centre of the box.

        file_to_SI : woma.Conversions (opt.)
            Simple unit conversion object from the file's units to SI. Defaults
            to staying in SI. See Conversions in misc/utils.py for more details.

        do_entropies : bool (opt.)
            If True then also save the particle specific entropies. See e.g.
            set_material_entropies() or similar to set the values first, or
            calculate_entropies() is called if not already set.
        """

        filename = utils.check_end(filename, ".hdf5")

        if verbosity >= 1:
            print('Saving "%s"...' % filename[-60:], end=" ", flush=True)

        if do_entropies:
            # Calculate the entropies if not already set
            if not hasattr(self, "A1_s"):
                self.calculate_entropies()
            A1_s = self.A1_s
        else:
            A1_s = None

        with h5py.File(filename, "w") as f:
            io.save_particle_data(
                f,
                self.A2_pos,
                self.A2_vel,
                self.A1_m,
                self.A1_h,
                self.A1_rho,
                self.A1_P,
                self.A1_u,
                self.A1_mat_id,
                A1_id=None,
                A1_s=A1_s,
                boxsize=boxsize,
                file_to_SI=file_to_SI,
                verbosity=verbosity,
            )
