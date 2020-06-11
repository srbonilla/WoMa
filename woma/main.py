"""
WoMa (World Maker)
====

Create models of rotating (and non-rotating) planets by solving the differential 
equations for hydrostatic equilibrium, and create initial conditions for 
smoothed particle hydrodynamics (SPH) or other particle-based simulations by 
placing particles to precisely match the planet's profiles.

Presented in Ruiz-Bonilla et al. (2020), MNRAS..., https://doi.org/...

Includes SEAGen (https://github.com/jkeger/seagen; Kegerreis et al. 2019, MNRAS 
487:4) with modifications for spinning planets.

Sergio Ruiz-Bonilla: sergio.ruiz-bonilla@durham.ac.uk  
Jacob Kegerreis: jacob.kegerreis@durham.ac.uk

Visit https://github.com/.../woma to download the code including examples and
for support.
"""

import numpy as np
import copy
import h5py
from scipy.interpolate import interp1d
from tqdm import tqdm
import seagen

from woma.spherical_funcs import L1_spherical, L2_spherical, L3_spherical
from woma.spin_funcs import L1_spin, L2_spin, L3_spin
import woma.spin_funcs.utils_spin as us
from woma.misc import glob_vars as gv
from woma.eos import eos
from woma.misc import utils
from woma.eos.T_rho import T_rho, set_T_rho_args, compute_A1_T_rho_id_and_args_from_type

# Output
Di_hdf5_planet_label = {
    "num_layer": "Number of Layers",
    "mat_layer": "Layer Materials",
    "mat_id_layer": "Layer Material IDs",
    "T_rho_type": "Layer T-rho Type",
    "T_rho_args": "Layer T-rho Args",
    "R_layer": "Layer Boundary Radii",
    "M_layer": "Mass in each Layer",
    "M": "Total Mass",
    "R": "Total Radius",
    "idx_layer": "Outer Index of each Layer",
    "P_s": "Surface Pressure",
    "T_s": "Surface Temperature",
    "rho_s": "Surface Density",
    "r": "Profile Radii",
    "m_enc": "Profile Enclosed Masses",
    "rho": "Profile Densities",
    "T": "Profile Temperatures",
    "u": "Profile Specific Internal Energies",
    "P": "Profile Pressures",
    "mat_id": "Profile Material IDs",
}


def get_planet_data(f, param):
    """ Load a planet attribute or array.

    Parameters
    ----------
    f : h5py File
        The opened hdf5 data file (with "r").

    param : str
        The array or attribute to get. See Di_hdf5_planet_label for details.

    Returns
    ----------
    data : np.ndarray
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

    Parameters
    ----------
    f : h5py File
        The opened hdf5 data file (with "r").

    A1_param : [str]
        List of the arrays or attributes to get. See Di_hdf5_planet_label for
        details.

    Returns
    ----------
    A1_data : [np.ndarray]
        The list of the arrays or attributes (std units).
    """
    A1_data = []
    # Load each requested array
    for param in A1_param:
        A1_data.append(get_planet_data(f, param))

    return A1_data


def load_planet(name, Fp_planet):
    """ Return a new Planet object loaded from a file.

    Parameters
    ----------
    name : str
        The name of the planet object.

    Fp_planet : str
        The file path.
        
    Returns
    -------
    p : Planet
        The loaded planet object.
    """
    p = Planet(name=name, Fp_planet=Fp_planet)

    Fp_planet = utils.check_end(p.Fp_planet, ".hdf5")

    print('Loading "%s"... ' % Fp_planet[-60:], end="")
    sys.stdout.flush()

    with h5py.File(Fp_planet, "r") as f:
        (
            p.num_layer,
            p.A1_mat_layer,
            p.A1_mat_id_layer,
            p.A1_T_rho_type,
            p.A1_T_rho_args,
            p.A1_R_layer,
            p.A1_M_layer,
            p.M,
            p.R,
            p.A1_idx_layer,
            p.P_s,
            p.T_s,
            p.rho_s,
            p.A1_r,
            p.A1_m_enc,
            p.A1_rho,
            p.A1_T,
            p.A1_P,
            p.A1_u,
            p.A1_mat_id,
        ) = multi_get_planet_data(
            f,
            [
                "num_layer",
                "mat_layer",
                "mat_id_layer",
                "T_rho_type",
                "T_rho_args",
                "R_layer",
                "M_layer",
                "M",
                "R",
                "idx_layer",
                "P_s",
                "T_s",
                "rho_s",
                "r",
                "m_enc",
                "rho",
                "T",
                "P",
                "u",
                "mat_id",
            ],
        )

    print("Done")

    p.update_attributes()
    p.print_info()

    return p


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

        'power=alpha': T = K * rho^alpha, K is set internally using each layer's 
                    outer temperature. Set alpha = 0 for isothermal.
        'adiabatic':  Adiabatic, constant s_adb is set internally, if applicable.

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

    P_0, P_1, ..., T_0, ..., rho_0, ... : float
        The pressure, temperature, and density at each layer boundary,
        from the centre (_0) up to the surface.

    I_MR2 : float
        The reduced moment of inertia (kg m^2).

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

    A1_mat_id_layer : [int]
        The ID of the material in each layer, from the central layer outwards.

    A1_idx_layer : [int]
        The profile index of each boundary, from the central layer outwards.

    A1_r : [float]
        The profile radii, in increasing order (m).

    A1_P : [float]
        The profile pressure with respect A1_r (Pa).
        
    A1_rho : [float]
        The profile density with respect A1_r (kg m^-3).
        
    A1_T : [float]
        The profile temperature with respect A1_r (K).
        
    A1_u : [float]
        The profile specific internal energy with respect A1_r (J kg^-1).
        
    A1_mat_id : [int]
        Material id for each layer (core, mantle, atmosphere). See glob_vars.py for more details.
        
    I_MR2 : float
        Moment of inertia (kg m^2).
    
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

        # P and T, or rho and T must be provided at the surface to calculate the
        # third. If all three are provided then rho is overwritten.
        if self.P_s is not None and self.T_s is not None:
            self.rho_s = eos.rho_P_T(self.P_s, self.T_s, self.A1_mat_id_layer[-1])
        elif self.rho_s is not None and self.T_s is not None:
            self.P_s = eos.P_T_rho(self.T_s, self.rho_s, self.A1_mat_id_layer[-1])

        # A1_T_rho_args, A1_T_rho_args
        if self.A1_T_rho_type is not None:
            (
                self.A1_T_rho_type_id,
                self.A1_T_rho_args,
            ) = compute_A1_T_rho_id_and_args_from_type(self.A1_T_rho_type)

        # Fp_planet, A1_R_layer, A1_M_layer
        if self.Fp_planet is None:
            self.Fp_planet = "data/%s.hdf5" % self.name
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

        ### default M_max and R_max?

        # Help info ###todo, maybe not necesary?
        if not True:
            if self.num_layer == 1:
                print("For a 1 layer planet, please specify:")
                print("pressure, temperature and density at the surface of the planet,")
                print(
                    "material, relation between temperature and density with any desired aditional parameters,"
                )
                print("for layer 1 of the planet.")
            elif self.num_layer == 2:
                print("For a 2 layer planet, please specify:")
                print("pressure, temperature and density at the surface of the planet,")
                print(
                    "materials, relations between temperature and density with any desired aditional parameters,"
                )
                print("for layer 1 and layer 2 of the planet.")
            elif self.num_layer == 3:
                print("For a 3 layer planet, please specify:")
                print("pressure, temperature and density at the surface of the planet,")
                print(
                    "materials, relations between temperature and density with any desired aditional parameters,"
                )
                print("for layer 1, layer 2, and layer 3 of the planet.")

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

        # Moment of inertia
        self.I_MR2 = utils.moi(self.A1_r, self.A1_rho)

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
        """ Print the Planet objects's main properties. """
        # Print and catch if any variables are None
        def print_try(string, variables):
            try:
                print(string % variables)
            except TypeError:
                print("    %s = None" % variables[0])

        space = 12
        print_try('Planet "%s": ', self.name)
        print_try(
            "    %s = %.5g kg = %.5g M_earth",
            (utils.add_whitespace("M", space), self.M, self.M / gv.M_earth),
        )
        print_try(
            "    %s = %.5g m = %.5g R_earth",
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
            "    %s = %s R_earth",
            (
                utils.add_whitespace("R_layer", space),
                utils.format_array_string(self.A1_R_layer / gv.R_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s M_earth",
            (
                utils.add_whitespace("M_layer", space),
                utils.format_array_string(self.A1_M_layer / gv.M_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s M_total",
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
        print_try("    %s = %.5g Pa", (utils.add_whitespace("P_s", space), self.P_s))
        print_try("    %s = %.5g K", (utils.add_whitespace("T_s", space), self.T_s))
        print_try(
            "    %s = %.5g kg/m^3", (utils.add_whitespace("rho_s", space), self.rho_s)
        )
        if self.num_layer > 2:
            print_try(
                "    %s = %.5g Pa", (utils.add_whitespace("P_2", space), self.P_2)
            )
            print_try("    %s = %.5g K", (utils.add_whitespace("T_2", space), self.T_2))
            print_try(
                "    %s = %.5g kg/m^3",
                (utils.add_whitespace("rho_2", space), self.rho_2),
            )
        if self.num_layer > 1:
            print_try(
                "    %s = %.5g Pa", (utils.add_whitespace("P_1", space), self.P_1)
            )
            print_try("    %s = %.5g K", (utils.add_whitespace("T_1", space), self.T_1))
            print_try(
                "    %s = %.5g kg/m^3",
                (utils.add_whitespace("rho_1", space), self.rho_1),
            )
        print_try("    %s = %.5g Pa", (utils.add_whitespace("P_0", space), self.P_0))
        print_try("    %s = %.5g K", (utils.add_whitespace("T_0", space), self.T_0))
        print_try(
            "    %s = %.5g kg/m^3", (utils.add_whitespace("rho_0", space), self.rho_0)
        )
        print_try(
            "    %s = %.5g M_tot*R_tot^2",
            (
                utils.add_whitespace("I_MR2", space),
                self.I_MR2 / self.M / self.R / self.R,
            ),
        )

    def print_declaration(self):
        """ Print the Planet objects formatted as a declaration. """
        space = 15
        print("%s = Planet(" % self.name)
        print('    %s = "%s",' % (utils.add_whitespace("name", space), self.name))
        print(
            '    %s = "%s",'
            % (utils.add_whitespace("Fp_planet", space), self.Fp_planet)
        )
        print(
            "    %s = %s,"
            % (
                utils.add_whitespace("A1_mat_layer", space),
                utils.format_array_string(self.A1_mat_layer, "string"),
            )
        )
        print(
            "    %s = %s,"
            % (
                utils.add_whitespace("A1_T_rho_type_id", space),
                utils.format_array_string(self.A1_T_rho_type_id, "%d"),
            )
        )
        print(
            "    %s = %s,"
            % (
                utils.add_whitespace("A1_T_rho_args", space),
                utils.format_array_string(self.A1_T_rho_args, "dorf"),
            )
        )
        print(
            "    %s = np.array(%s) * R_earth,"
            % (
                utils.add_whitespace("A1_R_layer", space),
                utils.format_array_string(self.A1_R_layer / gv.R_earth, "%.5g"),
            )
        )
        print(
            "    %s = %s,"
            % (
                utils.add_whitespace("A1_idx_layer", space),
                utils.format_array_string(self.A1_idx_layer, "%d"),
            )
        )
        print(
            "    %s = np.array(%s) * M_earth,"
            % (
                utils.add_whitespace("A1_M_layer", space),
                utils.format_array_string(self.A1_M_layer / gv.M_earth, "%.5g"),
            )
        )
        print(
            "    %s = %.5g * M_earth,"
            % (utils.add_whitespace("M", space), self.M / gv.M_earth)
        )
        print("    %s = %.5g," % (utils.add_whitespace("P_s", space), self.P_s))
        print("    %s = %.5g," % (utils.add_whitespace("T_s", space), self.T_s))
        print("    %s = %.5g," % (utils.add_whitespace("rho_s", space), self.rho_s))
        if self.num_layer > 2:
            print("    %s = %.5g," % (utils.add_whitespace("P_2", space), self.P_2))
            print("    %s = %.5g," % (utils.add_whitespace("T_2", space), self.T_2))
            print("    %s = %.5g," % (utils.add_whitespace("rho_2", space), self.rho_2))
        if self.num_layer > 1:
            print("    %s = %.5g," % (utils.add_whitespace("P_1", space), self.P_1))
            print("    %s = %.5g," % (utils.add_whitespace("T_1", space), self.T_1))
            print("    %s = %.5g," % (utils.add_whitespace("rho_1", space), self.rho_1))
        print("    %s = %.5g," % (utils.add_whitespace("P_0", space), self.P_0))
        print("    %s = %.5g," % (utils.add_whitespace("T_0", space), self.T_0))
        print("    %s = %.5g," % (utils.add_whitespace("rho_0", space), self.rho_0))
        print(
            "    %s = %.5g,"
            % (
                utils.add_whitespace("I_MR2", space),
                self.I_MR2 / (self.M * self.R ** 2),
            )
        )
        print("    )")

    def save_planet(self):
        Fp_planet = utils.check_end(self.Fp_planet, ".hdf5")

        print('Saving "%s"... ' % Fp_planet[-60:], end="")
        sys.stdout.flush()

        with h5py.File(Fp_planet, "w") as f:
            # Group
            grp = f.create_group("/planet")

            # Lists not numpy for attributes
            if type(self.A1_mat_layer).__module__ == np.__name__:
                self.A1_mat_layer = self.A1_mat_layer.tolist()

            # Attributes
            grp.attrs[Di_hdf5_planet_label["num_layer"]] = self.num_layer
            grp.attrs[Di_hdf5_planet_label["mat_layer"]] = self.A1_mat_layer
            grp.attrs[Di_hdf5_planet_label["mat_id_layer"]] = self.A1_mat_id_layer
            grp.attrs[Di_hdf5_planet_label["T_rho_type"]] = self.A1_T_rho_type
            grp.attrs[Di_hdf5_planet_label["T_rho_args"]] = self.A1_T_rho_args
            grp.attrs[Di_hdf5_planet_label["R_layer"]] = self.A1_R_layer
            grp.attrs[Di_hdf5_planet_label["M_layer"]] = self.A1_M_layer
            grp.attrs[Di_hdf5_planet_label["M"]] = self.M
            grp.attrs[Di_hdf5_planet_label["R"]] = self.R
            grp.attrs[Di_hdf5_planet_label["idx_layer"]] = self.A1_idx_layer
            grp.attrs[Di_hdf5_planet_label["P_s"]] = self.P_s
            grp.attrs[Di_hdf5_planet_label["T_s"]] = self.T_s
            grp.attrs[Di_hdf5_planet_label["rho_s"]] = self.rho_s

            # Arrays
            grp.create_dataset(Di_hdf5_planet_label["r"], data=self.A1_r, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["m_enc"], data=self.A1_m_enc, dtype="d"
            )
            grp.create_dataset(Di_hdf5_planet_label["rho"], data=self.A1_rho, dtype="d")
            grp.create_dataset(Di_hdf5_planet_label["T"], data=self.A1_T, dtype="d")
            grp.create_dataset(Di_hdf5_planet_label["P"], data=self.A1_P, dtype="d")
            grp.create_dataset(Di_hdf5_planet_label["u"], data=self.A1_u, dtype="d")
            grp.create_dataset(
                Di_hdf5_planet_label["mat_id"], data=self.A1_mat_id, dtype="i"
            )

        print("Done")

    def load_planet_profiles(self):
        """ Load the profiles arrays for an existing Planet object from a file. """
        Fp_planet = utils.check_end(self.Fp_planet, ".hdf5")

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
            ) = multi_get_planet_data(f, ["r", "m_enc", "rho", "T", "P", "u", "mat_id"])

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

    def gen_prof_L1_find_R_given_M(self):
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
        )
        self.A1_R_layer[-1] = self.R

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
        self.print_info()

    def gen_prof_L1_find_M_given_R(self, print_info=True):
        # Check for necessary input
        assert self.R is not None or self.A1_R_layer[0] is not None
        assert self.M_max is not None
        assert len(self.A1_R_layer) == 1
        self._1_layer_input()
        if self.R is None:
            self.R = self.A1_R_layer[0]

        if print_info:
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

        if print_info:
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
        if print_info:
            self.print_info()

    def gen_prof_L1_given_R_M(self):
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

    def gen_prof_L2_find_R1_given_R_M(self):
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
        )

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

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L2_find_M_given_R1_R(self, print_info=True):
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        assert self.M_max is not None
        self._2_layer_input()

        if print_info:
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

        if print_info:
            print("Done!")

        self.update_attributes()
        if print_info:
            self.print_info()

    def gen_prof_L2_find_R_given_M_R1(self):
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
        )
        self.A1_R_layer[-1] = self.R

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
        self.print_info()

    def gen_prof_L2_find_R1_R_given_M1_M2(self):

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
        )
        self.A1_R_layer[-1] = self.R

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
        self.print_info()

    def gen_prof_L2_given_R_M_R1(self):
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
        self.print_info()

    def gen_prof_L2_given_prof_L1(self, mat, T_rho_type_id, T_rho_args, rho_min):
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
        self.print_info()

    def gen_prof_L2_find_R1_given_M1_add_L2(self):
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
        )
        self.A1_R_layer[-1] = self.R

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
        print("Adding the second layer on top...")

        self.gen_prof_L2_given_prof_L1(
            mat=mat_L2,
            T_rho_type_id=T_rho_type_L2,
            T_rho_args=T_rho_args_L2,
            rho_min=self.rho_min,
        )

        print("Done!")

    def gen_prof_L2_find_M1_given_R1_add_L2(self):
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
        print("Adding the second layer on top...")

        self.gen_prof_L2_given_prof_L1(
            mat=mat_L2,
            T_rho_type_id=T_rho_type_L2,
            T_rho_args=T_rho_args_L2,
            rho_min=self.rho_min,
        )

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

    def gen_prof_L3_find_R1_R2_given_R_M_I(self):
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
        )

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

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_find_R_R1_R2_given_M_M1_M2(self):
        ###todo
        return None

    def gen_prof_L3_find_R2_given_R_M_R1(self):
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
        )

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

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_find_R1_given_R_M_R2(self):
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
        )

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

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_find_M_given_R_R1_R2(self):
        # Check for necessary input
        assert self.R is not None
        assert self.A1_R_layer[0] is not None
        assert self.A1_R_layer[1] is not None
        assert self.M_max is not None
        self._3_layer_input()

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

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_find_R_given_M_R1_R2(self):
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
        )
        self.A1_R_layer[-1] = self.R

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

        print("Done!")

        self.update_attributes()
        self.print_info()

    def gen_prof_L3_given_R_M_R1_R2(self):
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
        self.print_info()

    def gen_prof_L3_given_prof_L2(
        self, mat=None, T_rho_type_id=None, T_rho_args=None, rho_min=None
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
        self.print_info()

    def gen_prof_L3_find_R1_R2_given_M1_M2_add_L3(
        self, M1=None, M2=None, R_min=None, R_max=None, M_frac_tol=None, rho_min=None
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
        )
        self.A1_R_layer[-1] = self.R

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
        print("Adding the third layer on top...")

        self.gen_prof_L3_given_prof_L2(
            mat=mat_L3,
            T_rho_type_id=T_rho_type_L3,
            T_rho_args=T_rho_args_L3,
            rho_min=self.rho_min,
        )

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
        Tw=None,
        num_prof=1000,
        R_e_max=None,
        R_p_max=None,
    ):
        self.name = name
        self.num_prof = num_prof
        self.R_e_max = R_e_max
        self.R_p_max = R_p_max
        self.Tw = Tw
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
            self.Tw,
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
        """ Print the Planet objects's main properties. """
        # Print and catch if any variables are None
        def print_try(string, variables):
            try:
                print(string % variables)
            except TypeError:
                print("    %s = None" % variables[0])

        space = 12
        print_try('Planet "%s": ', self.name)
        print_try(
            "    %s = %.5g kg = %.5g M_earth",
            (utils.add_whitespace("M", space), self.M, self.M / gv.M_earth),
        )
        print_try(
            "    %s = %.5g m = %.5g R_earth",
            (utils.add_whitespace("R_equator", space), self.R_e, self.R_e / gv.R_earth),
        )
        print_try(
            "    %s = %.5g m = %.5g R_earth",
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
            "    %s = %s R_earth",
            (
                utils.add_whitespace("R_layer_eq", space),
                utils.format_array_string(self.A1_R_layer_equator / gv.R_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s R_earth",
            (
                utils.add_whitespace("R_layer_pole", space),
                utils.format_array_string(self.A1_R_layer_pole / gv.R_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s M_earth",
            (
                utils.add_whitespace("M_layer", space),
                utils.format_array_string(self.A1_M_layer / gv.M_earth, "%.5g"),
            ),
        )
        print_try(
            "    %s = %s M_total",
            (
                utils.add_whitespace("M_frac_layer", space),
                utils.format_array_string(self.A1_M_layer / self.M, "%.5g"),
            ),
        )
        print_try("    %s = %.5g Pa", (utils.add_whitespace("P_s", space), self.P_s))
        print_try("    %s = %.5g K", (utils.add_whitespace("T_s", space), self.T_s))
        print_try(
            "    %s = %.5g kg/m^3", (utils.add_whitespace("rho_s", space), self.rho_s)
        )
        if self.num_layer > 2:
            print_try(
                "    %s = %.5g Pa", (utils.add_whitespace("P_2", space), self.P_2)
            )
            print_try("    %s = %.5g K", (utils.add_whitespace("T_2", space), self.T_2))
            print_try(
                "    %s = %.5g kg/m^3",
                (utils.add_whitespace("rho_2", space), self.rho_2),
            )
        if self.num_layer > 1:
            print_try(
                "    %s = %.5g Pa", (utils.add_whitespace("P_1", space), self.P_1)
            )
            print_try("    %s = %.5g K", (utils.add_whitespace("T_1", space), self.T_1))
            print_try(
                "    %s = %.5g kg/m^3",
                (utils.add_whitespace("rho_1", space), self.rho_1),
            )
        print_try("    %s = %.5g Pa", (utils.add_whitespace("P_0", space), self.P_0))
        print_try("    %s = %.5g K", (utils.add_whitespace("T_0", space), self.T_0))
        print_try(
            "    %s = %.5g kg/m^3", (utils.add_whitespace("rho_0", space), self.rho_0)
        )

    def find_Tw_min(self, Tw_max=10, max_iter=20, print_info=True):

        Tw_min = us.find_Tw_min(
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
            Tw_max,
            max_iter,
            print_info,
        )

        self.Tw_min = Tw_min

    def spin(
        self,
        max_iter_1=20,
        max_iter_2=20,
        print_info=True,
        tol=0.001,
        check_Tw_min=True,
    ):
        # Check for necessary input
        self._check_input()

        for i in tqdm(
            range(max_iter_1), desc="Computing spinning profile", disable=not print_info
        ):
            # compute Tw_min
            if check_Tw_min:
                self.find_Tw_min(print_info=False, max_iter=max_iter_2)

                # select Tw for this iteration
                if self.Tw >= self.Tw_min:
                    Tw_iter = self.Tw
                else:
                    Tw_iter = self.Tw_min
            else:
                Tw_iter = self.Tw

            # compute profile
            A1_rho_equator, A1_rho_pole = us.spin_iteration(
                Tw_iter,
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

            # convergence criteria
            criteria = np.mean(
                np.abs(A1_rho_equator - self.A1_rho_equator) / self.rho_s
            )

            # save results
            self.A1_rho_equator = A1_rho_equator
            self.A1_rho_pole = A1_rho_pole

            # check if there is convergence
            # print(criteria)
            if criteria < tol:
                if print_info:
                    print("Convergence criteria reached.")
                break

        if self.Tw < Tw_iter:
            if print_info:
                print("")
                print("Minimum period found at", Tw_iter, "h")
            self.Tw = Tw_iter

        self.update_attributes()

        if print_info:
            self.print_info()


def L1_spin_planet_fix_M(
    planet,
    Tw,
    R_e_max=4 * gv.R_earth,
    R_p_max=2 * gv.R_earth,
    num_prof=1000,
    max_iter_1=20,
):

    assert isinstance(planet, Planet)
    assert planet.num_layer == 1

    tol = 0.001
    f_min = 0.0
    f_max = 1.0

    for i in tqdm(range(max_iter_1), desc="Computing spinning profile", disable=False):

        f = np.mean([f_min, f_max])

        # create copy of planet
        new_planet = copy.deepcopy(planet)

        # shrink it
        new_planet.A1_R_layer = f * new_planet.A1_R_layer
        new_planet.R = f * new_planet.R

        # make new profile
        new_planet.M_max = new_planet.M
        new_planet.gen_prof_L1_find_M_given_R(print_info=False)

        spin_planet = SpinPlanet(
            planet=new_planet, Tw=Tw, R_e_max=R_e_max, R_p_max=R_p_max
        )

        spin_planet.spin(print_info=False)

        criteria = np.abs(planet.M - spin_planet.M) / planet.M < tol
        # print(np.abs(planet.M - spin_planet.M)/planet.M)

        if criteria:
            break

        if spin_planet.M > planet.M:
            f_max = f
        else:
            f_min = f

    spin_planet.print_info()

    return spin_planet


def L2_spin_planet_fix_M(
    planet,
    Tw,
    R_e_max=4 * gv.R_earth,
    R_p_max=2 * gv.R_earth,
    num_prof=1000,
    max_iter_1=20,
    max_iter_2=5,
):

    assert isinstance(planet, Planet)
    assert planet.num_layer == 2

    tol = 0.01
    M = planet.M

    f_M_core = planet.A1_M_layer[0] / M

    new_planet = copy.deepcopy(planet)

    spin_planet = SpinPlanet(
        planet=new_planet, Tw=Tw, num_prof=num_prof, R_e_max=R_e_max, R_p_max=R_p_max
    )

    spin_planet.spin(print_info=False)

    for k in tqdm(range(max_iter_2), desc="Computing spinning profile"):

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
            new_planet.M_max = 1.05 * planet.M
            new_planet.gen_prof_L2_find_M_given_R1_R(print_info=False)

            spin_planet = SpinPlanet(
                planet=new_planet,
                Tw=Tw,
                num_prof=num_prof,
                R_e_max=R_e_max,
                R_p_max=R_p_max,
            )

            spin_planet.spin(print_info=False)

            criteria_1 = np.abs(planet.M - spin_planet.M) / planet.M < tol
            criteria_2 = (
                np.abs(
                    planet.A1_M_layer[0] / planet.M
                    - spin_planet.A1_M_layer[0] / spin_planet.M
                )
                < tol
            )

            if criteria_1 and criteria_2:
                return spin_planet

            if criteria_1:
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
            new_planet.M_max = 1.05 * planet.M
            new_planet.gen_prof_L2_find_M_given_R1_R(print_info=False)

            spin_planet = SpinPlanet(
                planet=new_planet,
                Tw=Tw,
                num_prof=num_prof,
                R_e_max=R_e_max,
                R_p_max=R_p_max,
            )

            spin_planet.spin(print_info=False)

            criteria_1 = np.abs(planet.M - spin_planet.M) / planet.M < tol
            criteria_2 = (
                np.abs(
                    planet.A1_M_layer[0] / planet.M
                    - spin_planet.A1_M_layer[0] / spin_planet.M
                )
                < tol
            )

            if criteria_1 and criteria_2:
                return spin_planet

            if criteria_2:
                break

            new_f_M_core = spin_planet.A1_M_layer[0] / spin_planet.M

            if new_f_M_core > f_M_core:
                R_core_max = R_core
            else:
                R_core_min = R_core

    spin_planet.print_info()

    return spin_planet


def spin_planet_fix_M(
    planet,
    Tw,
    R_e_max=4 * gv.R_earth,
    R_p_max=2 * gv.R_earth,
    num_prof=1000,
    max_iter_1=20,
    max_iter_2=5,
):

    if planet.num_layer == 1:

        spin_planet = L1_spin_planet_fix_M(
            planet, Tw, R_e_max, R_p_max, num_prof, max_iter_1
        )

    elif planet.num_layer == 2:

        spin_planet = L2_spin_planet_fix_M(
            planet, Tw, R_e_max, R_p_max, num_prof, max_iter_1, max_iter_2
        )

    elif planet.num_layer == 3:

        raise ValueError("Not implemented yet")

    else:

        raise ValueError(
            "Number of layers in your planet is incorrect. \
                         It should be 1, 2, or 3"
        )

    return spin_planet


class ParticleSet:
    """ Particle generator class.

    Parameters
    ----------
    planet : instance of Planet or SpinPlanet
        The opened hdf5 data file (with "r").

    N_particles : str
        List of the arrays or attributes to get. See Di_hdf5_planet_label for
        details.
        
    N_neig : int
        Number of nearest neighbours used to compute SPH density

    Attributes (in addition to the input parameters)
    ----------
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
        
    A1_P : [float]
        Array of pressures for all particles (Pa).
    
    A1_h : [float]
        Array of smoothing lengths for all particles (m).
        
    A1_mat_id : [int]
        Array of material ids for all particles. See glob_vars.py
        
    A1_id : [int]
        Array of ids for all particles.
        
    """

    def __init__(
        self, planet=None, N_particles=None, N_neig=48,
    ):
        self.N_particles = N_particles
        self.N_neig = N_neig

        assert isinstance(planet, Planet) or isinstance(planet, SpinPlanet)
        assert self.N_particles is not None

        if isinstance(planet, Planet):
            particles = seagen.GenSphere(
                self.N_particles, planet.A1_r[1:], planet.A1_rho[1:], verb=0
            )

            self.A1_x = particles.A1_x
            self.A1_y = particles.A1_y
            self.A1_z = particles.A1_z
            self.A1_vx = np.zeros_like(particles.A1_x)
            self.A1_vy = np.zeros_like(particles.A1_x)
            self.A1_vz = np.zeros_like(particles.A1_x)
            self.A1_m = particles.A1_m
            self.A1_rho = particles.A1_rho

            # need to complete these
            # self.A1_u = u
            # self.A1_P = P
            # self.A1_h = h
            # self.A1_mat_id = mat_id
            # self.A1_id = picle_id

            self.N_particles = particles.A1_x.shape[0]

        if isinstance(planet, SpinPlanet):
            if self.num_layer == 1:

                (
                    x,
                    y,
                    z,
                    vx,
                    vy,
                    vz,
                    m,
                    rho,
                    u,
                    P,
                    h,
                    mat_id,
                    picle_id,
                ) = L1_spin.picle_placement_L1(
                    planet.A1_r_equator,
                    planet.A1_rho_equator,
                    planet.A1_r_pole,
                    planet.A1_rho_pole,
                    planet.Tw,
                    self.N_particles,
                    planet.A1_mat_id_layer[0],
                    planet.A1_T_rho_type_id[0],
                    planet.A1_T_rho_args[0],
                    self.N_neig,
                )

                self.A1_x = x
                self.A1_y = y
                self.A1_z = z
                self.A1_vx = vx
                self.A1_vy = vy
                self.A1_vz = vz
                self.A1_m = m
                self.A1_rho = rho
                self.A1_u = u
                self.A1_P = P
                self.A1_h = h
                self.A1_mat_id = mat_id
                self.A1_id = picle_id
                self.N_particles = x.shape[0]

            elif self.num_layer == 2:

                rho_P_model = interp1d(planet.A1_P, planet.A1_rho)
                rho_1 = rho_P_model(planet.P_1)

                (
                    x,
                    y,
                    z,
                    vx,
                    vy,
                    vz,
                    m,
                    rho,
                    u,
                    P,
                    h,
                    mat_id,
                    picle_id,
                ) = L2_spin.picle_placement_L2(
                    planet.A1_r_equator,
                    planet.A1_rho_equator,
                    planet.A1_r_pole,
                    planet.A1_rho_pole,
                    planet.Tw,
                    self.N_particles,
                    rho_1,
                    planet.A1_mat_id_layer[0],
                    planet.A1_T_rho_type_id[0],
                    planet.A1_T_rho_args[0],
                    planet.A1_mat_id_layer[1],
                    planet.A1_T_rho_type_id[1],
                    planet.A1_T_rho_args[1],
                    self.N_neig,
                )

                self.A1_x = x
                self.A1_y = y
                self.A1_z = z
                self.A1_vx = vx
                self.A1_vy = vy
                self.A1_vz = vz
                self.A1_m = m
                self.A1_rho = rho
                self.A1_u = u
                self.A1_P = P
                self.A1_h = h
                self.A1_mat_id = mat_id
                self.A1_id = picle_id
                self.N_particles = x.shape[0]

            elif self.num_layer == 3:

                rho_P_model = interp1d(planet.A1_P, planet.A1_rho)
                rho_1 = rho_P_model(planet.P_1)
                rho_2 = rho_P_model(planet.P_2)

                (
                    x,
                    y,
                    z,
                    vx,
                    vy,
                    vz,
                    m,
                    rho,
                    u,
                    P,
                    h,
                    mat_id,
                    picle_id,
                ) = L3_spin.picle_placement_L3(
                    planet.A1_r_equator,
                    planet.A1_rho_equator,
                    planet.A1_r_pole,
                    planet.A1_rho_pole,
                    planet.Tw,
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
                    self.N_neig,
                )

                self.A1_x = x
                self.A1_y = y
                self.A1_z = z
                self.A1_vx = vx
                self.A1_vy = vy
                self.A1_vz = vz
                self.A1_m = m
                self.A1_rho = rho
                self.A1_u = u
                self.A1_P = P
                self.A1_h = h
                self.A1_mat_id = mat_id
                self.A1_id = picle_id
                self.N_particles = x.shape[0]
