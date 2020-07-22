"""
WoMa IO utilities

The spherical and spinning planets' data and attributes can be conveniently 
saved and loaded using HDF5 files (see www.hdfgroup.org and www.h5py.org).
See .save(), .load(), and load_file in the Planet , SpinPlanet and
ParticlePlanet classes.
"""

import numpy as np
import sys

from woma.misc import glob_vars as gv


# HDF5 labels
Di_hdf5_planet_label = {
    # Attributes
    "num_layer": "Number of Layers",
    "mat_layer": "Layer Materials",
    "mat_id_layer": "Layer Material IDs",
    "T_rho_type": "Layer T-rho Type",
    "T_rho_type_id": "Layer T-rho Type ID",
    "T_rho_args": "Layer T-rho Internal Arguments",
    "R_layer": "Layer Boundary Radii",
    "M_layer": "Mass in each Layer",
    "M": "Total Mass",
    "R": "Total Radius",
    "idx_layer": "Outer Index of each Layer",
    "P_s": "Surface Pressure",
    "T_s": "Surface Temperature",
    "rho_s": "Surface Density",
    # Profiles
    "r": "Profile Radii",
    "m_enc": "Profile Enclosed Masses",
    "rho": "Profile Densities",
    "T": "Profile Temperatures",
    "u": "Profile Specific Internal Energies",
    "P": "Profile Pressures",
    "mat_id": "Profile Material IDs",
}
Di_hdf5_spin_label = {
    # Attributes
    "num_layer": "Number of Layers",
    "mat_layer": "Layer Materials",
    "mat_id_layer": "Layer Material IDs",
    "T_rho_type": "Layer T-rho Type",
    "T_rho_type_id": "Layer T-rho Type ID",
    "T_rho_args": "Layer T-rho Internal Arguments",
    "R_layer": "Layer Boundary Equatorial Radii",
    "Z_layer": "Layer Boundary Polar Radii",
    "M_layer": "Mass in each Layer",
    "M": "Total Mass",
    "R_eq": "Total Equatorial Radius",
    "R_po": "Total Polar Radius",
    "idx_layer": "Outer Spheroid Index of each Layer",
    "period": "Spin Period",
    "P_s": "Surface Pressure",
    "T_s": "Surface Temperature",
    "rho_s": "Surface Density",
    # Profiles
    "r_eq": "Equatorial radii profile",
    "r_po": "Polar radii profile",
    "rho_eq": "Equatorial density profile",
    "rho_po": "Polar density profile",
    "R": "Spheroid Equatorial Radii",
    "Z": "Spheroid Polar Radii",
    "m": "Spheroid Masses",
    "rho": "Spheroid Densities",
    "T": "Spheroid Temperatures",
    "u": "Spheroid Specific Internal Energies",
    "P": "Spheroid Pressures",
    "mat_id": "Spheroid Material IDs",
}

Di_hdf5_picle_label = {  # Type
    "pos": "Coordinates",  # d
    "vel": "Velocities",  # f
    "m": "Masses",  # f
    "h": "SmoothingLengths",  # f
    "u": "InternalEnergies",  # f
    "rho": "Densities",  # f
    "P": "Pressures",  # f
    "s": "Entropies",  # f
    "id": "ParticleIDs",  # L
    "mat_id": "MaterialIDs",  # i
    "phi": "Potentials",  # f
    "T": "Temperatures",  # f
}


def get_planet_data(f, param):
    """ Load an attribute or profile from an HDF5 file. 
    
    See woma.Planet.save().
    
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
    """ Load multiple attributes and/or profiles from an HDF5 file. 
    
    See woma.Planet.save().

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


def get_spin_planet_data(f, param):
    """ Load an attribute or profile from an HDF5 file. 
    
    See woma.SpinPlanet.save().
    
    Parameters
    ----------
    f : h5py File
        The opened hdf5 data file (with "r").

    param : str
        The array or attribute to get. See Di_hdf5_spin_label for details.

    Returns
    ----------
    data : np.ndarray
        The array or attribute (std units).
    """
    # Attributes
    try:
        return f["spin_planet"].attrs[Di_hdf5_spin_label[param]]
    # Datasets
    except KeyError:
        return f["spin_planet/" + Di_hdf5_spin_label[param]][()]


def multi_get_spin_planet_data(f, A1_param):
    """ Load multiple attributes and/or profiles from an HDF5 file. 
    
    See woma.SpinPlanet.save().

    Parameters
    ----------
    f : h5py File
        The opened hdf5 data file (with "r").

    A1_param : [str]
        List of the arrays or attributes to get. See Di_hdf5_spin_label for
        details.

    Returns
    ----------
    A1_data : [np.ndarray]
        The list of the arrays or attributes (std units).
    """
    A1_data = []
    # Load each requested array
    for param in A1_param:
        A1_data.append(get_spin_planet_data(f, param))

    return A1_data


class Conversions:
    """ Class to store conversions from one set of units to another, derived
       using the base mass-, length-, and time-unit relations.

       Usage e.g.:
           cgs_to_SI   = Conversions(1e-3, 1e-2, 1)
           SI_to_cgs   = cgs_to_SI.inv()

           rho_SI  = rho_cgs * cgs_to_SI.rho
           G_cgs   = 6.67e-11 * SI_to_cgs.G

       Args:
           m (float)
               Value to convert mass from the first units to the second.

           l (float)
               Value to convert length from the first units to the second.

           t (float)
               Value to convert time from the first units to the second.

       Attrs: (all floats)
           m           Mass
           l           Length
           t           Time
           v           Velocity
           a           Acceleration
           rho         Density
           drho_dt     Rate of change of density
           P           Pressure
           u           Specific energy
           du_dt       Rate of change of specific energy
           E           Energy
           s           Specific entropy
           G           Gravitational constant
   """

    def __init__(self, m, l, t):
        # Input conversions
        self.m = m
        self.l = l
        self.t = t
        # Derived conversions
        self.v = l * t ** -1
        self.a = l * t ** -2
        self.rho = m * l ** -3
        self.drho_dt = m * l ** -4
        self.P = m * l ** -1 * t ** -2
        self.u = l ** 2 * t ** -2
        self.du_dt = l ** 2 * t ** -3
        self.E = m * l ** 2 * t ** -2
        self.s = l ** 2 * t ** -2
        self.G = m ** -1 * l ** 3 * t ** -2

    def inv(self):
        """ Return the inverse to this conversion """
        return Conversions(1 / self.m, 1 / self.l, 1 / self.t)


SI_to_SI = Conversions(1, 1, 1)  # No-op
swift_to_SI = Conversions(
    gv.M_earth, gv.R_earth, 1
)  # normal units for planetary impacts
cgs_to_SI = Conversions(1e-3, 1e-2, 1)
SI_to_cgs = cgs_to_SI.inv()


def save_picle_data(
    f,
    A2_pos,
    A2_vel,
    A1_m,
    A1_h,
    A1_rho,
    A1_P,
    A1_u,
    A1_id,
    A1_mat_id,
    boxsize,
    file_to_SI,
    verbosity=1,
):
    """ Print checks and save particle data to an hdf5 file.

        Args:
            f (h5py File)
                The opened hdf5 data file (with 'w').

            A2_pos, A2_vel, A1_m, A1_h, A1_rho, A1_P, A1_u, A1_id, A1_mat_id:
                The particle data arrays (std units). See Di_hdf5_label for
                details.

            boxsize (float)
                The simulation box side length (std units).

            file_to_SI (Conversions)
                Unit conversion object from the file's units to SI.
    """
    num_picle = len(A1_id)

    SI_to_file = file_to_SI.inv()

    # Print info to double check
    if verbosity >= 1:
        print("")
        print("num_picle    = %d" % num_picle)
        print("boxsize      = %.2g R_E" % (boxsize / gv.R_earth))
        print("mat_id       = ", end="")
        for mat_id in np.unique(A1_mat_id):
            print("%d " % mat_id, end="")
        print("\n")
        print("Unit mass    = %.5e g" % (file_to_SI.m * SI_to_cgs.m))
        print("Unit length  = %.5e cm" % (file_to_SI.l * SI_to_cgs.l))
        print("Unit time    = %.5e s" % file_to_SI.t)
        print("")
        print("Min, max values (file units):")
        print(
            "  pos = [%.5g, %.5g,    %.5g, %.5g,    %.5g, %.5g]"
            % (
                np.amin(A2_pos[:, 0]),
                np.amax(A2_pos[:, 0]),
                np.amin(A2_pos[:, 1]),
                np.amax(A2_pos[:, 1]),
                np.amin(A2_pos[:, 2]),
                np.amax(A2_pos[:, 2]),
            )
        )
        if np.amax(abs((A2_pos + boxsize / 2.0) * SI_to_file.l)) > boxsize:
            print("# Particles are outside the box!")
            sys.exit()
        print(
            "  vel = [%.5g, %.5g,    %.5g, %.5g,    %.5g, %.5g]"
            % (
                np.amin(A2_vel[:, 0]),
                np.amax(A2_vel[:, 0]),
                np.amin(A2_vel[:, 1]),
                np.amax(A2_vel[:, 1]),
                np.amin(A2_vel[:, 2]),
                np.amax(A2_vel[:, 2]),
            )
        )
        for name, array in zip(
            ["m", "rho", "P", "u", "h"], [A1_m, A1_rho, A1_P, A1_u, A1_h]
        ):
            print("  %s = %.5g, %.5g" % (name, np.amin(array), np.amax(array)))
        print("")

    # Save
    # Header
    grp = f.create_group("/Header")
    grp.attrs["BoxSize"] = [boxsize] * 3
    grp.attrs["NumPart_Total"] = [num_picle, 0, 0, 0, 0, 0]
    grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
    grp.attrs["NumPart_ThisFile"] = [num_picle, 0, 0, 0, 0, 0]
    grp.attrs["Time"] = 0.0
    grp.attrs["NumFilesPerSnapshot"] = 1
    grp.attrs["MassTable"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    grp.attrs["Flag_Entropy_ICs"] = 0
    grp.attrs["Dimension"] = 3

    # Runtime parameters
    grp = f.create_group("/RuntimePars")
    grp.attrs["PeriodicBoundariesOn"] = 0

    # Units
    grp = f.create_group("/Units")
    grp.attrs["Unit mass in cgs (U_M)"] = file_to_SI.m * SI_to_cgs.m
    grp.attrs["Unit length in cgs (U_L)"] = file_to_SI.l * SI_to_cgs.l
    grp.attrs["Unit time in cgs (U_t)"] = file_to_SI.t
    grp.attrs["Unit current in cgs (U_I)"] = 1.0
    grp.attrs["Unit temperature in cgs (U_T)"] = 1.0

    # Particles
    grp = f.create_group("/PartType0")
    grp.create_dataset(
        Di_hdf5_picle_label["pos"],
        data=(A2_pos + boxsize / 2.0) * SI_to_file.l,
        dtype="d",
    )
    grp.create_dataset(
        Di_hdf5_picle_label["vel"], data=A2_vel * SI_to_file.v, dtype="f"
    )
    grp.create_dataset(Di_hdf5_picle_label["m"], data=A1_m * SI_to_file.m, dtype="f")
    grp.create_dataset(Di_hdf5_picle_label["h"], data=A1_h * SI_to_file.l, dtype="f")
    grp.create_dataset(
        Di_hdf5_picle_label["rho"], data=A1_rho * SI_to_file.rho, dtype="f"
    )
    grp.create_dataset(Di_hdf5_picle_label["P"], data=A1_P * SI_to_file.P, dtype="f")
    grp.create_dataset(Di_hdf5_picle_label["u"], data=A1_u * SI_to_file.u, dtype="f")
    grp.create_dataset(Di_hdf5_picle_label["id"], data=A1_id, dtype="L")
    grp.create_dataset(Di_hdf5_picle_label["mat_id"], data=A1_mat_id, dtype="i")

    print('Saved "%s"' % f.filename[-64:])
