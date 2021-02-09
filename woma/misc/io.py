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
from woma.misc.utils import SI_to_SI, SI_to_cgs


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
Di_hdf5_particle_label = {  # Type
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
    """Load an attribute or profile from an HDF5 file.

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
    """Load multiple attributes and/or profiles from an HDF5 file.

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
    """Load an attribute or profile from an HDF5 file.

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
    """Load multiple attributes and/or profiles from an HDF5 file.

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


def save_particle_data(
    f,
    A2_pos,
    A2_vel,
    A1_m,
    A1_h,
    A1_rho,
    A1_P,
    A1_u,
    A1_mat_id,
    A1_id=None,
    A1_s=None,
    boxsize=0,
    file_to_SI=SI_to_SI,
    verbosity=1,
):
    """Save particle data to an hdf5 file.

    Uses the same format as the SWIFT simulation code (www.swiftsim.com).

    Parameters
    ----------
    f : h5py File
        The opened hdf5 data file (with "w").

    A2_pos, A2_vel, A1_m, A1_h, A1_rho, A1_P, A1_u, A1_mat_id
        : [float] or [int]
        The particle data arrays. See Di_hdf5_particle_label for details.

    A1_id : [int] (opt.)
        The particle IDs. Defaults to the order in which they're provided.

    A1_s : [float] (opt.)
        The particle specific entropies.

    boxsize : float (opt.)
        The simulation box side length (m). If provided, then the origin will be
        shifted to the centre of the box.

    file_to_SI : woma.Conversions (opt.)
        Simple unit conversion object from the file's units to SI. Defaults to
        staying in SI. See Conversions in misc/utils.py for more details.
    """
    num_particle = len(A1_m)
    if A1_id is None:
        A1_id = np.arange(num_particle)

    # Convert to file units
    SI_to_file = file_to_SI.inv()
    boxsize *= SI_to_file.l
    A2_pos *= SI_to_file.l
    A2_vel *= SI_to_file.v
    A1_m *= SI_to_file.m
    A1_h *= SI_to_file.l
    A1_rho *= SI_to_file.rho
    A1_P *= SI_to_file.P
    A1_u *= SI_to_file.u
    if A1_s is not None:
        A1_s *= SI_to_file.s

    # Shift to box coordinates
    A2_pos += boxsize / 2.0

    # Print info
    if verbosity >= 1:
        print("")
        print("num_particle = %d" % num_particle)
        print("boxsize      = %.2g" % boxsize)
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
        if A1_s is not None:
            print("  s = %.5g, %.5g" % (np.amin(A1_s), np.amax(A1_s)))
        print("")

    # Save
    # Header
    grp = f.create_group("/Header")
    grp.attrs["BoxSize"] = [boxsize] * 3
    grp.attrs["NumPart_Total"] = [num_particle, 0, 0, 0, 0, 0]
    grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
    grp.attrs["NumPart_ThisFile"] = [num_particle, 0, 0, 0, 0, 0]
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
    grp.create_dataset(Di_hdf5_particle_label["pos"], data=A2_pos, dtype="d")
    grp.create_dataset(Di_hdf5_particle_label["vel"], data=A2_vel, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["m"], data=A1_m, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["h"], data=A1_h, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["rho"], data=A1_rho, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["P"], data=A1_P, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["u"], data=A1_u, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["id"], data=A1_id, dtype="L")
    grp.create_dataset(Di_hdf5_particle_label["mat_id"], data=A1_mat_id, dtype="i")
    if A1_s is not None:
        grp.create_dataset(Di_hdf5_particle_label["s"], data=A1_s, dtype="f")

    if verbosity >= 1:
        print('Saved "%s"' % f.filename[-64:])
