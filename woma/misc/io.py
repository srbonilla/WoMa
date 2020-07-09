"""
WoMa IO utilities

The spherical and spinning planets' data and attributes can be conveniently 
saved and loaded using HDF5 files (see www.hdfgroup.org and www.h5py.org).
See .save(), .load(), and load_file in the Planet and SpinPlanet classes.
"""

import numpy as np
import sys
import h5py

from woma.misc import utils


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
