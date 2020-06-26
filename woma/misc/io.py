"""
WoMa IO utilities

### WIP ###

The profile and particle data and attributes can be conveniently saved and 
loaded with HDF5 files (see www.hdfgroup.org and www.h5py.org).

The particle data also match the format for SWIFT (see www.swiftsim.com).
"""

import numpy as np
import sys
import h5py

from woma.misc import utils


# HDF5 labels
Di_hdf5_planet_label = {
    "num_layer": "Number of Layers",
    "mat_layer": "Layer Materials",
    "mat_id_layer": "Layer Material IDs",
    "T_rho_type": "Layer T-rho Type",
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


def save_planet(planet, Fp_planet, verbosity=1):
    """ Save a planet object to a file.

    Parameters
    ----------
    planet : woma.Planet
        The planet object.

    Fp_planet : str
        The file path.
    """
    Fp_planet = utils.check_end(Fp_planet, ".hdf5")

    if verbosity >= 1:
        print('Saving "%s"... ' % Fp_planet[-60:], end="")
    sys.stdout.flush()

    with h5py.File(Fp_planet, "w") as f:
        # Group
        grp = f.create_group("/planet")

        # Lists not numpy for attributes
        if type(planet.A1_mat_layer).__module__ == np.__name__:
            planet.A1_mat_layer = planet.A1_mat_layer.tolist()

        # Attributes
        grp.attrs[Di_hdf5_planet_label["num_layer"]] = planet.num_layer
        grp.attrs[Di_hdf5_planet_label["mat_layer"]] = planet.A1_mat_layer
        grp.attrs[Di_hdf5_planet_label["mat_id_layer"]] = planet.A1_mat_id_layer
        grp.attrs[Di_hdf5_planet_label["T_rho_type"]] = planet.A1_T_rho_type
        grp.attrs[Di_hdf5_planet_label["R_layer"]] = planet.A1_R_layer
        grp.attrs[Di_hdf5_planet_label["M_layer"]] = planet.A1_M_layer
        grp.attrs[Di_hdf5_planet_label["M"]] = planet.M
        grp.attrs[Di_hdf5_planet_label["R"]] = planet.R
        grp.attrs[Di_hdf5_planet_label["idx_layer"]] = planet.A1_idx_layer
        grp.attrs[Di_hdf5_planet_label["P_s"]] = planet.P_s
        grp.attrs[Di_hdf5_planet_label["T_s"]] = planet.T_s
        grp.attrs[Di_hdf5_planet_label["rho_s"]] = planet.rho_s

        # Arrays
        grp.create_dataset(Di_hdf5_planet_label["r"], data=planet.A1_r, dtype="d")
        grp.create_dataset(
            Di_hdf5_planet_label["m_enc"], data=planet.A1_m_enc, dtype="d"
        )
        grp.create_dataset(Di_hdf5_planet_label["rho"], data=planet.A1_rho, dtype="d")
        grp.create_dataset(Di_hdf5_planet_label["T"], data=planet.A1_T, dtype="d")
        grp.create_dataset(Di_hdf5_planet_label["P"], data=planet.A1_P, dtype="d")
        grp.create_dataset(Di_hdf5_planet_label["u"], data=planet.A1_u, dtype="d")
        grp.create_dataset(
            Di_hdf5_planet_label["mat_id"], data=planet.A1_mat_id, dtype="i"
        )


def load_planet(name, Fp_planet, verbosity=1):
    """ Return a new Planet object loaded from a file.

    Parameters
    ----------
    name : str
        The name of the planet object.

    Fp_planet : str
        The file path.
        
    Returns
    -------
    p : woma.Planet
        The loaded planet object.
    """
    p = Planet(name=name, Fp_planet=Fp_planet)

    Fp_planet = utils.check_end(p.Fp_planet, ".hdf5")

    if verbosity >= 1:
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

    p.update_attributes()

    if verbosity >= 1:
        print("Done")
        p.print_info()

    return p
