import numpy as np
import h5py
import sys

# ========
# Standard constants
# ========
pi          = np.pi
G           = 6.67384e-11               # Gravitational constant (m^3 kg^-1 s^-2)
R_gas       = 8.3145                    # Gas constant (J K^-1 mol^-1)
gamma_idg   = 5/3                       # Ideal gas adiabatic index
R_earth     = 6371000
M_earth     = 5.972E24

# Material IDs ( = type_id * type_factor + unit_id)
type_factor = 100
Di_mat_type = {
   'Til'       : 1,
   'HM80'      : 2,
   'SESAME'    : 3,
   }
Di_mat_id   = {
   # Tillotson
   'Til_iron'      : Di_mat_type['Til']*type_factor,
   'Til_granite'   : Di_mat_type['Til']*type_factor + 1,
   'Til_water'     : Di_mat_type['Til']*type_factor + 2,
   # Hubbard & MacFarlane (1980) Uranus/Neptune
   'HM80_HHe'      : Di_mat_type['HM80']*type_factor,      # Hydrogen-helium atmosphere
   'HM80_ice'      : Di_mat_type['HM80']*type_factor + 1,  # H20-CH4-NH3 ice mix
   'HM80_rock'     : Di_mat_type['HM80']*type_factor + 2,  # SiO2-MgO-FeS-FeO rock mix
   # SESAME
   'SESAME_iron'   : Di_mat_type['SESAME']*type_factor,        # 2140
   'SESAME_basalt' : Di_mat_type['SESAME']*type_factor + 1,    # 7530
   'SESAME_water'  : Di_mat_type['SESAME']*type_factor + 2,    # 7154
   'SS08_water'    : Di_mat_type['SESAME']*type_factor + 3,    # Senft & Stewart (2008)
   }

# ========
# Unit conversions
# ========
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
       self.m          = m
       self.l          = l
       self.t          = t
       # Derived conversions
       self.v          = l * t**-1
       self.a          = l * t**-2
       self.rho        = m * l**-3
       self.drho_dt    = m * l**-4
       self.P          = m * l**-1 * t**-2
       self.u          = l**2 * t**-2
       self.du_dt      = l**2 * t**-3
       self.E          = m * l**2 * t**-2
       self.s          = l**2 * t**-2
       self.G          = m**-1 * l**3 * t**-2

   def inv(self):
       """ Return the inverse to this conversion """
       return Conversions(1 / self.m, 1 / self.l, 1 / self.t)

SI_to_SI    = Conversions(1, 1, 1)      # No-op

# Standard
cgs_to_SI   = Conversions(1e-3, 1e-2, 1)
SI_to_cgs   = cgs_to_SI.inv()
# Angles
deg_to_rad  = pi / 180
rad_to_deg  = 1 / deg_to_rad
# Time
hour_to_s   = 60**2                     # = 3.60e3 s
s_to_hour   = 1 / hour_to_s             # = 2.77e-4 h
day_to_s    = 24 * hour_to_s            # = 8.64e4 s
s_to_day    = 1 / day_to_s              # = 1.16e-5 days
week_to_s   = 7 * day_to_s              # = 6.05e5 s
s_to_week   = 1 / week_to_s             # = 1.65e-6 weeks
yr_to_s     = 365.25 * day_to_s         # = 3.15e7 s
s_to_yr     = 1 / yr_to_s               # = 3.17e-8 yr
# Distance
mile_to_m   = 1609.34
m_to_mile   = 1 / mile_to_m             # = 6.21e-4 miles
au_to_m     = 1.496e11
m_to_au     = 1 / au_to_m               # = 6.68e-12 AU
# Speed
mph_to_mps  = mile_to_m / hour_to_s     # = 0.447 m/s
mps_to_mph  = 1 / mph_to_mps            # = 2.237 mph
kmph_to_mps = 1e3 / hour_to_s           # = 0.278 m/s
mps_to_kmph = 1 / kmph_to_mps           # = 3.6 km/h
# Pressure
bar_to_Pa   = 1e5
Pa_to_bar   = 1 / bar_to_Pa             # = 1e-5 bar
Mbar_to_Pa  = bar_to_Pa * 1e6           # = 1e11 Pa
Pa_to_Mbar  = 1 / Mbar_to_Pa            # = 1e-11 Mbar
bar_to_Ba   = bar_to_Pa * SI_to_cgs.P   # = 1e6 Ba
Ba_to_bar   = 1 / bar_to_Ba             # = 1e-6 bar
Mbar_to_Ba  = Mbar_to_Pa * SI_to_cgs.P  # = 1e12 Ba
Ba_to_Mbar  = 1 / Mbar_to_Ba            # = 1e-12 Mbar
# Mass
amu_to_kg   = 1.6605e-27
kg_to_amu   = 1 / amu_to_kg             # = 6.02e26 amu

Di_hdf5_label       = {                 # Type
   'pos'       : 'Coordinates',        # d
   'vel'       : 'Velocities',         # f
   'm'         : 'Masses',             # f
   'h'         : 'SmoothingLength',    # f
   'u'         : 'InternalEnergy',     # f
   'rho'       : 'Density',            # f
   'P'         : 'Pressure',           # f
   's'         : 'Entropy',            # f
   'id'        : 'ParticleIDs',        # L
   'mat_id'    : 'MaterialID',         # i
   'phi'       : 'Potential',          # f
   'T'         : 'Temperature',        # f
   }

def save_picle_data(f, A2_pos, A2_vel, A1_m, A1_h, A1_rho, A1_P, A1_u, A1_id,
                    A1_mat_id, boxsize, file_to_SI):
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
    num_picle   = len(A1_id)

    # Convert to file units
    """
    SI_to_file  = file_to_SI.inv()
    A2_pos      *= R_earth * SI_to_file.l
    A2_vel      *= SI_to_file.v
    A1_m        *= M_earth * SI_to_file.m
    A1_rho      *= SI_to_file.rho
    A1_P        *= SI_to_file.P
    A1_u        *= SI_to_file.u
    A1_h        *= R_earth * SI_to_file.l
    boxsize     *= R_earth * SI_to_file.l
    """
    SI_to_file  = file_to_SI.inv()
    A2_pos      *= SI_to_file.l
    A2_vel      *= SI_to_file.v
    A1_m        *= SI_to_file.m
    A1_rho      *= SI_to_file.rho
    A1_P        *= SI_to_file.P
    A1_u        *= SI_to_file.u
    A1_h        *= SI_to_file.l
    boxsize     *= SI_to_file.l
    # Shift coordinates such that the origin is at the box corner and all
    # positions are positive, instead of the origin in the centre
    A2_pos  += boxsize / 2.0

    # Print info to double check
    print("")
    print("num_picle    = %d" % num_picle)
    print("boxsize      = %g R_E" % boxsize)
    print("mat_id       = ", end='')
    for mat_id in np.unique(A1_mat_id):
        print("%d " % mat_id, end='')
    print("\n")
    print("Unit mass    = %.5e g" % (file_to_SI.m * SI_to_cgs.m))
    print("Unit length  = %.5e cm" % (file_to_SI.l * SI_to_cgs.l))
    print("Unit time    = %.5e s" % file_to_SI.t)
    print("")
    print("Min, max values (file units):")
    print("  pos = [%.5g, %.5g,    %.5g, %.5g,    %.5g, %.5g]" %
          (np.amin(A2_pos[:, 0]), np.amax(A2_pos[:, 0]),
           np.amin(A2_pos[:, 1]), np.amax(A2_pos[:, 1]),
           np.amin(A2_pos[:, 2]), np.amax(A2_pos[:, 2])))
    if np.amax(abs(A2_pos)) > boxsize:
        print("# Particles are outside the box!")
        sys.exit()
    print("  vel = [%.5g, %.5g,    %.5g, %.5g,    %.5g, %.5g]" %
          (np.amin(A2_vel[:, 0]), np.amax(A2_vel[:, 0]),
           np.amin(A2_vel[:, 1]), np.amax(A2_vel[:, 1]),
           np.amin(A2_vel[:, 2]), np.amax(A2_vel[:, 2])))
    for name, array in zip(
        ["m", "rho", "P", "u", "h"],
        [A1_m, A1_rho, A1_P, A1_u, A1_h]
        ):
        print("  %s = %.5g, %.5g"
              % (name, np.amin(array), np.amax(array)))
    print("")

    # Save
    # Header
    grp = f.create_group("/Header")
    grp.attrs["BoxSize"]                = [boxsize] * 3
    grp.attrs["NumPart_Total"]          = [num_picle, 0, 0, 0, 0, 0]
    grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
    grp.attrs["NumPart_ThisFile"]       = [num_picle, 0, 0, 0, 0, 0]
    grp.attrs["Time"]                   = 0.0
    grp.attrs["NumFilesPerSnapshot"]    = 1
    grp.attrs["MassTable"]              = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    grp.attrs["Flag_Entropy_ICs"]       = 0
    grp.attrs["Dimension"]              = 3

    # Runtime parameters
    grp = f.create_group("/RuntimePars")
    grp.attrs["PeriodicBoundariesOn"]   = 0

    # Units
    grp = f.create_group("/Units")
    grp.attrs["Unit mass in cgs (U_M)"]         = file_to_SI.m * SI_to_cgs.m
    grp.attrs["Unit length in cgs (U_L)"]       = file_to_SI.l * SI_to_cgs.l
    grp.attrs["Unit time in cgs (U_t)"]         = file_to_SI.t
    grp.attrs["Unit current in cgs (U_I)"]      = 1.0
    grp.attrs["Unit temperature in cgs (U_T)"]  = 1.0

    # Particles
    grp = f.create_group("/PartType0")
    grp.create_dataset(Di_hdf5_label['pos'], data=A2_pos, dtype='d')
    grp.create_dataset(Di_hdf5_label['vel'], data=A2_vel, dtype='f')
    grp.create_dataset(Di_hdf5_label['m'], data=A1_m, dtype='f')
    grp.create_dataset(Di_hdf5_label['h'], data=A1_h, dtype='f')
    grp.create_dataset(Di_hdf5_label['rho'], data=A1_rho, dtype='f')
    grp.create_dataset(Di_hdf5_label['P'], data=A1_P, dtype='f')
    grp.create_dataset(Di_hdf5_label['u'], data=A1_u, dtype='f')
    grp.create_dataset(Di_hdf5_label['id'], data=A1_id, dtype='L')
    grp.create_dataset(Di_hdf5_label['mat_id'], data=A1_mat_id, dtype='i')

    print("Saved \"%s\"" % f.filename[-64:])

def get_picle_data(f, param, file_to_SI=SI_to_SI, A1_sort=None,
                   custom_info=None):
    """ Load, convert, and return a particle property array.

        Args:
            f (h5py File)
                The opened hdf5 data file (with 'r').

            param (str)
                The particle property to get. See Di_hdf5_label for details.

            file_to_SI (opt. Conversions)
                Unit conversion object from the file's units to SI.
                SI_to_SI:   (Default) No conversion.

            A1_sort (opt. [int])
                Array of indices for sorting the array.

            custom_info (opt. ?)
                Any extra information needed for param='is_custom' stuff.

        Returns:
            A1_param ([?])
                The array of the particle property data (std units).
    """
    try:
        num_picle   = f['Header'].attrs['NumPart_Total'][0]
    except KeyError:
        print(" ## Warning: Failed to get the standard 'Header/NumPart_Total' ")

    if param == 'pos':
        A1_param    = ((f['PartType0/' + Di_hdf5_label['pos']].value
                        - 0.5 * f['Header'].attrs['BoxSize']) * file_to_SI.l
                       / R_earth)

    elif param == 'x':
        A1_param    = get_picle_data(f, 'pos', file_to_SI)[:, 0]

    elif param == 'y':
        A1_param    = get_picle_data(f, 'pos', file_to_SI)[:, 1]

    elif param == 'z':
        A1_param    = get_picle_data(f, 'pos', file_to_SI)[:, 2]

    elif param == 'vel':
        A1_param    = (f['PartType0/' + Di_hdf5_label['vel']].value
                       * file_to_SI.v)

    elif param == 'v_x':
        A1_param    = get_picle_data(f, 'vel', file_to_SI)[:, 0]

    elif param == 'v_y':
        A1_param    = get_picle_data(f, 'vel', file_to_SI)[:, 1]

    elif param == 'v_z':
        A1_param    = get_picle_data(f, 'vel', file_to_SI)[:, 2]

    elif param == 'm':
        A1_param    = (f['PartType0/' + Di_hdf5_label['m']].value * file_to_SI.m
                       / M_earth)

    elif param == 'h':
        A1_param    = (f['PartType0/' + Di_hdf5_label['h']].value * file_to_SI.l
                       / R_earth)

    elif param == 'u':
        A1_param    = f['PartType0/' + Di_hdf5_label['u']].value * file_to_SI.u

    elif param == 'rho':
        A1_param    = (f['PartType0/' + Di_hdf5_label['rho']].value
                       * file_to_SI.rho)

    elif param == 'P':
        A1_param    = f['PartType0/' + Di_hdf5_label['P']].value * file_to_SI.P

    elif param == 's':
        A1_param    = f['PartType0/' + Di_hdf5_label['s']].value * file_to_SI.s

    elif param == 'id':
        A1_param    = f['PartType0/' + Di_hdf5_label['id']].value

    elif param == 'mat_id':
        A1_param    = f['PartType0/' + Di_hdf5_label['mat_id']].value

    elif param == 'phi':
        try:
            A1_param    = (f['PartType0/' + Di_hdf5_label['phi']].value
                           * file_to_SI.u)
        except KeyError:
            A1_param    = np.zeros(num_picle)

    elif param == 'T':
        try:
            A1_param    = f['PartType0/' + Di_hdf5_label['T']].value
        except KeyError:
            A1_param    = np.zeros(num_picle)

    # Invalid parameter
    else:
        raise Exception("Error: Invalid particle property: %s" % param)

    if A1_sort is not None:
        return A1_param[A1_sort]
    else:
        return A1_param


def multi_get_picle_data(f, A1_param, file_to_SI=SI_to_SI, A1_sort=None,
                         custom_info=None):
    """ Load, convert, and return multiple particle property arrays.

        Args:
            f (h5py File)
                The opened hdf5 data file (with 'r').

            A1_param ([str])
                List of the particle properties to get. See Di_hdf5_label for
                details.

            file_to_SI (opt. Conversions)
                Unit conversion object from the file's units to SI.
                SI_to_SI:   (Default) No conversion.

            A1_sort (opt. [int])
                Array of indices for sorting the arrays.

            custom_info (opt. ?)
                Any extra information needed for param='is_custom' stuff.

        Returns:
            A1_data ([[?]])
                The list of the arrays of particle property data (std units).
    """
    A1_data = []
    # Load each requested array
    for param in A1_param:
        A1_data.append(
            get_picle_data(f, param, file_to_SI=file_to_SI, A1_sort=A1_sort,
                           custom_info=custom_info)
            )

    return A1_data
