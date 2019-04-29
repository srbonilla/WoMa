""" SEAGen

    A python implementation of the stretched equal area (SEA) algorithm for
    generating spherically symmetric arrangements of particles with accurate
    particle densities, e.g. for SPH initial conditions that precisely match an
    arbitrary density profile (Kegerreis et al. 2019, 
    https://arxiv.org/pdf/1901.09934.pdf).

    See README.md and https://github.com/jkeger/seagen for more information.

    Jacob Kegerreis (2019) jacob.kegerreis@durham.ac.uk

    GNU General Public License v3+, see LICENSE.txt.

    See the __init__() doc strings of the GenShell and GenSphere classes for
    the main documentation details, and see examples.py for example uses.
"""
# ========
# Contents:
# ========
#   I   Functions
#   II  Classes
#   III Main

import numpy as np
import sys

# ========
# Constants
# ========
deg_to_rad  = np.pi/180
banner      = "#  SEAGen \n" "#  https://github.com/jkeger/seagen \n"


# //////////////////////////////////////////////////////////////////////////// #
#                               I. Functions                                   #
# //////////////////////////////////////////////////////////////////////////// #

def polar_to_cartesian(r, theta, phi):
    """ Convert spherical polars to cartesian coordinates.

        Args:
            r (float or [float])
                Radius.

            theta (float or [float])
                Zenith angle (colatitude) (radians).

            phi (float or [float])
                Azimuth angle (longitude) (radians).

        Returns:
            x, y, z (float or [float])
                Cartesian coordinates.
    """
    x   = r * np.cos(phi) * np.sin(theta)
    y   = r * np.sin(phi) * np.sin(theta)
    z   = r * np.cos(theta)

    return x, y, z


def get_euler_rotation_matrix(alpha, beta, gamma):
    """ Return the rotation matrix for three Euler angles.

        Args:
            alpha, beta, gamma (float)
                Euler angles (radians).

        Returns:
            A2_rot ([[float]])
                3x3 rotation matrix.
    """
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    sb  = np.sin(beta)
    cb  = np.cos(beta)
    sg  = np.sin(gamma)
    cg  = np.cos(gamma)

    return np.array([
        [cg*cb*ca - sg*sa,      cg*cb*sa + sg*ca,       -cg*sb],
        [-sg*cb*ca - cg*sa,     -sg*cb*sa + cg*ca,      sg*sb],
        [sb*ca,                 sb*sa,                  cb ]
        ])


def get_shell_mass(r_inner, r_outer, rho):
    """ Calculate the mass of a uniform-density shell. """
    return 4/3*np.pi * rho * (r_outer**3 - r_inner**3)


def get_weighted_mean(A1_weight, A1_value):
    """ Calculate the mean of the value array weighted by the weights array. """
    return np.sum(A1_weight * A1_value) / np.sum(A1_weight)


# //////////////////////////////////////////////////////////////////////////// #
#                               II. Classes                                    #
# //////////////////////////////////////////////////////////////////////////// #

class GenShell(object):
    """ Generate a single spherical shell of points ("particles") at a fixed
        radius, using the SEA method described in Kegerreis et al. 2019 ("K19").

        See __init__()'s documentation for more details.

        Basic Usage:
            e.g. Create a single shell of particles and print their positions:
                >>> import seagen
                >>> N = 100
                >>> r = 1
                >>> particles = seagen.GenShell(N, r)
                >>> print(particles.x, particles.y, particles.z)
    """
    def __init__(self, N, r, do_stretch=True, do_rotate=True):
        """ Generate a single spherical shell of particles.

            Args:
                N (int)
                    The number of cells/particles to create.

                r (float)
                    The radius of the shell.

                do_stretch (opt. bool)
                    Default True. Set False to not do the SEA method's latitude
                    stretching.

                do_rotate (opt. bool)
                    Default True. Set False to not randomly rotate the sphere of
                    particles after their intial placement.

            Output attrs: (Also accessable without the A1_ prefix)
                A1_x, A1_y, A1_z ([float])
                    Particle cartesian position arrays.

                Note: Spherical polar coordinates are used for the particles
                internally but do not have the final rotation applied to them.
        """
        self.N      = N
        self.A1_r   = r * np.ones(N)

        # Derived properties
        self.A_reg  = 4 * np.pi / N

        # Start in spherical polar coordinates for the initial placement
        self.get_collar_areas()

        self.update_collar_colatitudes()

        self.get_point_positions()

        if do_stretch:
            a, b    = self.get_stretch_params(self.N)
            self.apply_stretch_factor(a, b)

        # Now convert to cartesian coordinates for the rotation and output
        self.A1_x, self.A1_y, self.A1_z = polar_to_cartesian(
            self.A1_r, self.A1_theta, self.A1_phi)

        if do_rotate:
            self.apply_random_rotation()


    def get_cap_colatitude(self):
        """ Calculate the cap colatitude.

            K19 eqn. (3)

            Returns:
                theta_cap (float)
                    The cap colatitude (radians).
        """
        return 2 * np.arcsin(np.sqrt(1 / self.N))


    def get_number_of_collars(self):
        """ Calculate the number of collars (not including the polar caps).

            K19 eqn. (4)

            Sets and returns:
                N_col (int)
                    The number of collars (not including the polar caps).
        """
        theta_cap   = self.get_cap_colatitude()

        self.N_col  = int(round((np.pi - 2 * theta_cap)/(np.sqrt(self.A_reg))))

        return self.N_col


    def get_collar_colatitudes(self):
        """ Calculate the top colatitudes of all of the collars, including the
            bottom cap's.

            Sets and returns:
                A1_collar_theta ([float])
                    The top colatitudes of all of the collars.
        """
        self.get_number_of_collars()

        cap_height          = self.get_cap_colatitude()
        height_of_collar    = (np.pi - 2 * cap_height) / self.N_col

        # Allocate collars array
        self.A1_collar_theta    = np.arange(self.N_col + 1, dtype=float)
        # Collars have a fixed height initially
        self.A1_collar_theta    *= height_of_collar
        # Starting at the bottom of the top polar cap
        self.A1_collar_theta    += (cap_height)

        return self.A1_collar_theta


    def get_collar_area(self, theta_i, theta_i_minus_one):
        """ Calculate the area of a collar given the collar heights of itself
            and its neighbour.

            K19 eqn. (5)

            Args:
                theta_i (float)
                    The colatitude of the bottom of the collar.

                theta_i_minus_one (float)
                    The colatitudes of the bottom of the previous collar, i.e.
                    the top of this collar.

            Returns:
                collar_area (float)
                    The collar's area.
        """
        sin2_theta_i        = np.sin(theta_i / 2)**2
        sin2_theta_i_m_o    = np.sin(theta_i_minus_one / 2)**2

        return 4 * np.pi * (sin2_theta_i - sin2_theta_i_m_o)


    def get_collar_areas(self):
        """ Calculate the collar areas.

            Sets and returns:
                A1_collar_area ([float])
                    The collar areas.
        """
        self.get_collar_colatitudes()

        self.A1_collar_area = np.empty(self.N_col)

        self.A1_collar_area[:]  = self.get_collar_area(
            self.A1_collar_theta[1:], self.A1_collar_theta[:-1])

        return self.A1_collar_area


    def get_ideal_N_regions_in_collar(self, A_col):
        """ Calculate the ideal number of regions in a collar.

            K19 eqn (7).

            Returns:
                N_reg_ideal (float)
                    The ideal number of regions in a collar.
        """
        return A_col / self.A_reg


    def get_N_regions_in_collars(self):
        """ Calculate the number of regions in each collar, not including the
            top polar cap.

            K19 eqn (8,9).

            Sets and returns:
                A1_N_reg_in_collar ([int])
                    The number of regions in each collar.
        """
        self.A1_N_reg_in_collar = np.empty(self.N_col, dtype=int)
        A1_collar_area          = self.get_collar_areas()

        discrepancy = 0

        for i in range(len(self.A1_N_reg_in_collar)):
            N_reg_ideal = self.get_ideal_N_regions_in_collar(A1_collar_area[i])

            self.A1_N_reg_in_collar[i]  = int(round(N_reg_ideal + discrepancy))

            discrepancy += N_reg_ideal - self.A1_N_reg_in_collar[i]

        return self.A1_N_reg_in_collar


    def update_collar_colatitudes(self):
        """ Update the collar colatitudes to use the now-integer numbers of
            regions in each collar instead of the ideal.

            K19 eqn (10).

            Sets and returns:
                A1_collar_theta ([float])
                    The top colatitudes of all of the collars.
        """
        # First we must get the cumulative number of regions in each collar,
        # including the top polar cap
        A1_N_reg_in_collar_cum  = np.cumsum(self.get_N_regions_in_collars()) + 1
        A1_N_reg_in_collar_cum  = np.append([1], A1_N_reg_in_collar_cum)

        self.A1_collar_theta    = 2 * np.arcsin(
            np.sqrt(A1_N_reg_in_collar_cum * self.A_reg / (4 * np.pi)))

        return self.A1_collar_theta


    def choose_longitude_offset(self, N_i, N_i_minus_one, d_phi_i,
                                d_phi_i_minus_one):
        """ Choose the starting longitude for particles in this collar.

            K19 paragraph after eqn (12).

            Args:
                N_i, N_i_minus_one (int)
                    The number of regions in this collar and the previous one.

                d_phi_i, d_phi_i_minus_one (float)
                    The longitude angle between adjacent particles in this
                    collar and the previous one.

            Returns:
                phi_0 (float)
                    The starting longitude for particles in this collar.
        """
        is_N_i_even             = abs((N_i % 2) - 1)
        is_N_i_minus_one_even   = abs((N_i_minus_one % 2) - 1)

        # Exclusive or
        if is_N_i_even != is_N_i_minus_one_even:
            phi_0   = 0.5 * (is_N_i_even * d_phi_i
                             + is_N_i_minus_one_even * d_phi_i_minus_one)
        else:
            phi_0   = 0.5 * min(d_phi_i, d_phi_i_minus_one)

        return phi_0


    def get_point_positions(self):
        """ Calculate the point positions in the centres of every region.

            K19 eqn (11,12).

            Sets and returns:
                A1_theta ([float])
                    The point colatitudes.

                A1_phi ([float])
                    The point longitudes.
        """
        N_tot   = self.A1_N_reg_in_collar.sum() + 2

        self.A1_theta   = np.empty(N_tot)
        self.A1_phi     = np.empty(N_tot)

        # The cap particles are at the poles, listed at the end of these arrays.
        self.A1_theta[-2]   = 0.0
        self.A1_theta[-1]   = np.pi
        self.A1_phi[-2]     = 0.0
        self.A1_phi[-1]     = 0.0

        # All regions in a collar are at the same colatitude, theta.
        A1_theta        = np.zeros(self.N_col + 2)
        A1_theta[:-2]   = 0.5 * (
            self.A1_collar_theta[:-1] + self.A1_collar_theta[1:])

        # Particles in each collar are equally spaced in longitude, phi,
        # and offset appropriately from the previous collar.
        A1_d_phi    = 2 * np.pi / self.A1_N_reg_in_collar
        A1_phi_0    = np.empty(self.N_col)

        for i, phi_0_i in enumerate(A1_phi_0):
            # The first collar has no previous collar to rotate away from
            # so doesn't need an offset.
            if i == 0:
                phi_0_i = 0
            else:
                phi_0_i = self.choose_longitude_offset(
                    self.A1_N_reg_in_collar[i], self.A1_N_reg_in_collar[i-1],
                    A1_d_phi[i], A1_d_phi[i-1]
                    )

                # Also add a random initial offset to ensure that successive
                # collars do not create lines of ~adjacent particles.
                # (Second paragraph following K19 eqn (12).)
                m       = np.random.randint(0, self.A1_N_reg_in_collar[i-1])
                phi_0_i += (m * A1_d_phi[i-1])

        # Fill the position arrays.
        N_regions_done  = 0
        for region, N_regions_in_collar in enumerate(self.A1_N_reg_in_collar):
            N_regions_done_next = N_regions_in_collar + N_regions_done

            # Set A1_theta
            self.A1_theta[N_regions_done:N_regions_done_next] = A1_theta[region]

            # Set phi (K19 eqn (12))
            j               = np.arange(N_regions_in_collar, dtype=float)
            A1_phi_collar   = A1_phi_0[region] + j * A1_d_phi[region]

            self.A1_phi[N_regions_done:N_regions_done_next] = A1_phi_collar

            N_regions_done  = N_regions_done_next

        self.A1_phi         %= 2 * np.pi
        self.A1_theta       %= np.pi
        self.A1_theta[-1]   = np.pi

        return self.A1_theta, self.A1_phi


    def get_stretch_params(self, N):
        """ Return the a and b parameters for the latitude stretching.

            Empirically, b = 10 * a gives an excellent low density scatter.
            For N > 80, a = 0.2. For N < 80, a has been fit by trial and error.

            Args:
                N (int)
                    The number of particles in the shell.

            Returns:
                a, b (float)
                    The stretch parameters.
        """
        if N > 80:
            a   = 0.20
        elif N == 79:
            a   = 0.20
        elif N == 78:
            a   = 0.20
        elif N == 77:
            a   = 0.20
        elif N == 76:
            a   = 0.21
        elif N == 75:
            a   = 0.21
        elif N == 74:
            a   = 0.22
        elif N == 73:
            a   = 0.23
        elif N == 72:
            a   = 0.23
        elif N == 71:
            a   = 0.24
        elif N == 70:
            a   = 0.25
        elif N == 69:
            a   = 0.24
        elif N == 68:
            a   = 0.24
        elif N == 67:
            a   = 0.24
        elif N == 66:
            a   = 0.24
        elif N == 65:
            a   = 0.21
        elif N == 64:
            a   = 0.20
        elif N == 63:
            a   = 0.21
        elif N == 62:
            a   = 0.21
        elif N == 61:
            a   = 0.22
        elif N == 60:
            a   = 0.21
        elif N == 59:
            a   = 0.21
        elif N == 58:
            a   = 0.22
        elif N == 57:
            a   = 0.215
        elif N == 56:
            a   = 0.22
        elif N == 55:
            a   = 0.20
        elif N == 54:
            a   = 0.21
        elif N == 53:
            a   = 0.20
        elif N == 52:
            a   = 0.23
        elif N == 51:
            a   = 0.23
        elif N == 50:
            a   = 0.25
        elif N == 49:
            a   = 0.21
        elif N == 48:
            a   = 0.22
        elif N == 47:
            a   = 0.225
        elif N == 46:
            a   = 0.22
        elif N == 45:
            a   = 0.23
        elif N == 44:
            a   = 0.19
        elif N == 43:
            a   = 0.235
        elif N == 42:
            a   = 0.25
        elif N == 41:
            a   = 0.18
        elif N == 40:
            a   = 0.23
        elif N == 39:
            a   = 0.25
        elif N == 38:
            a   = 0.25
        elif N == 37:
            a   = 0.26
        elif N == 36:
            a   = 0.27
        elif N == 35:
            a   = 0.21
        elif N == 34:
            a   = 0.22
        elif N == 33:
            a   = 0.20
        elif N == 32:
            a   = 0.25
        elif N == 31:
            a   = 0.27
        elif N == 28:
            a   = 0.20
        elif N == 27:
            a   = 0.19
        else:
            a   = 0.20
            print("\nUntested stretching N = %d!" % N)

        return a, 10*a


    def apply_stretch_factor(self, a=0.2, b=2.0):
        """ Apply the SEA stretch factor.

            K19 eqn (13).

            Args:
                a, b (float)
                    The stretching parameters.

            Sets:
                A1_theta ([float])
                    The point colatitudes.
        """
        pi_o_2      = np.pi / 2
        inv_sqrtN   = 1 / np.sqrt(self.N)

        A1_prefactor    = (pi_o_2 - self.A1_theta) * a * inv_sqrtN

        A1_exp_factor   = - ((pi_o_2 - abs(pi_o_2 - self.A1_theta))
                             / (np.pi * b * inv_sqrtN))

        self.A1_theta   += (A1_prefactor * np.exp(A1_exp_factor))

        # Leave the cap points at the poles
        self.A1_theta[-2]   = 0.0
        self.A1_theta[-1]   = np.pi

        return


    def apply_random_rotation(self):
        """ Rotate the shell with three random Euler angles. """
        # Random Euler angles
        alpha   = np.random.rand() * 2*np.pi
        beta    = np.random.rand() * 2*np.pi
        gamma   = np.random.rand() * 2*np.pi
        A2_rot  = get_euler_rotation_matrix(alpha, beta, gamma)

        # Array of position vectors
        A2_pos  = np.array([self.A1_x, self.A1_y, self.A1_z]).transpose()

        # Rotate each position vector
        for i in range(len(A2_pos)):
            A2_pos[i]   = np.dot(A2_rot, A2_pos[i])

        # Unpack positions
        self.A1_x, self.A1_y, self.A1_z  = A2_pos.transpose()

        return


    # Aliases for the main outputs without my array notation
    @property
    def x(self):
        return self.A1_x
    @property
    def y(self):
        return self.A1_y
    @property
    def z(self):
        return self.A1_z


class GenSphere(object):
    """ Generate particle initial conditions with the SEA method and nested
        shells, following a density profile.

        See __init__()'s documentation for more details.

        Basic Usage:
            e.g. Create a full sphere of particles on a simple density profile
            and print their positions and masses:
                >>> import seagen
                >>> import numpy as np
                >>> N = 100000
                >>> radii = np.arange(0.01, 10, 0.01)
                >>> densities = np.ones(len(radii))     # e.g. constant density
                >>> particles = seagen.GenSphere(N, radii, densities)
                >>> print(particles.x, particles.y, particles.z, particles.m)
    """
    def __init__(self, N_picle_des, A1_r_prof, A1_rho_prof, A1_mat_prof=None,
                 A1_u_prof=None, A1_T_prof=None, A1_P_prof=None,
                 do_stretch=True, verb=1):
        """ Generate nested spherical shells of particles to match radial
            profiles.

            The profiles should give the outer radius of each thin profile shell
            and the corresponding density (and other values) within that shell.
            So, the first profile radius should be small but greater than zero.

            Args:
                N_picle_des (int)
                    The desired number of particles.

                A1_r_prof ([float])
                    The array of profile radii.

                A1_rho_prof ([float])
                    The array of densities at the profile radii.

                A1_mat_prof (opt. [int])
                    The array of material identifiers at the profile radii. If
                    not provided, then default to all zeros.

                A1_u_prof A1_T_prof A1_P_prof (opt. [float])
                    Optional arrays of other values at the profile radii:
                    specific internal energy, temperature, and pressure.
                    Default None.

                do_stretch (opt. bool)
                    Default True. Set False to not do the SEA method's latitude
                    stretching.

                verb (opt. int)
                    The verbosity to control printed output:
                    0       None
                    1       Standard (default)
                    2       Extra
                    3       Debug

            Output attrs: (Also accessable without the A1_ prefix)
                A1_x, A1_y, A1_z, A1_r ([float])
                    The arrays of the particles' cartesian coordinates and
                    radii.

                A1_m, A1_rho ([float])
                    The arrays of the particles' masses and densities,
                    using the profiles at the right radius.

                A1_mat ([int])
                    The arrays of the particles' material identifiers.

                A1_u, A1_T, A1_P ([float] or None)
                    If the profile arrays were provided, then the corresponding
                    arrays of the particles' sp. int. energies, temperature,
                    and/or pressures. Otherwise, None.

                N_picle (int)
                    The final number of particles.

                N_shell_tot (int)
                    The total number of shells.
        """
        print(banner)

        # ========
        # Setup
        # ========
        self.N_picle_des    = N_picle_des
        self.A1_r_prof      = A1_r_prof
        self.A1_rho_prof    = A1_rho_prof
        self.A1_mat_prof    = A1_mat_prof
        self.A1_u_prof      = A1_u_prof
        self.A1_T_prof      = A1_T_prof
        self.A1_P_prof      = A1_P_prof
        self.do_stretch     = do_stretch
        self.verb           = verb

        # Maximum number of attempts allowed for tweaking particle mass and
        # number of particles in the first shell of an outer layer
        attempt_max = int(max(self.N_picle_des / 1e3, 1e3))

        # Optional profiles
        if self.A1_u_prof is not None:
            self.do_u   = True
        else:
            self.do_u   = False
        if self.A1_T_prof is not None:
            self.do_T   = True
        else:
            self.do_T   = False
        if self.A1_P_prof is not None:
            self.do_P   = True
        else:
            self.do_P   = False

        # Verbosity
        if self.verb >= 1:
            self.verb_options   = {0: "None", 1: "Standard", 2: "Extra",
                                   3: "Debug"}
            print("Verbosity %d: %s printing" %
                  (self.verb, self.verb_options[self.verb]))

        self.N_prof         = len(self.A1_r_prof)
        self.N_shell_tot    = 0

        # Default material IDs if not provided
        if self.A1_mat_prof is None:
            self.A1_mat_prof    = np.zeros(self.N_prof)

        # Values for each shell
        A1_N_shell          = []
        A1_m_shell          = []
        A1_m_picle_shell    = []
        A1_r_shell          = []
        A1_rho_shell        = []
        A1_mat_shell        = []
        if self.do_u:
            A1_u_shell          = []
        if self.do_T:
            A1_T_shell          = []
        if self.do_P:
            A1_P_shell          = []
        # All particle data
        self.A1_m   = []
        self.A1_r   = []
        self.A1_rho = []
        self.A1_mat = []
        self.A1_x   = []
        self.A1_y   = []
        self.A1_z   = []
        if self.do_u:
            self.A1_u   = []
        if self.do_T:
            self.A1_T   = []
        if self.do_P:
            self.A1_P   = []

        # Calculate the mass profile
        self.get_mass_profile()
        # Enclosed mass profile
        self.A1_m_enc_prof  = np.cumsum(self.A1_m_prof)
        self.m_tot          = self.A1_m_enc_prof[-1]
        self.m_picle_des    = self.m_tot / self.N_picle_des

        # Find the radii of all material boundaries (including the outer edge)
        self.find_material_boundaries()
        self.N_layer    = len(self.A1_r_bound)

        # Check the profiles and update them if necessary
        self.check_interp_profiles()

        # Max allowed particle mass
        m_picle_max = self.m_picle_des * 1.01
        # Initial relative particle mass tweak
        dm_picle_init   = 1e-3

        if self.verb >= 1:
            print("\n%d layer(s):" % self.N_layer)
            print("    Outer radius   Mass          Material")
            for r_bound, idx_bound, mat in zip(
                self.A1_r_bound, self.A1_idx_bound, self.A1_mat_layer):

                print("    %5e   %.5e   %d" %
                      (r_bound, self.A1_m_enc_prof[idx_bound], mat))

            print("\n> Divide the profile into shells")

        # ================
        # First (innermost) layer
        # ================
        i_layer = 0
        if self.verb >= 1 and self.N_layer > 1:
            print("\n==== Layer %d ====" % (i_layer + 1))

        # Index of first profile shell in the next layer, or of the final shell
        idx_bound   = self.A1_idx_bound[0] + 1
        if idx_bound == self.N_prof:
            idx_bound   = self.N_prof - 1
        r_bound = self.A1_r_bound[0]

        # ========
        # Vary the particle mass until the particle shell boundary matches the
        # profile boundary
        # ========
        # Start at the maximum allowed particle mass then decrease to fit
        self.m_picle    = m_picle_max
        self.dm_picle   = dm_picle_init
        N_shell_init    = 0

        if self.verb >= 1:
            print("\n> Tweak the particle mass to fix the outer boundary")
        if self.verb == 3:
            header  = "    Attempt  Particle mass   Relative tweak "
            print(header, end='')

        # Tweak the particle mass
        is_done = False
        # This also sets the shell data: A1_idx_outer and A1_r_outer
        for attempt in range(attempt_max + 1):
            if attempt == attempt_max:
                print("\nFailed after %d attempts!" % attempt_max)
                sys.exit()

            if self.verb == 3:
                # No endline so can add more on this line in the loop
                print("\n    %07d  %.5e     %.1e " %
                      (attempt, self.m_picle, self.dm_picle), end='')
                sys.stdout.flush()

            # Find the outer boundary radii of all shells
            A1_idx_outer    = []
            A1_r_outer      = []

            # Set the core dr with the radius containing the mass of the central
            # tetrahedron of 4 particles
            N_picle_shell   = 4
            idx_outer       = np.searchsorted(self.A1_m_enc_prof,
                                              N_picle_shell * self.m_picle)
            r_outer         = self.A1_r_prof[idx_outer]
            self.dr_core    = r_outer

            # Mass-weighted mean density
            self.rho_core   = get_weighted_mean(self.A1_m_prof[:idx_outer],
                                                self.A1_rho_prof[:idx_outer])

            # Record shell boundary
            A1_idx_outer.append(idx_outer)
            A1_r_outer.append(self.dr_core)

            # ========
            # Find the shells that fit in this layer
            # ========
            N_shell = 0
            # Continue until a break from inside, to do properly the final shell
            while N_shell < self.N_prof:
                # Calculate the shell width from the profile density relative to
                # the core radius and density
                rho = self.A1_rho_prof[idx_outer]
                dr  = self.dr_core * np.cbrt(self.rho_core / rho)

                # Find the profile radius just beyond this shell (r_outer + dr)
                idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)

                # Hit outer edge, stop
                if idx_outer >= idx_bound:
                    # Extend the final shell to include the tiny extra bit of
                    # this layer
                    if is_done:
                        A1_idx_outer[-1]    = idx_bound
                        A1_r_outer[-1]      = r_bound

                    break
                r_outer = self.A1_r_prof[idx_outer]

                # Record the shell
                A1_idx_outer.append(idx_outer)
                A1_r_outer.append(r_outer)

                N_shell += 1

            if is_done:
                if self.verb == 3:
                    print("")
                break

            # Number of shells for the starting particle mass
            if N_shell_init == 0:
                N_shell_init    = N_shell

            # ========
            # Reduce the particle mass until one more shell *just* fits
            # ========
            # Not got another shell yet, so reduce the mass
            if N_shell == N_shell_init:
                self.m_picle    *= 1 - self.dm_picle

            # Got one more shell, but need it to *just* fit, so go back one step
            # and retry with smaller mass changes (repeat this twice!)
            elif self.dm_picle > dm_picle_init * 1e-2:
                if self.verb == 3:
                    print("  Reduce tweak", end='')

                self.m_picle    *= 1 + self.dm_picle
                self.dm_picle   *= 1e-1

            # Got one more shell and refined the mass so it just fits, so done!
            else:
                # Repeat one more time to extend the final shell to include the
                # tiny extra bit of this layer
                is_done = True

        if self.verb == 3:
            print(header)
        if self.verb >= 1:
            print("> Done particle mass tweaking!")
        if self.verb >= 2:
            print("    from %.5e to %.5e after %d attempts" %
                  (self.m_picle_des, self.m_picle, attempt))
        if self.verb >= 1:
            print("\n%d shells in layer %d" % (N_shell, i_layer + 1))

        i_layer += 1

        # ================
        # Outer layer(s)
        # ================
        while i_layer < self.N_layer:
            if self.verb >= 1:
                print("\n==== Layer %d ====" % (i_layer + 1))

            # Index of first profile shell in the next layer, or of the edge
            idx_bound   = self.A1_idx_bound[i_layer] + 1
            if idx_bound == self.N_prof:
                idx_bound   = self.N_prof - 1
            r_bound = self.A1_r_bound[i_layer]

            # ========
            # First find the initial number of particles continuing from the
            # previous layer
            # ========
            # Calculate the shell width from the profile density
            # relative to the core radius and density
            idx_inner   = self.A1_idx_bound[i_layer - 1] + 1
            r_inner     = self.A1_r_bound[i_layer - 1]
            rho         = self.A1_rho_prof[idx_inner]
            dr          = self.dr_core * np.cbrt(self.rho_core / rho)

            # Find the profile radius just beyond this shell (r_outer + dr)
            idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)

            # Shell mass and initial number of particles
            m_shell         = sum(self.A1_m_prof[idx_inner:idx_outer])
            N_picle_shell   = int(round(m_shell / self.m_picle))
            N_picle_init    = N_picle_shell

            # ========
            # Vary the number of particles in the first shell of this layer
            # until the particle shell boundary matches the profile boundary
            # ========
            if self.verb >= 1:
                print("\n> Tweak the number of particles in the first shell "
                      "to fix the outer boundary")
            if self.verb == 3:
                header  = "    Attempt  Particles  1st shell width "
                print(header, end='')

            # Initialise
            N_shell_init    = 0
            dN_picle_shell  = 1
            is_done         = False
            # Continue until a break from inside, because we want one more loop
            # after is_done is set True
            for attempt in range(attempt_max + 1):
                if attempt == attempt_max:
                    print("\nFailed after %d attempts!" % attempt_max)
                    sys.exit()

                if self.verb == 3:
                    # No endline so can add more on this line in the loop
                    print("\n    %07d  %07d    " %
                          (attempt, N_picle_shell), end='')

                # Find the outer boundary radii of all shells
                A1_idx_outer_tmp    = []
                A1_r_outer_tmp      = []

                # Set the starting dr by the shell that contains the mass of
                # N_picle_shell particles, instead of continuing to use dr_core
                idx_outer   = idx_inner + np.searchsorted(
                    self.A1_m_enc_prof[idx_inner:]
                    - self.A1_m_enc_prof[idx_inner],
                    N_picle_shell * self.m_picle
                    )
                r_outer     = self.A1_r_prof[idx_outer]
                self.dr_0   = r_outer - r_inner

                if self.verb == 3:
                    print("%.3e" % self.dr_0, end='')
                    sys.stdout.flush()

                # Mass-weighted mean density
                self.rho_0  = get_weighted_mean(
                    self.A1_m_prof[idx_inner:idx_outer],
                    self.A1_rho_prof[idx_inner:idx_outer]
                    )

                # Record shell boundary
                A1_idx_outer_tmp.append(idx_outer)
                A1_r_outer_tmp.append(r_outer)

                # ========
                # Find the shells that fit in this layer
                # ========
                N_shell = 1
                while idx_outer < idx_bound:
                    # Calculate the shell width from the profile density
                    # relative to the first shell in this layer
                    rho = self.A1_rho_prof[idx_outer]
                    dr  = self.dr_0 * np.cbrt(self.rho_0 / rho)

                    # Find the profile radius just beyond this shell (r_outer +
                    # dr)
                    idx_outer   = np.searchsorted(self.A1_r_prof, r_outer + dr)

                    # Hit outer edge, stop
                    if idx_outer >= idx_bound:
                        # Extend the final shell to include the tiny extra bit
                        # of this layer
                        if is_done:
                            A1_idx_outer_tmp[-1]    = idx_bound
                            A1_r_outer_tmp[-1]      = r_bound

                        break
                    r_outer = self.A1_r_prof[idx_outer]

                    # Record the shell
                    A1_idx_outer_tmp.append(idx_outer)
                    A1_r_outer_tmp.append(r_outer)

                    N_shell += 1

                if is_done:
                    if self.verb == 3:
                        print("")
                    break

                # Number of shells for the initial number of particles
                if N_shell_init == 0:
                    N_shell_init    = N_shell

                # ========
                # Change the number of particles in the first shell until either
                # one more shell just fits or just until this number of shells
                # just fits
                # ========
                # Got one more shell, so done!
                if N_shell == N_shell_init + 1:
                    # Repeat one more time to extend the final shell to include
                    # the tiny extra bit of this layer
                    is_done = True

                # Got one less shell, so go back one step then done!
                elif N_shell == N_shell_init - 1:
                    N_picle_shell   -= 1

                    # Repeat one more time to extend the final shell to include
                    # the tiny extra bit of this layer
                    is_done = True

                # Shell number changed by more than +/-1 which shouldn't happen
                elif N_shell != N_shell_init:
                    print("\nN_shell jumped from %d to %d! " %
                          (N_shell_init, N_shell))
                    print("Check that the profile radii steps are dense enough "
                          "for these outer shells... ")
                    sys.exit()

                # Not yet done so vary the number of particles in the first
                # shell (i.e. try: N-1, N+1, N-2, N+2, ...)
                else:
                    N_picle_shell   += dN_picle_shell
                    dN_picle_shell  = (-np.sign(dN_picle_shell)
                                       * (abs(dN_picle_shell) + 1))

            if self.verb == 3:
                print(header)

            # Add these to the previous layer(s)' shells
            A1_idx_outer.append(A1_idx_outer_tmp)
            A1_r_outer.append(A1_r_outer_tmp)

            if self.verb >= 1:
                print("> Done first-shell particle number tweaking!")
            if self.verb >= 2:
                print("    from %d to %d after %d attempts" %
                      (N_picle_init, N_picle_shell, attempt))
            if self.verb >= 1:
                print("\n%d shells in layer %d" % (N_shell, i_layer + 1))

            i_layer += 1

        # Stack all layers' shells together
        A1_idx_outer        = np.hstack(A1_idx_outer)
        A1_r_outer          = np.hstack(A1_r_outer)
        self.N_shell_tot    = len(A1_idx_outer)

        if self.verb >= 1:
            print("\n> Done profile division into shells!")

        if self.verb >= 1:
            print("\n==== Particles ====\n")
            print("> Find the values for the particles in each shell")
        if self.verb >= 2:
            header  = ("    Shell   Radius     i_outer  Number   Mass       "
                       "Density    Material")
            print(header)

        # ================
        # Set the particle values for each shell
        # ================
        idx_inner   = 0
        for i_shell, idx_outer in enumerate(A1_idx_outer):
            # Profile slice for this shell
            A1_m_prof_shell = self.A1_m_prof[idx_inner:idx_outer]

            # Mass
            A1_m_shell.append(np.sum(A1_m_prof_shell))
            # Number of particles
            if i_shell == 0:
                A1_N_shell.append(4)
            else:
                A1_N_shell.append(int(round(A1_m_shell[-1] / self.m_picle)))
            # Actual particle mass
            A1_m_picle_shell.append(A1_m_shell[-1] / A1_N_shell[-1])

            # Radius (mean of half-way and mass-weighted radii)
            r_half  = (self.A1_r_prof[idx_inner]
                       + self.A1_r_prof[idx_outer]) / 2
            r_mw    = get_weighted_mean(A1_m_prof_shell,
                                        self.A1_r_prof[idx_inner:idx_outer])
            A1_r_shell.append((r_half + r_mw) / 2)

            # Other properties
            A1_rho_shell.append(get_weighted_mean(
                A1_m_prof_shell, self.A1_rho_prof[idx_inner:idx_outer]))
            A1_mat_shell.append(self.A1_mat_prof[idx_inner])
            if self.do_u:
                A1_u_shell.append(get_weighted_mean(
                    A1_m_prof_shell, self.A1_u_prof[idx_inner:idx_outer]))
            if self.do_T:
                A1_T_shell.append(get_weighted_mean(
                    A1_m_prof_shell, self.A1_T_prof[idx_inner:idx_outer]))
            if self.do_P:
                A1_P_shell.append(get_weighted_mean(
                    A1_m_prof_shell, self.A1_P_prof[idx_inner:idx_outer]))

            idx_inner   = idx_outer

            if self.verb >= 2:
                print("    %06d  %.3e  %07d  %07d  %.3e  %.3e  %d" %
                      (i_shell, A1_r_shell[-1], idx_outer, A1_N_shell[-1],
                       A1_m_picle_shell[-1], A1_rho_shell[-1],
                       A1_mat_shell[-1]))
                # Print the header again for a new layer
                if idx_outer - 1 in self.A1_idx_bound:
                    print(header)

        if self.verb >= 2:
            print(header)
        if self.verb >= 1:
            extra   = ""
            if self.do_u:
                extra   += "sp. int. energy"
                if self.do_T:
                    if self.do_P:
                        extra   += ", "
                    else:
                        extra   += " and "
            if self.do_T:
                extra   += "temperature"
                if self.do_P:
                    if self.do_u:
                        extra   += ", and "
                    else:
                        extra   += " and "
            if self.do_P:
                extra   += "pressure"
            if any([self.do_u, self.do_T, self.do_P]):
                print("  Also set %s values." % extra)
            else:
                print("  No extra values.")

            print("> Done shell particle values!")

        # ================
        # Generate the particles in each shell
        # ================
        if self.verb >= 1:
            print("\n> Arrange the particles in each shell")

        # Placeholders
        if not self.do_u:
            A1_u_shell  = [None] * self.N_shell_tot
        if not self.do_T:
            A1_T_shell  = [None] * self.N_shell_tot
        if not self.do_P:
            A1_P_shell  = [None] * self.N_shell_tot

        for i, N, m, r, rho, mat, u, T, P in zip(
            np.arange(self.N_shell_tot), A1_N_shell, A1_m_picle_shell,
            A1_r_shell, A1_rho_shell, A1_mat_shell, A1_u_shell, A1_T_shell,
            A1_P_shell):
            if self.verb >= 1:
                # Print progress
                print("\r    Shell %d of %d" % (i+1, self.N_shell_tot), end='')
                sys.stdout.flush()

            self.generate_shell_particles(N, m, r, rho, mat, u=u, T=T, P=P)

        self.flatten_particle_arrays()

        if self.verb >= 1:
            print("\n> Done particles!")

        self.N_picle    = len(self.A1_r)
        if self.verb >= 1:
            print("\nFinal number of particles = %d" % self.N_picle)
            print("\n> SEAGen done!")


    def check_interp_profiles(self):
        """ Check that the profiles are suitable and fix them if not.

            Check the radii are increasing and start from non-zero.

            Interpolate the profiles if either: the radii steps are not dense
            enough to place accurately the central 4 particles; or the outermost
            shell is too massive to determine accurately the outer layer shell
            boundaries.

            Also re-calculate the mass profile and re-find the material
            boundaries after interpolation, if needed.

            Sets:
                A1_r_prof, A1_rho_prof, A1_mat_prof, A1_u_prof, A1_T_prof,
                A1_P_prof, A1_m_prof, A1_m_enc_prof, A1_idx_bound, A1_r_bound,
                A1_mat_layer ([float] or [int])
                    The interpolated profile arrays and re-derived arrays, if
                    interpolation was needed.

                N_prof (int)
                    The updated number of profile steps, if interpolation was
                    needed.
        """
        # Check monotonically increasing radius
        assert(np.all(np.diff(self.A1_r_prof) > 0))

        # Check first radius is positive
        assert(self.A1_r_prof[0] > 0)

        # ========
        # Interpolate profiles if not dense enough in radius
        # ========
        n_interp    = 0

        # Check the first 4 particle masses are enclosed by many profile steps
        idx_min     = int(1e3)
        idx_outer   = np.searchsorted(self.A1_m_enc_prof, 4 * self.m_picle_des)
        if idx_outer < idx_min:
            # Estimate the interpolation needed assuming constant density
            n_interp    = int(np.floor(idx_min / idx_outer) + 1)
            if self.verb >= 2:
                print("\nFirst 4 particle masses bounded by profile index "
                      "idx_outer = %d: n_interp = %d " %
                      (idx_outer, n_interp))

        # Check the outermost profile shell contains a low mass
        m_shell_min = max(self.N_picle_des / 1e4, 1) * self.m_picle_des
        if self.A1_m_prof[-1] > m_shell_min:
            # Estimate the interpolation needed assuming constant density
            n_interp    = max(
                n_interp,
                int(np.floor(self.A1_m_prof[-1] / m_shell_min) + 1)
                )
            if self.verb >= 2:
                if n_interp == 0:
                    print("")
                print("Final profile shell has a mass of m_prof = %.1f "
                      "particles: n_interp = %d " %
                      (self.A1_m_prof[-1] / self.m_picle_des, n_interp))

        # Interpolate if either check failed
        if n_interp > 0:
            print("\n> Interpolating profiles to increase radial density ")
            if self.verb >= 2:
                print("    N_prof = %d,  n_interp = %d " %
                      (self.N_prof, n_interp))

            # Add a zero-radius element to each profile for interpolation
            self.A1_r_prof      = np.append(0, self.A1_r_prof)
            self.A1_rho_prof    = np.append(self.A1_rho_prof[0],
                                            self.A1_rho_prof)
            self.A1_mat_prof    = np.append(self.A1_mat_prof[0],
                                            self.A1_mat_prof)
            if self.do_u:
                self.A1_u_prof  = np.append(self.A1_u_prof[0], self.A1_u_prof)
            if self.do_T:
                self.A1_T_prof  = np.append(self.A1_T_prof[0], self.A1_T_prof)
            if self.do_P:
                self.A1_P_prof  = np.append(self.A1_P_prof[0], self.A1_P_prof)

            # ========
            # Interpolate
            # ========
            # Radii
            A1_idx          = np.arange(self.N_prof + 1)
            A1_idx_interp   = (np.arange((self.N_prof + 1) * n_interp)
                               / n_interp)[:-(n_interp - 1)]
            A1_r_prof_old   = self.A1_r_prof.copy()
            self.A1_r_prof  = np.interp(A1_idx_interp, A1_idx, self.A1_r_prof)
            self.N_prof     = len(self.A1_r_prof)

            # Density
            self.A1_rho_prof    = np.interp(self.A1_r_prof, A1_r_prof_old,
                                            self.A1_rho_prof)

            # Material (can't interpolate the integers)
            self.A1_mat_prof    = np.empty(self.N_prof)
            # Indices of the first profile shells in each layer
            A1_idx  = np.append(0, self.A1_idx_bound[:-1] + 1)
            A1_idx  *= n_interp
            for idx, mat in zip(A1_idx, self.A1_mat_layer):
                self.A1_mat_prof[idx:]  = mat

            # Other profiles
            if self.do_u:
                self.A1_u_prof  = np.interp(self.A1_r_prof, A1_r_prof_old,
                                            self.A1_u_prof)
            if self.do_T:
                self.A1_T_prof  = np.interp(self.A1_r_prof, A1_r_prof_old,
                                            self.A1_T_prof)
            if self.do_P:
                self.A1_P_prof  = np.interp(self.A1_r_prof, A1_r_prof_old,
                                            self.A1_P_prof)

            # Remove the zero-radius elements
            self.A1_r_prof      = self.A1_r_prof[1:]
            self.A1_rho_prof    = self.A1_rho_prof[1:]
            self.A1_mat_prof    = self.A1_mat_prof[1:]
            if self.do_u:
                self.A1_u_prof  = self.A1_u_prof[1:]
            if self.do_T:
                self.A1_T_prof  = self.A1_T_prof[1:]
            if self.do_P:
                self.A1_P_prof  = self.A1_P_prof[1:]
            self.N_prof         = len(self.A1_r_prof)

            # Re-calculate the mass profile
            self.get_mass_profile()
            # Enclosed mass profile
            self.A1_m_enc_prof  = np.cumsum(self.A1_m_prof)

            # Re-find the radii of all material boundaries
            self.find_material_boundaries()

            print("> Done interpolating profiles! ")

            if self.verb >= 2:
                idx_outer   = np.searchsorted(self.A1_m_enc_prof,
                                              4 * self.m_picle_des)
                print("    N_prof = %d,  first idx_outer = %d,  "
                      "final m_prof = %.1f particles " %
                      (self.N_prof, idx_outer,
                       self.A1_m_prof[-1] / self.m_picle_des))

        return


    def get_mass_profile(self):
        """ Calculate the mass profile from the density profile.

            Sets:
                A1_m_prof ([float])
                    The array of mass at each profile radius.
        """
        # Find the mass in each profile shell, starting with the central sphere
        self.A1_m_prof      = np.empty(self.N_prof)
        self.A1_m_prof[0]   = get_shell_mass(
            0.0, self.A1_r_prof[0], self.A1_rho_prof[0])
        self.A1_m_prof[1:]  = get_shell_mass(
            self.A1_r_prof[:-1], self.A1_r_prof[1:], self.A1_rho_prof[1:])

        return


    def find_material_boundaries(self):
        """ Find the radii of any layer boundaries, including the outer edge.

            Sets:
                A1_idx_bound ([int])
                    The indices of boundaries in the profile, i.e. the index of
                    the last profile shell in each layer.

                A1_r_bound ([float])
                    The radii of boundaries in the profile.

                A1_mat_layer ([int])
                    The material identifiers in each layer.
        """
        # Find where the material ID changes
        A1_idx_bound    = np.where(np.diff(self.A1_mat_prof) != 0)[0]

        # Include the outer edge
        self.A1_idx_bound   = np.append(A1_idx_bound, self.N_prof - 1)

        self.A1_r_bound     = self.A1_r_prof[self.A1_idx_bound]

        self.A1_mat_layer   = self.A1_mat_prof[self.A1_idx_bound]

        return


    def get_tetrahedron_points(self, r):
        """ Calculate the positions of the vertices of a tetrahedron.

            Args:
                r (float)
                    The radius.

            Returns:
                A1_x, A1_y, A1_z ([float])
                    The cartesian coordinates of the four vertices.
        """
        # Radius scale
        r_scale = r / np.sqrt(3)
        # Tetrahedron vertex coordinates
        A1_x    = np.array([1, 1, -1, -1]) * r_scale
        A1_y    = np.array([1, -1, 1, -1]) * r_scale
        A1_z    = np.array([1, -1, -1, 1]) * r_scale

        return A1_x, A1_y, A1_z


    def generate_shell_particles(self, N, m, r, rho, mat,
                                 u=None, T=None, P=None):
        """ Make a single spherical shell of particles.

            Args:
                N (int)
                    The number of particles.

                m (float)
                    The particle mass.

                r (float)
                    The radius.

                rho (float)
                    The density..

                u, T, P (float or None)
                    The sp. int. energy, temperature, and pressure. Or None if
                    not provided.

            Sets:
                Appends the args to A1_m, A1_r, A1_rho, A1_mat.
                And to A1_u, A1_T, A1_P, if provided.

                Appends the particle positions to A1_x, A1_y, A1_z.
        """
        # Append the data to the all-particle arrays
        self.A1_m.append([m] * N)
        self.A1_r.append([r] * N)
        self.A1_rho.append([rho] * N)
        self.A1_mat.append([mat] * N)
        if u is not None:
            self.A1_u.append([u] * N)
        if T is not None:
            self.A1_T.append([T] * N)
        if P is not None:
            self.A1_P.append([P] * N)

        # Make a tetrahedron for the central 4 particles
        if N == 4:
            A1_x, A1_y, A1_z    = self.get_tetrahedron_points(r)
            self.A1_x.append(A1_x)
            self.A1_y.append(A1_y)
            self.A1_z.append(A1_z)
        else:
            shell = GenShell(N, r, do_stretch=self.do_stretch)

            self.A1_x.append(shell.A1_x)
            self.A1_y.append(shell.A1_y)
            self.A1_z.append(shell.A1_z)

        return


    def flatten_particle_arrays(self):
        """ Flatten the particle data arrays for output.

            Sets:
                A1_x, A1_y, A1_z, A1_r, A1_m, A1_rho, A1_mat,
                and A1_u, A1_T, A1_P if provided.
                    See __init__()'s documentation.
        """
        for array in ["r", "x", "y", "z", "m", "rho", "mat"]:
            exec("self.A1_%s = np.hstack(self.A1_%s)" % (array, array))

        for array in ["u", "T", "P"]:
            if eval("self.do_" + array):
                exec("self.A1_%s = np.hstack(self.A1_%s)" % (array, array))

        return


    # Aliases for the main outputs without my array notation
    @property
    def x(self):
        return self.A1_x
    @property
    def y(self):
        return self.A1_y
    @property
    def z(self):
        return self.A1_z
    @property
    def r(self):
        return self.A1_r
    @property
    def m(self):
        return self.A1_m
    @property
    def rho(self):
        return self.A1_rho
    @property
    def mat(self):
        return self.A1_mat
    @property
    def u(self):
        return self.A1_u
    @property
    def T(self):
        return self.A1_T
    @property
    def P(self):
        return self.A1_P


# //////////////////////////////////////////////////////////////////////////// #
#                               III. Main                                      #
# //////////////////////////////////////////////////////////////////////////// #

if __name__ == '__main__':
    print(banner)
