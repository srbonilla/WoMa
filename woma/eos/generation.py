"""
WoMa equations of state table generation.
"""
import sys
import numpy as np
from scipy.optimize import minimize_scalar

from woma.misc import utils as ut
from woma.misc import glob_vars as gv
from woma.eos.sesame import find_index_and_interp

# Gas constant (J K^-1 mol^-1)
R_gas = 8.3145

# Pressure units
bar_to_Pa = 1e5
Pa_to_bar = 1 / bar_to_Pa  # = 1e-5 bar
Mbar_to_Pa = bar_to_Pa * 1e6  # = 1e11 Pa
Pa_to_Mbar = 1 / Mbar_to_Pa  # = 1e-11 Mbar
bar_to_Ba = bar_to_Pa * ut.SI_to_cgs.P  # = 1e6 Ba
Ba_to_bar = 1 / bar_to_Ba  # = 1e-6 bar
Mbar_to_Ba = Mbar_to_Pa * ut.SI_to_cgs.P  # = 1e12 Ba
Ba_to_Mbar = 1 / Mbar_to_Ba  # = 1e-12 Mbar


# ========
# Hubbard & MacFarlane (1980)
# ========
class Material_HM80:
    """Hubbard & MacFarlane (1980) materials.

    Some attributes apply to only hydrogen--helium or to only the ice and rock.

    Note: HM80's EoS parameter units are cgs and Mbar, not SI.

    Parameters
    ----------
    name : str
        The material name.

    version_date : int
        The file version date (YYYYMMDD).

    rho_0 : float
        Reference density (kg m^-3)

    rho_min, rho_max, num_rho, u_min, u_max, num_u : float
        Minimum, maximum, and number of values for the 1D density (kg m^-3) and
        sp. int. energy (J kg^-1) arrays.

    T_max : float
        Maximum temperature (K) for root-finding.

    A1_p : [float]
        Coefficients for the pressure (HM80 table 1 or Eqns. (4) or (5)).

    A1_c : [float]
        Coefficients for the specific heat capacity (HM80 table 1).

    m_mol : float
        The molar mass (g mol^-1).

    A1_m_mol, A1_abun, A1_nu, f_nu : [float] or [int]
        For each component molecule: the molar mass (g mol^-1),
        fractional abundance, number of atoms, and factor for C_V.

    bulk_mod : float
        The bulk modulus (Pa) for estimating the sound speed.

    P_min_for_c_min : float
        A pressure (Pa) for setting the minimum sound speed.
    """

    def __init__(
        self,
        name,
        version_date,
        rho_0,
        rho_min,
        rho_max,
        num_rho,
        u_min,
        u_max,
        num_u,
        T_max,
        A1_p,
        A1_c=None,
        m_mol=None,
        A1_m_mol=None,
        A1_abun=None,
        A1_nu=None,
        f_nu=None,
        bulk_mod=None,
        P_min_for_c_min=None,
    ):
        self.name = name
        self.version_date = version_date
        self.rho_0 = rho_0
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.num_rho = num_rho
        self.u_min = u_min
        self.u_max = u_max
        self.num_u = num_u
        self.A1_p = np.array(A1_p)
        self.T_max = T_max

        # Derived
        self.log_rho_min = np.log(rho_min)
        self.log_rho_max = np.log(rho_max)
        self.log_rho_step = (self.log_rho_max - self.log_rho_min) / (self.num_rho - 1)
        self.log_rho_step_inv = 1 / self.log_rho_step
        self.log_u_min = np.log(u_min)
        self.log_u_max = np.log(u_max)
        self.log_u_step = (self.log_u_max - self.log_u_min) / (self.num_u - 1)
        self.log_u_step_inv = 1 / self.log_u_step

        if self.name == "HM80_HHe":
            # Atmosphere only
            self.A1_c = np.array(A1_c)
            self.m_mol = m_mol
            self.P_min_for_c_min = P_min_for_c_min
        else:
            # Not-atmosphere only
            self.A1_abun = np.array(A1_abun)
            self.A1_nu = np.array(A1_nu)
            self.f_nu = f_nu
            self.m_mol = np.sum(np.array(A1_m_mol) * A1_abun)
            self.bulk_mod = bulk_mod

            # Cold curve (placeholder)
            self.num_cold = int(1e3)
            self.A1_rho_cold = None
            self.A1_u_cold = None

    def C_V(self, rho=None, T=None):
        """Return the specific heat capacity with a possible dependence on
        density and temperature.

        Parameters
        ----------
        rho : float
            Density (kg m^-3)

        T : float
            Temperature (K)

        Returns:
            C_V : float
                Specific heat capacity (J kg^-1 K^-1)
        """
        if self.name == "HM80_HHe":
            assert rho != None and T != None
            # Convert to cgs for HM80's units
            rho_cgs = rho * ut.SI_to_cgs.rho

            C_V = (
                (
                    self.A1_c[0]
                    + self.A1_c[1] * T
                    + self.A1_c[2] * T**2
                    + self.A1_c[3] * T * rho_cgs
                    + self.A1_c[4] * rho_cgs
                    + self.A1_c[5] * rho_cgs**2
                )
                * R_gas
                * ut.SI_to_cgs.E
                / self.m_mol
            )

            return C_V * ut.cgs_to_SI.u
        else:
            # No rho or T dependence
            C_V = (
                np.sum(self.A1_abun * self.A1_nu)
                * self.f_nu
                * R_gas
                * ut.SI_to_cgs.E
                / self.m_mol
            )

            return C_V * ut.cgs_to_SI.u

    def P_rho_T(self, rho, T):
        """Compute pressure as a function of density and temperature."""
        # Convert for HM80's units
        rho_cgs = rho * ut.SI_to_cgs.rho

        if self.name == "HM80_HHe":
            x = np.log(rho / self.rho_0)
            y = np.log(T)
            P = np.exp(
                self.A1_p[0]
                + self.A1_p[1] * y
                + self.A1_p[2] * y**2
                + self.A1_p[3] * x * y
                + self.A1_p[4] * x
                + self.A1_p[5] * x**2
                + self.A1_p[6] * x**3
                + self.A1_p[7] * x**2 * y
            )
            P *= Mbar_to_Pa
        else:
            # Cold pressure (0 K)
            P = rho_cgs ** self.A1_p[0] * np.exp(
                self.A1_p[1] + self.A1_p[2] * rho_cgs + self.A1_p[3] * rho_cgs**2
            )
            P *= Mbar_to_Pa

            # Add thermal pressure
            P += self.C_V() * rho * T

        return P

    def compute_cold_curve(self):
        """Compute the 'cold curve' densities and sp. int. energies at T=0 K.

        Sets
        ----
        A1_rho_cold : [float]
            Array of densities (kg m^-3).

        A1_u_cold : [float]
            Array of specific (cold) internal energies (J kg^-1).
        """
        # Densities
        # Evenly spaced in log(density) with rho_0 as one of the elements
        self.log_rho_cold_step = (self.log_rho_max - self.log_rho_min) / self.num_cold
        self.log_rho_cold_step_inv = 1 / self.log_rho_cold_step

        # Append densities from rho_0 until rho_min and rho_max are included
        A1_rho_cold = [self.rho_0]

        # Up to rho_max
        log_rho = np.log(self.rho_0)
        while log_rho < self.log_rho_max:
            log_rho += self.log_rho_cold_step

            A1_rho_cold.append(np.exp(log_rho))

        # Down to rho_min
        log_rho = np.log(self.rho_0) - self.log_rho_cold_step
        while log_rho > self.log_rho_min:
            log_rho -= self.log_rho_cold_step

            A1_rho_cold.append(np.exp(log_rho))

        A1_rho_cold = np.sort(np.array(A1_rho_cold))
        self.num_rho_cold = len(A1_rho_cold)

        # Energies
        # Integrate to find u_cold(rho):
        #   u_cold(rho_0) == 0
        #   du_cold / drho = P(rho, T=0) / rho^2
        A1_u_cold = np.zeros(self.num_rho_cold)
        i_rho_0 = np.searchsorted(A1_rho_cold, self.rho_0)

        # Integrate up from rho_0 to rho_max
        u = A1_u_cold[i_rho_0]
        rho_prv = A1_rho_cold[i_rho_0]
        for i_rho in range(i_rho_0 + 1, self.num_rho_cold, 1):
            rho = A1_rho_cold[i_rho]
            drho = rho - rho_prv

            P = self.P_rho_T(rho, 0)
            du = P * rho**-2 * drho
            u += du
            rho_prv = rho

            A1_u_cold[i_rho] = u

        # Integrate down from rho_0 to rho_min
        u = A1_u_cold[i_rho_0]
        rho_prv = A1_rho_cold[i_rho_0]
        for i_rho in range(i_rho_0 - 1, -1, -1):
            rho = A1_rho_cold[i_rho]
            drho = rho - rho_prv

            P = self.P_rho_T(rho, 0)
            du = P * rho**-2 * drho
            u += du
            rho_prv = rho

            A1_u_cold[i_rho] = u

        self.A1_rho_cold = A1_rho_cold
        self.A1_u_cold = A1_u_cold

    def u_cold(self, rho):
        """Compute the specific internal energy at 0 K."""
        if self.name == "HM80_HHe":
            return 0
        else:
            # Create the u_cold table if necessary
            if self.A1_rho_cold is None:
                self.compute_cold_curve()

            # Interpolate to find u_cold(rho)
            idx_rho = int(
                np.floor((np.log(rho) - self.log_rho_min) * self.log_rho_cold_step_inv)
            )

            # Check if outside the table
            if idx_rho == -1:
                idx_rho = 0
            elif idx_rho >= self.num_cold - 1:
                idx_rho = self.num_cold - 2

            intp = (
                np.log(rho) - np.log(self.A1_rho_cold[idx_rho])
            ) * self.log_rho_cold_step_inv

            u_cold = (1 - intp) * self.A1_u_cold[idx_rho] + intp * self.A1_u_cold[
                idx_rho + 1
            ]

            return u_cold

    def u_rho_T(self, rho, T):
        """Compute the sp. internal energy at this density and temperature.

        Parameters
        ----------
        rho : float
            Density (kg m^-3).

        T : float
            Temperature (K).

        Returns
        -------
        u : float
            Specific internal energy (J kg^-1).
        """
        if self.name == "HM80_HHe":
            # Convert to cgs for HM80's units
            rho_cgs = rho * ut.SI_to_cgs.rho

            # Integration: u = int_0^T C_V(rho, T') dT'
            u = (
                (
                    self.A1_c[0]
                    + self.A1_c[1] * T / 2
                    + self.A1_c[2] * T**2 / 3
                    + self.A1_c[3] * T / 2 * rho_cgs
                    + self.A1_c[4] * rho_cgs
                    + self.A1_c[5] * rho_cgs**2
                )
                * T
                * R_gas
                * ut.SI_to_cgs.E
                / self.m_mol
            )

            u *= ut.cgs_to_SI.u

        else:
            u = self.u_cold(rho) + self.C_V() * T

        return u

    def T_rho_u(self, rho, u):
        """Compute the temperature at this density and sp. internal energy.

        Simply do a crude root-find.

        Parameters
        ----------
        rho : float
            Density (kg m^-3).

        u : float
            Specific internal energy (J kg^-1).

        Returns
        -------
        T : float
            Temperature (K).
        """

        def find_u_T(T):
            """Function for minimize_scalar() to find T s.t. u_rho_T(rho, T) = u."""
            return abs(self.u_rho_T(rho, T) - u)

        T_min = 0
        T_max = self.T_max
        T_mid_low = 1e-5
        T_mid_hig = T_max * (1 - 1e-5)
        try:
            res = minimize_scalar(find_u_T, bracket=[T_min, T_mid_low, T_max])

        except ValueError:
            try:
                res = minimize_scalar(find_u_T, bracket=[T_min, T_mid_hig, T_max])
            except ValueError:
                print(
                    "\n # %s:%s:  Failed to find T for u(rho, T) "
                    % (os.path.basename(__file__), sys._getframe().f_lineno)
                )
                print(" u       = %.5e J kg^-1 " % u)
                print(" rho     = %.5e kg m^-3 " % rho)
                print(
                    " T       = %.5e, %.5e, %.5e, %.5e K "
                    % (T_min, T_mid_low, T_mid_hig, T_max)
                )
                print(
                    " --> u   = %.5e, %.5e, %.5e, %.5e J kg^-1 "
                    % (
                        self.u_rho_T(rho, T_min),
                        self.u_rho_T(rho, T_mid_low),
                        self.u_rho_T(rho, T_mid_hig),
                        self.u_rho_T(rho, T_max),
                    )
                )
                exit()

        T = res.x

        return T

    def write_table(self, Fp_table):
        """Generate and save the table file.

        File contents
        -------------
        # header (11 lines)
        version_date                                            (YYYYMMDD)
        log_rho_min  log_rho_max  num_rho  log_u_min  log_u_max  num_u
        P[0,0]         P[0,1]   ...   P[0,num_u]                (Pressures, Pa)
        P[1,0]         ...      ...   P[1,num_u]
        ...            ...      ...   ...
        P[num_rho,0]   ...      ...   P[num_rho,num_u]
        T[0,0]         T[0,1]   ...   T[0,num_u]                (Temperatures, K)
        T[1,0]         ...      ...   T[1,num_u]
        ...            ...      ...   ...
        T[num_rho,0]   ...      ...   T[num_rho,num_u]

        Parameters
        ----------
        Fp_table : str
            The table file path.
        """
        print('Saving to "%s"... ' % Fp_table[-48:])

        # Initialise the arrays
        A1_rho = np.exp(np.linspace(self.log_rho_min, self.log_rho_max, self.num_rho))
        A1_u = np.exp(np.linspace(self.log_u_min, self.log_u_max, self.num_u))
        A2_P = np.empty((self.num_rho, self.num_u))
        A2_T = np.empty((self.num_rho, self.num_u))

        # Fill the arrays
        for i_rho, rho in enumerate(A1_rho):
            # Print progress
            print("\ri_rho = %d of %d " % (i_rho, self.num_rho - 1), end="")
            sys.stdout.flush()
            u_cold = self.u_cold(rho)

            for i_u, u in enumerate(A1_u):
                # Check for energies that are below the T=0 K energy
                if u < u_cold:
                    A2_T[i_rho, i_u] = 0
                    A2_P[i_rho, i_u] = self.P_rho_T(rho, 0)
                    continue

                A2_T[i_rho, i_u] = self.T_rho_u(rho, u)
                A2_P[i_rho, i_u] = self.P_rho_T(rho, A2_T[i_rho, i_u])

        # Write to file
        with open(Fp_table, "w") as f:
            # Header
            f.write("# Material %s\n" % self.name)
            f.write(
                "# version_date                                            (YYYYMMDD)\n"
                "# log_rho_min  log_rho_max  num_rho  log_u_min  log_u_max  num_u\n"
                "# P[0,0]         P[0,1]   ...   P[0,num_u]                (Pressures, Pa)\n"
                "# P[1,0]         ...      ...   P[1,num_u]\n"
                "# ...            ...      ...   ...\n"
                "# P[num_rho,0]   ...      ...   P[num_rho,num_u]\n"
                "# T[0,0]         T[0,1]   ...   T[0,num_u]                (Temperatures, K)\n"
                "# T[1,0]         ...      ...   T[1,num_u]\n"
                "# ...            ...      ...   ...\n"
                "# T[num_rho,0]   ...      ...   T[num_rho,num_u]\n"
            )

            # Metadata
            f.write("%d \n" % self.version_date)
            f.write(
                "%e %e %d %e %e %d \n"
                % (
                    self.log_rho_min,
                    self.log_rho_max,
                    self.num_rho,
                    self.log_u_min,
                    self.log_u_max,
                    self.num_u,
                )
            )

            # Pressures
            for i_rho, rho in enumerate(A1_rho):
                for i_u, u in enumerate(A1_u):
                    f.write("%e " % A2_P[i_rho, i_u])
                f.write("\n")

            # Temperatures
            for i_rho, rho in enumerate(A1_rho):
                for i_u, u in enumerate(A1_u):
                    f.write("%e " % A2_T[i_rho, i_u])
                f.write("\n")

        print("Done")


# Assume H2-He mass fraction x = 0.75 = 2*n_H2 / (2*n_H2 + 4*n_He) --> ratio:
n_H2_n_He = 2 / (1 / 0.75 - 1)
m_mol_HHe = (2 * n_H2_n_He + 4) / (n_H2_n_He + 1)  # = 2.2857 g mol^-1

mat_HM80_HHe = Material_HM80(
    "HM80_HHe",
    version_date=20220822,
    rho_0=5,
    rho_min=1e-1,
    rho_max=1e3,
    num_rho=100,
    u_min=1e4,
    u_max=5e9,
    num_u=100,
    T_max=1e6,
    A1_p=[
        -16.05895,
        1.22808,
        -0.0217930,
        0.141021,
        0.147156,
        0.277708,
        0.0455347,
        -0.0558596,
    ],
    A1_c=[2.3638, -4.9842e-5, 1.1788e-8, -3.8101e-4, 2.6182, 0.45053],
    m_mol=m_mol_HHe,
    P_min_for_c_min=1e3,
)
mat_HM80_ice = Material_HM80(
    "HM80_ice",
    version_date=20220822,
    rho_0=1000,
    rho_min=1e0,
    rho_max=6e3,
    num_rho=100,
    u_min=1e3,
    u_max=5e9,
    num_u=100,
    T_max=2e6,
    A1_p=[4.067, -3.097, -0.228, -0.0102],
    A1_abun=[0.565, 0.325, 0.11],  # H20, CH4, NH3
    A1_nu=[3, 5, 4],
    A1_m_mol=[18, 18, 18],
    f_nu=2.067,
    bulk_mod=2.0e9,
)
mat_HM80_rock = Material_HM80(
    "HM80_rock",
    version_date=20220822,
    rho_0=3000,
    rho_min=1e0,
    rho_max=2e4,
    num_rho=100,
    u_min=1e4,
    u_max=1e9,
    num_u=100,
    T_max=2e6,
    A1_p=[14.563, -15.041, -2.130, 0.0483],
    A1_abun=[0.38, 0.25, 0.25, 0.12],  # SiO, MgO, FeS, FeO
    A1_nu=[3, 2, 2, 2],
    A1_m_mol=[44, 40, 88, 72],
    f_nu=3,
    bulk_mod=3.49e10,  # Vince: m_mol=63.4,
)


def write_all_HM80_tables():
    mat_HM80_HHe.write_table(gv.Fp_HM80_HHe)
    mat_HM80_ice.write_table(gv.Fp_HM80_ice)
    mat_HM80_rock.write_table(gv.Fp_HM80_rock)


# ========
# SESAME and SESAME-style
# ========
def write_table_SESAME(
    Fp_table, name, version_date, A1_rho, A1_T, A2_u, A2_P, A2_c, A2_s
):
    """Write the data to a file, in a SESAME-like format plus header info, etc.

    File contents
    -------------
    # header (12 lines)
    version_date                                                (YYYYMMDD)
    num_rho  num_T
    rho[0]   rho[1]  ...  rho[num_rho]                          (kg/m^3)
    T[0]     T[1]    ...  T[num_T]                              (K)
    u[0, 0]                 P[0, 0]     c[0, 0]     s[0, 0]     (J/kg, Pa, m/s, J/K/kg)
    u[1, 0]                 ...         ...         ...
    ...                     ...         ...         ...
    u[num_rho-1, 0]         ...         ...         ...
    u[0, 1]                 ...         ...         ...
    ...                     ...         ...         ...
    u[num_rho-1, num_T-1]   ...         ...         s[num_rho-1, num_T-1]

    Parameters
    ----------
    Fp_table : str
        The table file path.

    name : str
        The material name.

    version_date : int
        The file version date (YYYYMMDD).

    A1_rho, A1_T : [float]
        Density (kg m^-3) and temperature (K) arrays.

    A2_u, A2_P, A2_c, A2_s : [[float]]
        Table arrays of sp. int. energy (J kg^-1), pressure (Pa), sound speed
        (m s^-1), and sp. entropy (J K^-1 kg^-1).
    """
    Fp_table = ut.check_end(Fp_table, ".txt")
    num_rho = len(A1_rho)
    num_T = len(A1_T)

    with open(Fp_table, "w") as f:
        # Header
        f.write("# Material %s\n" % name)
        f.write(
            "# version_date                                                (YYYYMMDD)\n"
            "# num_rho  num_T\n"
            "# rho[0]   rho[1]  ...  rho[num_rho-1]                        (kg/m^3)\n"
            "# T[0]     T[1]    ...  T[num_T-1]                            (K)\n"
            "# u[0, 0]                 P[0, 0]     c[0, 0]     s[0, 0]     (J/kg, Pa, m/s, J/K/kg)\n"
            "# u[1, 0]                 ...         ...         ...\n"
            "# ...                     ...         ...         ...\n"
            "# u[num_rho-1, 0]         ...         ...         ...\n"
            "# u[0, 1]                 ...         ...         ...\n"
            "# ...                     ...         ...         ...\n"
            "# u[num_rho-1, num_T-1]   ...         ...         s[num_rho-1, num_T-1]\n"
        )

        # Metadata
        f.write("%d \n" % version_date)
        f.write("%d %d \n" % (num_rho, num_T))

        # Density and temperature arrays
        for i_rho in range(num_rho):
            f.write("%.8e " % A1_rho[i_rho])
        f.write("\n")
        for i_T in range(num_T):
            f.write("%.8e " % A1_T[i_T])
        f.write("\n")

        # Table arrays
        for i_T in range(num_T):
            for i_rho in range(num_rho):
                f.write(
                    "%.8e %.8e %.8e %.8e \n"
                    % (
                        A2_u[i_rho, i_T],
                        A2_P[i_rho, i_T],
                        A2_c[i_rho, i_T],
                        A2_s[i_rho, i_T],
                    )
                )


# ========
# Chabrier, Mazevet and Soubiran (2019) and Chabrier and Debras (2021)
# ========
class Material_Chabrier_HHe:
    """Chabrier, Mazevet and Soubiran (2019) and Chabrier and Debras (2021) Hydrogen-Helium.

    To generate pure H or He EoS, use Chabrier, Mazevet and Soubiran (2019) tables.

    To generate a H-He mixture, use the updated Chabrier and Debras (2021) H table.

    Parameters
    ----------
    version_date : int
        The file version date (YYYYMMDD).

    Y : float
        Helium mass fraction 0 <= Y <= 1

    input_format : str
        Either "Trho" or "TP" depending on format of input table(s)

    Fp_H : str
        File path for Hydrogen EoS table

    Fp_He : str
        File path for Helium EoS table

    """

    def __init__(self, version_date, Y, input_format, Fp_H=None, Fp_He=None):
        self.version_date = version_date
        self.Y = Y
        self.input_format = input_format

        if input_format == "Trho":
            if Y == 0:
                self.Fp_H = Fp_H
            elif Y == 1:
                self.Fp_He = Fp_He
            else:
                raise ValueError(
                    'input_format "Trho" is only compatible with Y = 0 or 1'
                )
        elif input_format == "TP":
            if Y == 0:
                self.Fp_H = Fp_H
            elif Y == 1:
                self.Fp_He = Fp_He
            else:
                assert Fp_H is not None and Fp_He is not None
                assert 0 <= Y <= 1
                self.Fp_H = Fp_H
                self.Fp_He = Fp_He
        else:
            raise ValueError('input_format must have value "Trho" or "TP"')

    def load_table_Trho(self, file):

        """Load and return the table file data from table in Trho format.

        Based on format of the downloadable tables from CMS19 and CD21

        Parameters
        ----------
        file : str
            The table file path.

        Returns
        -------
        A1_rho, A1_T : [float]
            Density (kg m^-3) and temperature (K) arrays.

        A2_u, A2_P, A2_c, A2_s : [[float]]
            Table arrays of sp. int. energy (J kg^-1), pressure (Pa), sound speed
            (m s^-1), and sp. entropy (J K^-1 kg^-1).
        """

        # These are hardcoded for the downloadable tables
        num_T = 121
        num_rho = 241  # for HHe table 0.275
        # num_rho = 281 # for H table

        # Load
        with open(file) as f:

            f.readline()

            A1_T = np.empty(num_T)
            A1_rho = np.empty(num_rho)
            A2_P = np.empty((num_rho, num_T))
            A2_u = np.empty((num_rho, num_T))
            A2_s = np.empty((num_rho, num_T))
            A2_c = np.empty((num_rho, num_T))

            for i_T in range(num_T):
                f.readline()
                for i_rho in range(num_rho):
                    (
                        A1_T[i_T],
                        A2_P[i_rho, i_T],
                        A1_rho[i_rho],
                        A2_u[i_rho, i_T],
                        A2_s[i_rho, i_T],
                        dlrho_dlT,
                        dlrho_dlP,
                        dls_dlT,
                        dls_dlP,
                        div_ad,
                    ) = np.array(f.readline().split(), dtype=float)

                    # Calculate sounds speed
                    XP = 1 / dlrho_dlP
                    XT = -dlrho_dlT / dlrho_dlP
                    rho = 10 ** A1_rho[i_rho] * 10**3
                    P = 10 ** A2_P[i_rho, i_T] * 10**9
                    c = 1 / ((rho / (XP * P)) * (1 - XT * div_ad))

                    # If negative, set to 0. This is only at edge of table
                    if c < 0:
                        c = 0

                    A2_c[i_rho, i_T] = np.sqrt(c)

            # partial P / partial rho at fixed T must be >= 0
            for i_T in range(num_T):
                for i_rho in range(num_rho - 1):
                    if A2_P[i_rho + 1, i_T] < A2_P[i_rho, i_T]:
                        A2_P[i_rho + 1, i_T] = A2_P[i_rho, i_T]

        # Convert from log10 and convert units
        A1_rho = 10 ** (A1_rho) * 10**3
        A1_T = 10 ** (A1_T)
        A2_P = 10 ** (A2_P) * 10**9
        A2_u = 10 ** (A2_u) * 10**6
        A2_s = 10 ** (A2_s) * 10**6

        return A1_T, A1_rho, A2_P, A2_u, A2_s, A2_c

    def load_table_TP(self, file):

        """Load and return the table file data from table in TP format.

        Based on format of the downloadable tables from CMS19 and CD21

        Parameters
        ----------
        file : str
            The table file path.

        Returns
        -------
        A1_P, A1_T : [float]
            Pressure (Pa) and temperature (K) arrays.

        A2_u, A2_rho, A2_c, A2_s : [[float]]
            Table arrays of sp. int. energy (J kg^-1), density (kg m^-3), sound speed
            (m s^-1), and sp. entropy (J K^-1 kg^-1).
        """
        # These are hardcoded for the downloadable tables
        num_T = 121
        num_P = 441
        # Load
        with open(file) as f:

            f.readline()
            if 0 < self.Y < 1 and file == self.Fp_H:
                f.readline()
                f.readline()

            A1_P = np.empty(num_P)
            A1_T = np.empty(num_T)
            A2_rho = np.empty((num_T, num_P))
            A2_u = np.empty((num_T, num_P))
            A2_s = np.empty((num_T, num_P))
            A2_c = np.empty((num_T, num_P))

            for i_T in range(num_T):
                f.readline()
                for i_P in range(num_P):
                    (
                        A1_T[i_T],
                        A1_P[i_P],
                        A2_rho[i_T, i_P],
                        A2_u[i_T, i_P],
                        A2_s[i_T, i_P],
                        dlrho_dlT,
                        dlrho_dlP,
                        dls_dlT,
                        dls_dlP,
                        div_ad,
                    ) = np.array(f.readline().split(), dtype=float)

                    # Calculate sounds speed
                    XP = 1 / dlrho_dlP
                    XT = -dlrho_dlT / dlrho_dlP
                    rho = 10 ** A2_rho[i_T, i_P] * 10**3
                    P = 10 ** A1_P[i_P] * 10**9
                    c = 1 / ((rho / (XP * P)) * (1 - XT * div_ad))

                    # If negative, set to 0. This is only at edge of table
                    if c < 0:
                        c = 0

                    A2_c[i_T, i_P] = np.sqrt(c)

        # Convert from log10 and convert units
        A2_rho = 10 ** (A2_rho) * 10**3
        A1_T = 10 ** (A1_T)
        A1_P = 10 ** (A1_P) * 10**9
        A2_u = 10 ** (A2_u) * 10**6
        A2_s = 10 ** (A2_s) * 10**6

        return A1_T, A1_P, A2_rho, A2_u, A2_s, A2_c

    def additive_volume_law(self):

        """Combine H and He tables at same T and P to make a H-He mix

        Returns
        -------
        A1_P, A1_T : [float]
            Pressure (Pa) and temperature (K) arrays.

        A2_u, A2_rho, A2_c, A2_s : [[float]]
            Table arrays of sp. int. energy (J kg^-1), density (kg m^-3), sound speed
            (m s^-1), and sp. entropy (J K^-1 kg^-1).
        """

        assert self.input_format == "TP"

        assert (self.A1_T_H_TP == self.A1_T_He_TP).all()
        assert (self.A1_P_H_TP == self.A1_P_He_TP).all()

        # Masses of hydrogen and helium (kg)
        m_H = 1.6735575e-27
        m_He = 6.6464731e-27
        # Boltzmann const (m^2 kg s^-2 K^-2)
        k = 1.38064852e-23

        Y = self.Y

        A1_T = self.A1_T_H_TP.copy()
        A1_P = self.A1_P_H_TP.copy()
        # Combine H and He tables to get a table for HHe at Y in TP format
        A2_rho = 1 / ((1 - Y) / self.A2_rho_H_TP + Y / self.A2_rho_He_TP)

        # partial P / partial rho at fixed T must be >= 0
        for i, T in enumerate(A1_T):
            for j, P in enumerate(A1_P[:-1]):
                if A2_rho[i, j + 1] < A2_rho[i, j]:
                    A2_rho[i, j + 1] = A2_rho[i, j]

        A2_u = (1 - Y) * self.A2_u_H_TP + Y * self.A2_u_He_TP
        A2_c = (1 - Y) * self.A2_c_H_TP + Y * self.A2_c_He_TP

        # s has additional term:
        # Mass fraction
        x_H = ((1 - Y) / m_H) / (((1 - Y) / m_H) + (Y / m_He))
        x_He = (Y / m_He) / (((1 - Y) / m_H) + (Y / m_He))
        A = x_H + x_He * m_He / m_H
        s_mix = -k * ((x_H * np.log(x_H) + x_He * np.log(x_He)) / (A * m_H))

        A2_s = (1 - Y) * self.A2_s_H_TP + Y * self.A2_s_He_TP + s_mix

        return A1_T, A1_P, A2_rho, A2_u, A2_s, A2_c

    def P_T_rho(self, T, rho, A1_P, A1_T, A2_rho):

        """Compute the pressure from the temperature and density.
        Similar to functions in e.g. sesame.py

        Parameters
        ----------
        T : float
            Temperature.

        rho : float
            Density (kg m^-3).

        A1_P, A1_T : [float]
            Pressure (Pa) and temperature (K) arrays. Axes of tables.

        A2_rho : [[float]]
            Density (kg m^-3) table.

        Returns
        -------
        P : float
            Pressure (Pa).
        """

        A1_log_P = np.log(A1_P)
        A1_log_T = np.log(A1_T)

        # Convert to log
        log_T = np.log(T)

        idx_T_intp_T = find_index_and_interp(log_T, A1_log_T)
        idx_T = int(idx_T_intp_T[0])
        intp_T = idx_T_intp_T[1]

        idx_rho_1_intp_rho_1 = find_index_and_interp(rho, A2_rho[idx_T, :])
        idx_rho_1 = int(idx_rho_1_intp_rho_1[0])
        intp_rho_1 = idx_rho_1_intp_rho_1[1]
        idx_rho_2_intp_rho_2 = find_index_and_interp(rho, A2_rho[idx_T + 1, :])
        idx_rho_2 = int(idx_rho_2_intp_rho_2[0])
        intp_rho_2 = idx_rho_2_intp_rho_2[1]

        # Normal interpolation
        log_P = (1 - intp_T) * (
            (1 - intp_rho_1) * A1_log_P[idx_rho_1]
            + intp_rho_1 * A1_log_P[idx_rho_1 + 1]
        ) + intp_T * (
            (1 - intp_rho_2) * A1_log_P[idx_rho_2]
            + intp_rho_2 * A1_log_P[idx_rho_2 + 1]
        )

        # Convert back from log
        P = np.exp(log_P)
        if P < 0:
            P = 0

        return P

    def X_T_rho(self, T, rho, A2_X, A1_T, A2_rho):

        """Compute the internal energy/entropy/sound speed from the temperature and density.
        Similar to functions in e.g. sesame.py

        Parameters
        ----------
        T : float
            Temperature.

        rho : float
            Density (kg m^-3).

        A2_X : [[float]]
            Table arrays of either sp. int. energy (J kg^-1), sound speed
            (m s^-1), or sp. entropy (J K^-1 kg^-1).

        A1_T : [float]
            Temperature (K) arrays. Axes of tables.

        A2_rho : [[float]]
            Density (kg m^-3) table.

        Returns
        -------
        X : float
            either sp. int. energy (J kg^-1), sound speed (m s^-1), or
            sp. entropy (J K^-1 kg^-1).
        """

        A1_log_T = np.log(A1_T)
        A2_log_rho = np.log(A2_rho)

        # Convert to log
        log_T = np.log(T)
        log_rho = np.log(rho)

        idx_T_intp_T = find_index_and_interp(log_T, A1_log_T)
        idx_T = int(idx_T_intp_T[0])
        intp_T = idx_T_intp_T[1]

        idx_rho_1_intp_rho_1 = find_index_and_interp(log_rho, A2_log_rho[idx_T])
        idx_rho_1 = int(idx_rho_1_intp_rho_1[0])
        intp_rho_1 = idx_rho_1_intp_rho_1[1]
        idx_rho_2_intp_rho_2 = find_index_and_interp(log_rho, A2_log_rho[idx_T + 1])
        idx_rho_2 = int(idx_rho_2_intp_rho_2[0])
        intp_rho_2 = idx_rho_2_intp_rho_2[1]

        X_1 = A2_X[idx_T, idx_rho_1]
        X_2 = A2_X[idx_T, idx_rho_1 + 1]
        X_3 = A2_X[idx_T + 1, idx_rho_2]
        X_4 = A2_X[idx_T + 1, idx_rho_2 + 1]

        if idx_T >= 0 and (intp_rho_1 < 0 or intp_rho_2 < 0):
            intp_rho_1 = 0
            intp_rho_2 = 0

        # Interpolate with the log values
        X_1 = np.log(X_1)
        X_2 = np.log(X_2)
        X_3 = np.log(X_3)
        X_4 = np.log(X_4)

        # P(rho, u)
        X = (1 - intp_T) * ((1 - intp_rho_1) * X_1 + intp_rho_1 * X_2) + intp_T * (
            (1 - intp_rho_2) * X_3 + intp_rho_2 * X_4
        )

        # Convert back from log
        return np.exp(X)

    def transform_TP_to_Trho(self, A1_rho, A1_T):

        """Convert tabes in TP forma to Trho format

        Parameters
        ----------
        A1_rho, A1_T : [float]
            Density (kg m^-3) and temperature (K) arrays. desired axes of tables.

        A2_rho : [[float]]
            Density (kg m^-3) table.

        Returns
        -------
        A2_P_Trho, A2_u_Trho, A2_s_Trho, A2_c_Trho : [float]
            Table arrays of sp. int. energy (J kg^-1), density (kg m^-3), sound speed
            (m s^-1), and sp. entropy (J K^-1 kg^-1). Now converted to Trho format.
        """

        assert self.input_format == "TP"

        A2_P_Trho = np.empty((len(A1_rho), len(A1_T)))
        A2_u_Trho = np.empty((len(A1_rho), len(A1_T)))
        A2_s_Trho = np.empty((len(A1_rho), len(A1_T)))
        A2_c_Trho = np.empty((len(A1_rho), len(A1_T)))

        for i, rho in enumerate(A1_rho):
            for j, T in enumerate(A1_T):

                A2_P_Trho[i, j] = self.P_T_rho(
                    T, rho, self.A1_P_TP, self.A1_T_TP, self.A2_rho_TP
                )
                A2_u_Trho[i, j] = self.X_T_rho(
                    T, rho, self.A2_u_TP, self.A1_T_TP, self.A2_rho_TP
                )
                A2_s_Trho[i, j] = self.X_T_rho(
                    T, rho, self.A2_s_TP, self.A1_T_TP, self.A2_rho_TP
                )
                A2_c_Trho[i, j] = self.X_T_rho(
                    T, rho, self.A2_c_TP, self.A1_T_TP, self.A2_rho_TP
                )

        return A2_P_Trho, A2_u_Trho, A2_s_Trho, A2_c_Trho

    def write_table(self, Fp_table, A1_rho=None):

        """Generates and writes tables to SESAME-style file

        Parameters
        ----------
        Fp_table : str
            The table file path.

        A1_rho : [float]
            Density (kg m^-3) array.
        """

        if self.input_format == "Trho":
            # If input table format is "Trho" we only need to load table
            if self.Y == 0:
                (
                    self.A1_T_Trho,
                    self.A1_rho_Trho,
                    self.A2_P_Trho,
                    self.A2_u_Trho,
                    self.A2_s_Trho,
                    self.A2_c_Trho,
                ) = self.load_table_Trho(self.Fp_H)
            elif self.Y == 1:
                (
                    self.A1_T_Trho,
                    self.A1_rho_Trho,
                    self.A2_P_Trho,
                    self.A2_u_Trho,
                    self.A2_s_Trho,
                    self.A2_c_Trho,
                ) = self.load_table_Trho(self.Fp_He)
            else:
                raise ValueError(
                    'input_format "Trho" is only compatible with Y = 0 or 1'
                )
        elif self.input_format == "TP":
            # If input table format is "TP" we need to transform to "Trho" before saving
            assert A1_rho is not None

            if self.Y == 0:
                (
                    self.A1_T_TP,
                    self.A1_P_TP,
                    self.A2_rho_TP,
                    self.A2_u_TP,
                    self.A2_s_TP,
                    self.A2_c_TP,
                ) = self.load_table_TP(self.Fp_H)

            elif self.Y == 1:
                (
                    self.A1_T_TP,
                    self.A1_P_TP,
                    self.A2_rho_TP,
                    self.A2_u_TP,
                    self.A2_s_TP,
                    self.A2_c_TP,
                ) = self.load_table_TP(self.Fp_He)

            elif 0 < self.Y < 1:

                (
                    self.A1_T_H_TP,
                    self.A1_P_H_TP,
                    self.A2_rho_H_TP,
                    self.A2_u_H_TP,
                    self.A2_s_H_TP,
                    self.A2_c_H_TP,
                ) = self.load_table_TP(self.Fp_H)
                (
                    self.A1_T_He_TP,
                    self.A1_P_He_TP,
                    self.A2_rho_He_TP,
                    self.A2_u_He_TP,
                    self.A2_s_He_TP,
                    self.A2_c_He_TP,
                ) = self.load_table_TP(self.Fp_He)
                # if Y isn't exacly 0 or 1 we need to combine tables to get mixture
                (
                    self.A1_T_TP,
                    self.A1_P_TP,
                    self.A2_rho_TP,
                    self.A2_u_TP,
                    self.A2_s_TP,
                    self.A2_c_TP,
                ) = self.additive_volume_law()

            self.A1_T_Trho = self.A1_T_TP
            self.A1_rho_Trho = A1_rho

            # These are our final tables
            (
                self.A2_P_Trho,
                self.A2_u_Trho,
                self.A2_s_Trho,
                self.A2_c_Trho,
            ) = self.transform_TP_to_Trho(self.A1_rho_Trho, self.A1_T_Trho)

        else:
            raise ValueError('input_format must have value "Trho" or "TP"')

        # Name based on mass fraction
        if self.Y == 0:
            name = "CMS19 H"
        elif self.Y == 1:
            name = "CMS19 He"
        else:
            name = "Chabrier & Debras (2021) H-He Y=" + str(self.Y)

        # Save as SESAME-style table
        write_table_SESAME(
            Fp_table,
            name,
            self.version_date,
            self.A1_rho_Trho,
            self.A1_T_Trho,
            self.A2_u_Trho,
            self.A2_P_Trho,
            self.A2_c_Trho,
            self.A2_s_Trho,
        )
