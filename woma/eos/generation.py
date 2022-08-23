"""WoMa equations of state table generation."""

import numpy as np
from woma.misc import utils as ut


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
