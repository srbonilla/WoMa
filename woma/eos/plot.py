"""
WoMa equations of state (EoS).

Note: Numba places odd requirements and limitations for e.g. global-scope
variables and (avoiding) custom classes, which leads to some awkward or ugly but
functional approaches here...
"""

from numba import njit
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from woma.misc import glob_vars as gv
from woma.eos import tillotson, sesame, idg, hm80, mixed
from woma.eos import eos, generation
from woma.misc import utils as ut


# ========
# Table values
# ========
def plot_table_HM80(mat, Z_choice, A1_fig_ax=None):
    """Plot all entries in a Hubbard & MacFarlane (1980) (rho, u) table.

    Parameters
    ----------
    mat : str
        The material name.

    Z_choice : str
        The table parameter to colour by, choose from:
            "P"     Pressure.
            "T"     Temperature.

    A1_fig_ax : [fig, ax] (opt.)
        If provided, then plot on this existing figure and axes instead of
        making new ones.

    Returns
    -------
    A1_fig_ax : [fig, ax]
        The figure and axes.
    """
    # Figure
    if A1_fig_ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.gca()
    else:
        fig, ax = A1_fig_ax
        plt.figure(fig.number)

    # Table to load
    if mat == "HM80_HHe":
        Fp_load = gv.Fp_HM80_HHe
    elif mat == "HM80_ice":
        Fp_load = gv.Fp_HM80_ice
    elif mat == "HM80_rock":
        Fp_load = gv.Fp_HM80_rock
    else:
        raise ValueError("Invalid material name", mat)

    # Load table data
    (
        log_rho_min,
        log_rho_max,
        num_rho,
        log_rho_step,
        log_u_min,
        log_u_max,
        num_u,
        log_u_step,
        A2_log_P,
        A2_log_T,
    ) = hm80.load_table_HM80(Fp_load)
    A1_rho = np.exp(np.linspace(log_rho_min, log_rho_max, num_rho))
    A1_u = np.exp(np.linspace(log_u_min, log_u_max, num_u))

    # Colour parameter
    if Z_choice == "P":
        A2_Z = np.exp(A2_log_P)
    elif Z_choice == "T":
        A2_Z = np.exp(A2_log_T)
    else:
        raise ValueError("Invalid colour choice", Z_choice)
    cmap = plt.get_cmap("viridis")
    vmin = np.nanmin(A2_Z)
    vmax = np.nanmax(A2_Z)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    # Plot each row
    for i_u, u in enumerate(A1_u):
        scat = ax.scatter(
            A1_rho,
            np.full(num_rho, u),
            marker=".",
            s=5**2,
            c=A2_Z[:, i_u],
            edgecolor="none",
            cmap=cmap,
            norm=norm,
        )

    # Colour bar
    cbar = plt.colorbar(scat)
    if Z_choice == "P":
        cbar.set_label(r"Pressure (Pa)")
    elif Z_choice == "T":
        cbar.set_label(r"Temperature (K)")

    # Axes etc
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Density ($\rm{kg m}^{-3}$)")
    ax.set_ylabel(r"Sp. Int. Energy ($\rm{J kg}^{-1}$)")
    ax.set_title(r"%s $%s(\rho, u)$" % (mat, Z_choice))

    plt.tight_layout()

    return fig, ax


def plot_all_HM80_tables():
    for mat in ["HM80_HHe", "HM80_ice", "HM80_rock"]:
        for param in "P", "T":
            plot_table_HM80(mat, param)

            Fp_save = "%s_table_%s_rho_u.png" % (mat, param)
            plt.savefig(Fp_save, dpi=200)
            plt.close()
            print('Saved "%s"' % Fp_save)


def plot_table_SESAME(mat, Z_choice, A1_fig_ax=None):
    """Plot all entries in a SESAME or ANEOS etc (rho, T) table.

    Parameters
    ----------
    mat : str
        The material name.

    Z_choice : str
        The table parameter to colour by, choose from:
            "P"     Pressure.
            "u"     Specific internal energy.
            "s"     Specific entropy.

    A1_fig_ax : [fig, ax] (opt.)
        If provided, then plot on this existing figure and axes instead of
        making new ones.

    Returns
    -------
    A1_fig_ax : [fig, ax]
        The figure and axes.
    """
    mat_id = gv.Di_mat_id[mat]

    # Load tables
    ut.load_eos_tables(mat)

    # Load table data
    if mat_id == gv.id_SESAME_iron:
        A1_log_T = sesame.A1_log_T_SESAME_iron
        A1_log_rho = sesame.A1_log_rho_SESAME_iron
        if Z_choice == "P":
            A2_Z = sesame.A2_P_SESAME_iron
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_SESAME_iron
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_SESAME_iron
    elif mat_id == gv.id_SESAME_basalt:
        A1_log_T = sesame.A1_log_T_SESAME_basalt
        A1_log_rho = sesame.A1_log_rho_SESAME_basalt
        if Z_choice == "P":
            A2_Z = sesame.A2_P_SESAME_basalt
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_SESAME_basalt
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_SESAME_basalt
    elif mat_id == gv.id_SESAME_water:
        A1_log_T = sesame.A1_log_T_SESAME_water
        A1_log_rho = sesame.A1_log_rho_SESAME_water
        if Z_choice == "P":
            A2_Z = sesame.A2_P_SESAME_water
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_SESAME_water
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_SESAME_water
    elif mat_id == gv.id_SS08_water:
        A1_log_T = sesame.A1_log_T_SS08_water
        A1_log_rho = sesame.A1_log_rho_SS08_water
        if Z_choice == "P":
            A2_Z = sesame.A2_P_SS08_water
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_SS08_water
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_SS08_water
    elif mat_id == gv.id_ANEOS_forsterite:
        A1_log_T = sesame.A1_log_T_ANEOS_forsterite
        A1_log_rho = sesame.A1_log_rho_ANEOS_forsterite
        if Z_choice == "P":
            A2_Z = sesame.A2_P_ANEOS_forsterite
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_ANEOS_forsterite
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_ANEOS_forsterite
        # elif Z_choice == "phase":
        #     A2_Z = sesame.A2_phase_ANEOS_forsterite
    elif mat_id == gv.id_ANEOS_iron:
        A1_log_T = sesame.A1_log_T_ANEOS_iron
        A1_log_rho = sesame.A1_log_rho_ANEOS_iron
        if Z_choice == "P":
            A2_Z = sesame.A2_P_ANEOS_iron
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_ANEOS_iron
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_ANEOS_iron
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A1_log_T = sesame.A1_log_T_ANEOS_Fe85Si15
        A1_log_rho = sesame.A1_log_rho_ANEOS_Fe85Si15
        if Z_choice == "P":
            A2_Z = sesame.A2_P_ANEOS_Fe85Si15
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_ANEOS_Fe85Si15
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_ANEOS_Fe85Si15
    elif mat_id == gv.id_AQUA:
        A1_log_T = sesame.A1_log_T_AQUA
        A1_log_rho = sesame.A1_log_rho_AQUA
        if Z_choice == "P":
            A2_Z = sesame.A2_P_AQUA
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_AQUA
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_AQUA
    elif mat_id == gv.id_CMS19_H:
        A1_log_T = sesame.A1_log_T_CMS19_H
        A1_log_rho = sesame.A1_log_rho_CMS19_H
        if Z_choice == "P":
            A2_Z = sesame.A2_P_CMS19_H
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_CMS19_H
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_CMS19_H
    elif mat_id == gv.id_CMS19_He:
        A1_log_T = sesame.A1_log_T_CMS19_He
        A1_log_rho = sesame.A1_log_rho_CMS19_He
        if Z_choice == "P":
            A2_Z = sesame.A2_P_CMS19_He
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_CMS19_He
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_CMS19_He
    elif mat_id == gv.id_CD21_HHe:
        A1_log_T = sesame.A1_log_T_CD21_HHe
        A1_log_rho = sesame.A1_log_rho_CD21_HHe
        if Z_choice == "P":
            A2_Z = sesame.A2_P_CD21_HHe
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_CD21_HHe
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_CD21_HHe
    else:
        raise ValueError("Invalid material ID")
    A1_T = np.exp(A1_log_T)
    A1_rho = np.exp(A1_log_rho)
    num_T = len(A1_T)
    num_rho = len(A1_rho)

    # Skip if no data (e.g. entropies all zero)
    if np.all(A2_Z == 0):
        print("%s %s all zero" % (mat, Z_choice))
        return None, None

    # Colour parameter
    cmap = plt.get_cmap("viridis")
    vmin = np.nanmin(A2_Z[A2_Z > 0])
    vmax = np.nanmax(A2_Z[A2_Z < np.inf])
    if "_H" in mat and Z_choice == "s":
        vmax = min(vmax, 1e7)
        vmin = max(vmin, 1e2)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    # Figure
    if A1_fig_ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.gca()
    else:
        fig, ax = A1_fig_ax
        plt.figure(fig.number)

    # Roughly adjust marker size by number of points
    s_def = 5**2
    num_def = 200
    s = min(s_def, (np.sqrt(s_def) * num_def / max(num_rho, num_T)) ** 2)

    # Plot each row
    for i_T, T in enumerate(A1_T):
        # Raise very low edge rho,T for visibility
        T_min = 0.01 * A1_T[1]
        if i_T == 0 and T < T_min:
            T = T_min
            m = "v"
            s_ = s * 0.5**2
        else:
            m = "."
            s_ = s

        scat = ax.scatter(
            A1_rho,
            np.full(num_rho, T),
            marker=m,
            s=s_,
            c=A2_Z[:, i_T],
            edgecolor="none",
            cmap=cmap,
            norm=norm,
        )

        # Plot non-positive values in grey
        A1_sel_zero = np.where(A2_Z[:, i_T] <= 0)[0]
        ax.scatter(
            A1_rho[A1_sel_zero],
            np.full(num_rho, T)[A1_sel_zero],
            marker=m,
            s=s_,
            c="0.6",
            edgecolor="none",
        )

    # Colour bar
    cbar = plt.colorbar(scat)
    if Z_choice == "P":
        cbar.set_label(r"Pressure (Pa)")
    elif Z_choice == "u":
        cbar.set_label(r"Specific internal energy (J kg$^{-1}$)")
    elif Z_choice == "s":
        cbar.set_label(r"Specific entropy (J K$^{-1}$ kg$^{-1}$)")

    # Axes etc
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Density ($\rm{kg m}^{-3}$)")
    ax.set_ylabel(r"Temperature (K)")
    ax.set_title(r"%s $%s(\rho, T)$" % (mat, Z_choice))

    plt.tight_layout()

    return fig, ax


def plot_all_SESAME_tables():
    for mat in [
        "SESAME_iron",
        "SESAME_basalt",
        "SESAME_water",
        "SS08_water",
        "AQUA",
        "CMS19_H",
        "CMS19_He",
        "CD21_HHe",
        "ANEOS_forsterite",
        "ANEOS_iron",
        "ANEOS_Fe85Si15",
    ]:
        for param in ["P", "u", "s"]:
            fig, ax = plot_table_SESAME(mat, param)

            if fig is not None:
                dir = "plots/%s" % mat
                os.makedirs(dir, exist_ok=True)
                Fp_save = "%s/%s_table_%s_rho_T.png" % (dir, mat, param)
                plt.savefig(Fp_save, dpi=600)
                plt.close()
                print('Saved "%s"' % Fp_save)


def plot_table_mixed(mat, Z_choice, mix, A1_fig_ax=None):
    """Plot all entries in a mixed (rho, T) table.

    Parameters
    ----------
    mat : str
        The material name.

    Z_choice : str
        The table parameter to colour by, choose from:
            "P"     Pressure.
            "u"     Specific internal energy.
            "s"     Specific entropy.
            "c"     Sound speed.

    mix : float
        Mixing mass fraction.

    A1_fig_ax : [fig, ax] (opt.)
        If provided, then plot on this existing figure and axes instead of
        making new ones.

    Returns
    -------
    A1_fig_ax : [fig, ax]
        The figure and axes.
    """
    mat_id = gv.Di_mat_id[mat]

    # Load tables
    ut.load_eos_tables(mat)

    # Unpack the arrays of Z, mix, log(rho), and log(T)
    A3_Z = np.zeros((2, 2, 2))
    if mat_id == gv.id_mixed_HHe_rock:
        A1_mix, A1_log_rho, A1_log_T = (
            mixed.A1_mix_mixed_HHe_rock,
            mixed.A1_log_rho_mixed_HHe_rock,
            mixed.A1_log_T_mixed_HHe_rock,
        )
        if Z_choice == "P":
            A3_Z = mixed.A3_P_mixed_HHe_rock
        elif Z_choice == "u":
            A3_Z = mixed.A3_u_mixed_HHe_rock
        elif Z_choice == "c":
            A3_Z = mixed.A3_c_mixed_HHe_rock
        elif Z_choice == "s":
            A3_Z = mixed.A3_s_mixed_HHe_rock
    elif mat_id == gv.id_mixed_HHe_water:
        A1_mix, A1_log_rho, A1_log_T = (
            mixed.A1_mix_mixed_HHe_water,
            mixed.A1_log_rho_mixed_HHe_water,
            mixed.A1_log_T_mixed_HHe_water,
        )
        if Z_choice == "P":
            A3_Z = mixed.A3_P_mixed_HHe_water
        elif Z_choice == "u":
            A3_Z = mixed.A3_u_mixed_HHe_water
        elif Z_choice == "c":
            A3_Z = mixed.A3_c_mixed_HHe_water
        elif Z_choice == "s":
            A3_Z = mixed.A3_s_mixed_HHe_water
    else:
        raise ValueError("Invalid material ID")
    A1_T = np.exp(A1_log_T)
    A1_rho = np.exp(A1_log_rho)
    num_T = len(A1_T)
    num_rho = len(A1_rho)
    idx_mix = int(ut.find_index_and_interp(mix, A1_mix)[0])
    A2_Z = A3_Z[idx_mix]

    # Colour parameter
    cmap = plt.get_cmap("viridis")
    vmin = np.nanmin(A2_Z[A2_Z > 0])
    vmax = np.nanmax(A2_Z[A2_Z < np.inf])
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    # Figure
    if A1_fig_ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.gca()
    else:
        fig, ax = A1_fig_ax
        plt.figure(fig.number)

    # Plot each row
    for i_T, T in enumerate(A1_T):
        scat = ax.scatter(
            A1_rho,
            np.full(num_rho, T),
            marker="s",
            s=4.25**2,
            c=A2_Z[:, i_T],
            edgecolor="none",
            cmap=cmap,
            norm=norm,
        )

        # Plot non-positive values in grey
        A1_sel_zero = np.where(A2_Z[:, i_T] <= 0)[0]
        ax.scatter(
            A1_rho[A1_sel_zero],
            np.full(num_rho, T)[A1_sel_zero],
            marker="s",
            s=4.25**2,
            c="0.4",
            edgecolor="none",
        )

    # Colour bar
    cbar = plt.colorbar(scat)
    if Z_choice == "P":
        cbar.set_label(r"Pressure (Pa)")
    elif Z_choice == "u":
        cbar.set_label(r"Specific internal energy (J kg$^{-1}$)")
    elif Z_choice == "s":
        cbar.set_label(r"Specific entropy (J K$^{-1}$ kg$^{-1}$)")
    elif Z_choice == "c":
        cbar.set_label(r"Sound speed (m s$^{-1}$)")

    # Axes etc
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Density ($\rm{kg m}^{-3}$)")
    ax.set_ylabel(r"Temperature (K)")
    ax.set_title(r"%s, mix=%g, $%s(\rho, T)$" % (mat, mix, Z_choice))

    plt.tight_layout()

    return fig, ax


def plot_all_mixed_tables():
    for mat in [
        generation.mat_mixed_HHe_rock,
        generation.mat_mixed_HHe_water,
    ]:
        for mix in mat.A1_mix_raw:
            for param in ["P", "u", "s", "c"]:
                fig, ax = plot_table_mixed(mat.name, param, mix)

                if fig is not None:
                    dir = "plots/mixed_HHe_%s" % param
                    os.makedirs(dir, exist_ok=True)
                    Fp_save = "%s/%s_%.2f_table_%s_rho_T.png" % (dir, mat.name, mix, param)
                    plt.savefig(Fp_save, dpi=600)
                    plt.close()
                    print('Saved "%s"' % Fp_save)


# ========
# Arbitary iso lines, specific comparisons, etc
# ========
def plot_eos_Z_X_iso_Y(
    mat,
    Z_choice,
    X_choice,
    Y_choice,
    X_min=None,
    X_max=None,
    Y_min=None,
    Y_max=None,
    Z_min=None,
    Z_max=None,
    num_X=100,
    num_Y=20,
    A1_Y=None,
    num_label=None,
    lw=1,
    ls="-",
    alpha=1,
    cmap=plt.get_cmap("viridis"),
    A1_fig_ax=None,
):
    """Plot lines of "Z" values as a function of "X" at a range of constant "Y".

    e.g. Z(X, Y) = P(rho, T) = pressure(density) for a set of isotherms.
    Not all Z(X, Y) choices are available for all equations of state.

    Paramters
    ---------
    mat : int
        The material name.

    Z_choice : str
        The parameter to calculate (for the vertical axis), choose from:
            "P"         Pressure.
            "u"         Specific internal energy.
            "s"         Specific entropy.
            "c"         Sound speed.

    X_choice, Y_choice : str
        The input parameters for the horizontal axis and iso-values respectively:
            "rho"       Density.
            "T"         Temperature.
            "u"         Specific internal energy.

    X_min, X_max, Y_min, Y_max, Z_min, Z_max : float (opt.)
        The minimum and maximum values for the each parameter. Defaults to
        crude guesses of maybe-sensible values.

    num_X, num_Y : float (opt.)
        The number of values for the x-axis and iso parameters.

    A1_Y : [float] (opt.)
        Alternatively, an array of arbitrary iso values, overrides num_Y etc.

    num_label : int (opt.)
        Optional override for the number of iso lines to label. Set 0 for none.

    lw, ls, alpha, cmap : float, str, float, cmap (opt.)
        The linewidth, linestyle, opacity, and colour map.

    A1_fig_ax : [fig, ax] (opt.)
        If provided, then plot on this existing figure and axes instead of
        making new ones.

    Returns
    -------
    A1_fig_ax : [fig, ax]
        The figure and axes.
    """
    # Load tables, if necessary
    ut.load_eos_tables(mat)

    # Figure
    if A1_fig_ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
    else:
        fig, ax = A1_fig_ax
        plt.figure(fig.number)

    assert Z_choice != X_choice and Z_choice != Y_choice and X_choice != Y_choice

    # Parameter labels and default limits
    Di_choice_label_min_max = {
        "P": [r"Pressure (Pa)", 3e4, 1e12],
        "u": [r"Specific internal energy (J kg$^{-1}$)", 1e4, 1e10],
        "s": [r"Specific entropy (J K$^{-1}$ kg$^{-1}$)", 3e1, 3e5],
        "rho": [r"Density (kg m$^{-3}$)", 3e0, 3e4],
        "T": [r"Temperature (K)", 1e2, 1e5],
    }
    X_label, X_min_def, X_max_def = Di_choice_label_min_max[X_choice]
    Y_label, Y_min_def, Y_max_def = Di_choice_label_min_max[Y_choice]
    Z_label, Z_min_def, Z_max_def = Di_choice_label_min_max[Z_choice]

    # Set parameter ranges
    if X_min is None:
        X_min = X_min_def
    if X_max is None:
        X_max = X_max_def
    if A1_Y is not None:
        A1_Y = np.array(A1_Y)
        Y_min = np.nanmin(A1_Y)
        Y_max = np.nanmax(A1_Y)
        num_Y = len(A1_Y)
    else:
        if Y_min is None:
            Y_min = Y_min_def
        if Y_max is None:
            Y_max = Y_max_def
        A1_Y = np.exp(np.linspace(np.log(Y_min), np.log(Y_max), num_Y))

    # X values
    A1_X = np.exp(np.linspace(np.log(X_min), np.log(X_max), num_X))

    # Material ID
    mat_id = gv.Di_mat_id[mat]
    A1_mat_id = np.full(num_X, mat_id)

    # Calculate Z(X, Y)
    A2_Z = np.empty((num_Y, num_X))
    for i_Y, Y in enumerate(A1_Y):
        A2_Z[i_Y] = eos.A1_Z_X_Y(
            A1_X=A1_X,
            A1_Y=np.full(num_X, Y),
            A1_mat_id=A1_mat_id,
            Z_choice=Z_choice,
            X_choice=X_choice,
            Y_choice=Y_choice,
        )

    # Set colour for each iso-Y value
    vmin = np.nanmin(A1_Y)
    vmax = np.nanmax(A1_Y)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    A1_colour = [mapper.to_rgba(Y) for Y in A1_Y]

    # Label the iso lines, or only a subset if there are many
    if num_label == 0:
        A1_Y_label = []
    else:
        if num_label is None:
            if num_Y < 20:
                num_label = num_Y
            else:
                num_label = 10
        A1_Y_label = A1_Y[:: num_Y // num_label]
    handles = []

    # Plot
    for i_Y, Y in enumerate(A1_Y):
        # Label (a subset of) lines
        if Y in A1_Y_label:
            label = r"%.3g" % Y
        else:
            label = None

        line = ax.plot(
            A1_X,
            A2_Z[i_Y],
            c=A1_colour[i_Y],
            lw=lw,
            ls=ls,
            alpha=alpha,
            label=label,
        )[0]

        if label is not None:
            handles.append(line)

    # Colour bar
    if not True:
        line = ax.plot(
            [],
            [],
            c=[],
            cmap=cmap,
            norm=norm,
            lw=lw,
            ls=ls,
            alpha=alpha,
        )
        cbar = plt.colorbar(line)
        cbar.set_label(Y_label)

    # Legend
    if len(A1_Y_label) > 0:
        ax.legend(title=Y_label, handles=handles)

    # Axes etc
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(X_label)
    ax.set_ylabel(Z_label)
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(Z_min_def, Z_max_def)

    plt.tight_layout()

    return fig, ax


def plot_mixed_eos_Z_X_iso_Y(
    Z_choice,
    X_choice,
    Y_choice,
    W_choice,
    W,
    X_min=None,
    X_max=None,
    Y_min=None,
    Y_max=None,
    Z_min=None,
    Z_max=None,
    num_X=200,
    num_Y=49,
    A1_Y=None,
    num_label=None,
    lw=1,
    ls="-",
    alpha=1,
    cmap=plt.get_cmap("viridis"),
    A1_fig_ax=None,
):
    """Plot lines of "Z" values as a function of "X" at a range of constant "Y".

    For mixed-material EoS.

    e.g. Z(X, Y) = P(rho, T) = pressure(density) for a set of isotherms.
    Not all Z(X, Y) choices are available for all equations of state.

    Paramters
    ---------
    Z_choice : str
        The parameter to calculate (for the vertical axis), choose from:
            "P"         Pressure.
            "u"         Specific internal energy.
            "s"         Specific entropy.
            "c"         Sound speed.

    X_choice, Y_choice : str
        The input parameters for the horizontal axis and iso-values respectively:
            "rho"       Density.
            "T"         Temperature.
            "u"         Specific internal energy.
            "mix_*"     Mixing mass fraction of * = rock, water.

    W_choice, W : str, float or [float]
        If Y_choice is a mixing fraction, then use this choice and constant
        value for the other input. If X_choice is a mixing fraction, then this
        must be "rho" and a constant density. If neither, then this must be
        "A1_mix": the mixing mass fraction of each heavy component.

    X_min, X_max, Y_min, Y_max, Z_min, Z_max : float (opt.)
        The minimum and maximum values for the each parameter. Defaults to
        crude guesses of maybe-sensible values.

    num_X, num_Y : float (opt.)
        The number of values for the x-axis and iso parameters.

    A1_Y : [float] (opt.)
        Alternatively, an array of arbitrary iso values, overrides num_Y etc.

    rho_fixed : float (opt.)
        If X_choice is a mixing fraction, then use this constant density value.

    num_label : int (opt.)
        Optional override for the number of iso lines to label. Set 0 for none.

    lw, ls, alpha, cmap : float, str, float, cmap (opt.)
        The linewidth, linestyle, opacity, and colour map.

    A1_fig_ax : [fig, ax] (opt.)
        If provided, then plot on this existing figure and axes instead of
        making new ones.

    Returns
    -------
    A1_fig_ax : [fig, ax]
        The figure and axes.
    """
    # Load tables, if necessary
    ut.load_eos_tables(["mixed_HHe_rock", "mixed_HHe_water"])

    # Figure
    if A1_fig_ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
    else:
        fig, ax = A1_fig_ax
        plt.figure(fig.number)

    assert Z_choice != X_choice and Z_choice != Y_choice and X_choice != Y_choice

    # Currently implemented options
    if "mix_" in X_choice:
        assert W_choice == "rho"
    elif "mix_" in Y_choice:
        assert X_choice == "rho"
        assert "mix_" not in W_choice
    else:
        assert X_choice in ["rho", "T"]
        assert W_choice == "A1_mix"

    # Parameter labels and default limits
    Di_choice_label_min_max = {
        "P": [r"Pressure (Pa)", 3e1, 3e15],
        "u": [r"Specific internal energy (J kg$^{-1}$)", 1e6, 7e10],
        "s": [r"Specific entropy (J K$^{-1}$ kg$^{-1}$)", 3e2, 5e5],
        "c": [r"Sound speed (m s$^{-1}$)", 1e1, 1e8],
        "rho": [r"Density (kg m$^{-3}$)", 1e-4, 3e4],
        "T": [r"Temperature (K)", 1e2, 1e5],
        "mix_rock": [r"Mixed mass fraction of rock", 0, 1],
        "mix_water": [r"Mixed mass fraction of water", 0, 1],
    }
    X_label, X_min_def, X_max_def = Di_choice_label_min_max[X_choice]
    Y_label, Y_min_def, Y_max_def = Di_choice_label_min_max[Y_choice]
    Z_label, Z_min_def, Z_max_def = Di_choice_label_min_max[Z_choice]

    # Set parameter ranges
    if X_min is None:
        X_min = X_min_def
    if X_max is None:
        X_max = X_max_def
    if A1_Y is None:
        if Y_min is None:
            Y_min = Y_min_def
        if Y_max is None:
            Y_max = Y_max_def
        A1_Y = np.exp(np.linspace(np.log(Y_min), np.log(Y_max), num_Y))
    else:
        A1_Y = np.array(A1_Y)
        Y_min = np.nanmin(A1_Y)
        Y_max = np.nanmax(A1_Y)
        num_Y = len(A1_Y)

    # X values
    if X_min > 0:
        A1_X = np.exp(np.linspace(np.log(X_min), np.log(X_max), num_X))
    else:
        A1_X = np.linspace(X_min, X_max, num_X)

    # Calculate Z(X, Y)
    A2_Z = np.empty((num_Y, num_X))
    if "mix_" in X_choice:
        A1_A1_mix = np.zeros((num_X, 2))
        if X_choice == "mix_rock":
            A1_A1_mix[:, 0] = A1_X
        else:
            A1_A1_mix[:, 1] = A1_X

        for i_Y, Y in enumerate(A1_Y):
            A2_Z[i_Y] = mixed.A1_Z_rho_Y_mix(
                A1_rho=np.full(num_X, W),
                A1_Y=np.full(num_X, Y),
                A1_A1_mix=A1_A1_mix,
                Z_choice=Z_choice,
                Y_choice=Y_choice,
            )
    elif "mix_" in Y_choice:
        for i_Y, Y in enumerate(A1_Y):
            A2_Z[i_Y] = mixed.A1_Z_rho_Y_mix(
                A1_rho=A1_X,
                A1_Y=np.full(num_X, W),
                A1_A1_mix=np.full((num_X, len(A1_mix)), A1_mix),
                Z_choice=Z_choice,
                Y_choice=W_choice,
            )
    elif X_choice == "rho":
        for i_Y, Y in enumerate(A1_Y):
            A2_Z[i_Y] = mixed.A1_Z_rho_Y_mix(
                A1_rho=A1_X,
                A1_Y=np.full(num_X, Y),
                A1_A1_mix=np.full((num_X, len(W)), W),
                Z_choice=Z_choice,
                Y_choice=Y_choice,
            )
    elif X_choice == "T":
        for i_Y, Y in enumerate(A1_Y):
            A2_Z[i_Y] = mixed.A1_Z_T_Y_mix(
                A1_T=A1_X,
                A1_Y=np.full(num_X, Y),
                A1_A1_mix=np.full((num_X, len(W)), W),
                Z_choice=Z_choice,
                Y_choice=Y_choice,
            )

    # Set colour for each iso-Y value
    vmin = np.nanmin(A1_Y)
    vmax = np.nanmax(A1_Y)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    A1_colour = [mapper.to_rgba(Y) for Y in A1_Y]

    # Label the iso lines, or only a subset if there are many
    if num_label == 0:
        A1_Y_label = []
    else:
        if num_label is None:
            if num_Y < 20:
                num_label = num_Y
            else:
                num_label = 10
        A1_Y_label = np.unique(np.append(A1_Y[:: num_Y // num_label], A1_Y[-1]))
    handles = []

    # Plot
    for i_Y, Y in enumerate(A1_Y):
        # Label (a subset of) lines
        if Y in A1_Y_label:
            if Y_choice == "T":
                label = r"%d" % np.round(Y)
            else:
                label = r"%.3g" % Y
        else:
            label = None

        line = ax.plot(
            A1_X,
            A2_Z[i_Y],
            c=A1_colour[i_Y],
            lw=lw,
            ls=ls,
            alpha=alpha,
            label=label,
        )[0]

        if label is not None:
            handles.append(line)

    # Print constant input W
    if W_choice == "A1_mix":
        const_label = "r, w = %.3g, %.3g" % (W[0], W[1])
    else:
        const_label = "%s = %.3g" % (W_choice, W)
    handles.append(ax.plot([], [], c="w", alpha=0, label=const_label)[0])

    # Colour bar
    if not True:
        line = ax.plot(
            [],
            [],
            c=[],
            cmap=cmap,
            norm=norm,
            lw=lw,
            ls=ls,
            alpha=alpha,
        )
        cbar = plt.colorbar(line)
        cbar.set_label(Y_label)

    # Legend
    if len(A1_Y_label) > 0:
        ax.legend(title=Y_label, handles=handles)

    # Axes etc
    if "mix" not in X_choice:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(X_label)
    ax.set_ylabel(Z_label)
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(Z_min_def, Z_max_def)

    plt.tight_layout()

    return fig, ax


def test_plot_mixed_eos_iso_lines():
    """Test plots for mixed materials."""
    # Fixed mix_rock,water combinations
    for Z_choice, X_choice, Y_choice in [
        ["P", "rho", "T"],
        ["P", "T", "rho"],
        ["u", "rho", "T"],
        ["u", "T", "rho"],
        ["P", "rho", "u"],
    ]:
        for A1_mix in [
            [0.0, 0.0],
            [0.234, 0.0],
            [1.0, 0.0],
            [0.0, 0.234],
            [0.0, 1.0],
            [0.234, 0.234],
            [1.0, 1.0],
        ]:
            # Figure
            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()

            # Plot
            plot_mixed_eos_Z_X_iso_Y(
                Z_choice=Z_choice,
                X_choice=X_choice,
                Y_choice=Y_choice,
                W_choice="A1_mix",
                W=A1_mix,
                A1_fig_ax=[fig, ax],
            )

            # Save the figure
            Fp_save = "mixed_eos_r%.3f_w%.3f_%s_%s_%s.png" % (
                A1_mix[0],
                A1_mix[1],
                Z_choice,
                X_choice,
                Y_choice,
            )
            plt.savefig(Fp_save, dpi=300)
            plt.close()
            print('Saved "%s"' % Fp_save)

    # Varied mixing fractions
    for Z_choice, X_choice, Y_choice in [
        ["P", "mix_rock", "T"],
        ["P", "mix_water", "T"],
        ["P", "mix_rock", "u"],
        ["P", "mix_water", "u"],
    ]:
        for rho in [2e3, 4e3, 6e3]:
            # Figure
            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()
    
            # Plot
            plot_mixed_eos_Z_X_iso_Y(
                Z_choice=Z_choice,
                X_choice=X_choice,
                Y_choice=Y_choice,
                W_choice="rho",
                W=rho,
                A1_fig_ax=[fig, ax],
            )
    
            # Save the figure
            Fp_save = "mixed_eos_%s_%s_%s_rho%g.png" % (
                Z_choice,
                X_choice,
                Y_choice,
                rho / 1e3,
            )
            plt.savefig(Fp_save, dpi=300)
            plt.close()
            print('Saved "%s"' % Fp_save)


if __name__ == "__main__":
    print(__file__)

    # plot_all_HM80_tables()
    # plot_all_SESAME_tables()
    # plot_all_mixed_tables()
    # test_plot_mixed_eos_iso_lines()
