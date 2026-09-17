"""
Create Supplementary Figure S1 from experimental and simulated frequencies.
"""

import sys                                                                      # module search path management
from pathlib import Path                                                        # project paths
from types import SimpleNamespace                                               # shared plotting state

import matplotlib as mpl                                                        # plotting backend management
import numpy as np                                                              # numerical plotting arrays
import pandas as pd                                                             # validation data loading
from matplotlib.colors import Normalize                                         # probability color scales
from matplotlib.patches import Rectangle                                        # unavailable-row hatching

mpl.use('Agg')                                                                  # non-interactive plotting backend
import matplotlib.pyplot as plt                                                 # plotting interface

# directory management
# -----------------------------------------------------------------------------
src  = Path(__file__).resolve().parent                                          # visualization directory
wdir = src.parent                                                               # code directory
rdir = wdir.parent                                                              # project root directory
ddir = rdir / 'data' / 'experiment' / 'derivatives'                             # observed data and metadata
vdir = rdir / 'data' / 'validation'                                             # validation analysis artifacts
fdir = rdir / 'figures'                                                         # Supplementary figure directory
udir = wdir / 'abm'                                                             # shared ABM utilities

# project utility imports
# -----------------------------------------------------------------------------
sys.path.append(str(udir))                                                      # make shared ABM modules importable
from abm_count import abm_count                                                 # pooled transition frequencies
from abm_describe import abm_describe                                           # observed participant data loading
from abm_figure import abm_figure                                               # shared Matplotlib defaults


def abm_figure_S1(sta):                                                         # Supplementary frequency figures
    """
    Save the PTD and RMC frequency figures for one validation source.

    Input
        sta : SimpleNamespace() object with required attributes:
            .D   : condition-specific trial data (pandas.DataFrame)
            .F   : task-specific normalized frequency dictionaries
            .c   : condition label, 'PTD-RMC'
            .m   : experimental metadata directory (pathlib.Path)
            .f   : figure output directory (pathlib.Path)
            .plt : configured matplotlib.pyplot module
            .src : source label, 'experiment' or 'simulation'

    Output
        sta : input SimpleNamespace with the new attribute:
            .ff : written Supplementary PNG and PDF paths

    The experimental source produces S1A and S1C. The design-matched A0
    simulation produces S1B and S1D. Each output contains two label-frequency
    panels and four conditional identifier-frequency panels.
    """
    # source and output specifications
    # -------------------------------------------------------------------------
    if sta.src == 'experiment':                                                 # observed experimental statistics
        nam = {'PTD': 'abm_figure_S1A', 'RMC': 'abm_figure_S1C'}                # experimental output names
    elif sta.src == 'simulation':                                               # A0-generated state statistics
        nam = {'PTD': 'abm_figure_S1B', 'RMC': 'abm_figure_S1D'}                # simulated output names
    else:                                                                       # reject ambiguous figure provenance
        raise ValueError("src must be 'experiment' or 'simulation'.")           # invalid validation source
    plt    = sta.plt                                                            # configured pyplot module
    cat    = pd.read_csv(sta.m / 'stimuli' / 'stimuli.csv')                     # complete stimulus catalogue
    sta.ff = []                                                                 # written output paths
    sta.f.mkdir(parents = True, exist_ok = True)                                # figure output directory

    # task-specific figures
    # -------------------------------------------------------------------------
    evt = (('inn', 'inp'), ('ipn', 'ipp'))                                      # old-label by new-label matrices
    for j, tsk in enumerate(('PTD', 'RMC'), start = 1):                         # one figure per validation task
        tab = cat[cat.task_type == tsk].sort_values('stimulus_id')              # task-specific catalogue
        n   = len(tab)                                                          # task stimulus count
        var = tab.fixed_label.isna().all()                                      # variable target assignments
        k   = len(sta.D['T'].iloc[0].split(' | ')) if var else 4                # positive-label identifier count
        fig, axs = plt.subplots(                                                # six-panel Supplementary figure
        3,                                                                      # label row and two identifier rows
        2,                                                                      # answer or successor-label columns
        figsize     = (7.7, 8.8),                                               # near-full report page
        gridspec_kw = {'height_ratios': (0.58, 1, 1)})                          # compact label panels
        fig.subplots_adjust(                                                    # publication spacing
        left   = 0.09,                                                          # room for row labels and letters
        right  = 0.95,                                                          # room for colorbars
        bottom = 0.07,                                                          # lower identifier labels
        top    = 0.91,                                                          # figure heading
        wspace = 0.31,                                                          # independent panel titles
        hspace = 0.55)                                                          # separation between panel rows
        fig.suptitle(                                                           # task and source heading
        rf'{tsk} $|$ {sta.c} $|$ N = {sta.D.P.nunique()}',                      # task, condition, and sample
        fontsize = 11,                                                          # heading size
        y        = 0.975)                                                       # heading position

        # label frequencies after No and Yes actions
        # ---------------------------------------------------------------------
        for a, key in enumerate(('lno', 'lys')):                                # top-row answer conditions
            ax  = axs[0, a]                                                     # current label-frequency axis
            val = sta.F[tsk][key]                                               # positive successor-label probabilities
            ok  = np.isfinite(val)                                              # observed conditioning identifiers
            pos = np.arange(1, n + 1)                                           # original identifier positions
            ax.bar(                                                             # empirical or simulated frequencies
            pos[ok],                                                            # observed current identifiers
            val[ok],                                                            # positive-label probabilities
            width     = 0.82,                                                   # compact adjacent bars
            color     = '0.88',                                                 # neutral fill
            edgecolor = '0.25',                                                 # restrained outline
            linewidth = 0.45)                                                   # outline width
            ax.plot(pos[~ok], np.zeros((~ok).sum()), 'x', color = 'black')      # undefined denominators
            ttl = ( r'$\hat{p}\left('                                           # conditional probability title
                    rf's_{{t+1}}^{{{j},1}} = 1 | '                              # positive successor label
                    rf's_t^{{{j},2}} = i,'                                      # current identifier
                    rf'a_t = a^{{{j},{a}}}\right)$')                            # selected response action
            ax.set_title(ttl, fontsize = 8, pad = 5)                            # panel estimand
            ax.set_xlabel(rf'$s_t^{{{j},2}}$', fontsize = 9)                    # current identifier
            ax.set_xlim(0.4, n + 0.6)                                           # catalogue bounds
            ax.set_ylim(0, 1)                                                   # probability range
            ax.set_xticks(pos)                                                  # all PTD and RMC identifiers
            ax.set_yticks((0, 0.5, 1))                                          # interpretable probability ticks
            ax.tick_params(labelsize = 6, length = 2, pad = 2)                  # compact tick labels
            ax.text(                                                            # panel letter left of axis
            -0.12, 1.12, 'AB'[a],                                               # label and relative position
            transform = ax.transAxes,                                           # axis-relative placement
            fontsize  = 11,                                                     # panel-label size
            va        = 'top')                                                  # align with panel title

        # conditional identifier frequencies
        # ---------------------------------------------------------------------
        for o in (0, 1):                                                        # current stimulus labels
            for v in (0, 1):                                                    # successor stimulus labels
                ax  = axs[o + 1, v]                                             # current heatmap axis
                mat = sta.F[tsk][evt[o][v]]                                     # conditional identifier matrix
                sym = 's'                                                       # original identifier symbol
                txt = [str(i + 1) for i in range(n)]                            # original identifier labels
                cmp = np.ones(n, dtype = bool)                                  # compatible current rows
                if var and v == 1:                                              # blockwise PTD target ranks
                    sym = r'\tilde{s}'                                          # recoded identifier symbol
                    cmp = (np.arange(n) < k) == o                               # compatible target-rank rows
                    txt = [rf'$t_{{{i + 1}}}$' for i in range(k)]               # target ranks
                    txt += [rf'$n_{{{i + 1}}}$' for i in range(n - k)]          # non-target ranks
                elif not var:                                                   # fixed RMC label mapping
                    lab = tab.fixed_label.to_numpy(dtype = int)                 # catalogue label function
                    cmp = lab == o                                              # compatible original identifiers
                val  = mat[np.isfinite(mat)]                                    # defined probability values
                vmax = max(0.1, np.ceil(val.max(initial = 0) * 10) / 10)        # readable shared range
                im   = ax.imshow(                                               # conditional frequency heatmap
                np.ma.masked_invalid(mat),                                      # preserve undefined rows
                cmap          = 'Greys',                                        # neutral probability scale
                norm          = Normalize(0, vmax),                             # comparable linear scale
                aspect        = 'auto',                                         # fill available panel area
                interpolation = 'nearest')                                      # retain exact cells
                for rid in np.flatnonzero(~cmp):                                # structurally incompatible rows
                    ax.add_patch(Rectangle(                                     # hatched row overlay
                    (-0.5, rid - 0.5),                                          # lower-left cell boundary
                    n, 1,                                                       # complete matrix row
                    facecolor = 'white',                                        # distinguish from zero probability
                    edgecolor = '0.8',                                          # light hatch color
                    hatch     = '///',                                          # incompatible conditioning event
                    linewidth = 0.2))                                           # restrained border
                bad  = np.flatnonzero(~cmp)                                     # incompatible row indices
                cut  = np.flatnonzero(np.diff(bad) > 1) + 1                     # contiguous row groups
                for seg in np.split(bad, cut):                                  # grouped N/A labels
                    if len(seg):                                                # nonempty incompatible group
                        ax.text(                                                # incompatibility annotation
                        (n - 1) / 2, seg.mean(), 'N/A',                         # center of the hatched area
                        ha       = 'center',                                    # horizontal alignment
                        va       = 'center',                                    # vertical alignment
                        fontsize = 5.5,                                         # restrained annotation size
                        bbox     = dict(facecolor = 'white', pad = 2))          # text backing
                if n <= 18:                                                     # readable numerical cell labels
                    for rid, row in enumerate(mat):                             # displayed current identifiers
                        for cid, prb in enumerate(row):                         # displayed successor identifiers
                            if np.isfinite(prb) and cmp[rid]:                   # defined compatible cells
                                col = 'white' if prb > 0.55 * vmax else 'black' # contrast with background
                                ax.text(                                        # rounded cell probability
                                cid, rid, f'{prb:.3f}',                         # location and formatted value
                                ha       = 'center',                            # horizontal centering
                                va       = 'center',                            # vertical centering
                                fontsize = 3.0 if n == 18 else 6.3,             # task-specific readability
                                color      = col,                               # contrasting text color
                                fontfamily = 'cmr10',                           # native Computer Modern
                                usetex     = False)                             # efficient native rendering
                ttl = ( r'$\hat{p}\left('                                       # conditional probability title
                        rf'{sym}_{{t+1}}^{{{j},2}} = k | '                      # successor identifier
                        rf's_{{t+1}}^{{{j},1}} = {v},'                          # successor label
                        rf's_t^{{{j},1}} = {o},'                                # current label
                        rf'{sym}_t^{{{j},2}} = i,'                              # current identifier
                        rf'a_t \in A^{{{j}}}\right)$')                          # selected task action
                ax.set_title(ttl, fontsize = 7.2, pad = 5)                      # panel estimand
                ax.set_xlabel(                                                  # successor identifier
                rf'${sym}_{{t+1}}^{{{j},2}} = k$',                              # mathematical axis label
                fontsize = 8)                                                   # axis-label size
                ax.set_ylabel(                                                  # current identifier
                rf'${sym}_t^{{{j},2}} = i$',                                    # mathematical axis label
                fontsize = 8, rotation = 0, labelpad = 18)                      # horizontal label near axis
                ax.set_xticks(np.arange(n), txt)                                # successor identifiers
                ax.set_yticks(np.arange(n), txt)                                # current identifiers
                ax.tick_params(labelsize = 5.5, length = 0, pad = 1)            # compact matrix ticks
                cbr = fig.colorbar(im, ax = ax, fraction = 0.035, pad = 0.025)  # probability scale
                cbr.ax.tick_params(labelsize = 6, length = 2, pad = 1)          # compact colorbar labels
                idx = 2 + 2 * o + v                                             # panel-letter index
                ax.text(                                                        # panel letter left of axis
                -0.12, 1.12, 'ABCDEF'[idx],                                     # label and relative position
                transform = ax.transAxes,                                       # axis-relative placement
                fontsize  = 11,                                                 # panel-label size
                va        = 'top')                                              # align with panel title

        # output
        # ---------------------------------------------------------------------
        for ext in ('png', 'pdf'):                                              # raster and vector formats
            file = sta.f / f'{nam[tsk]}.{ext}'                                  # Supplementary output path
            fig.savefig(file, dpi = 360)                                        # high-resolution figure
            sta.ff.append(file)                                                 # retain output order
        plt.close(fig)                                                          # release figure resources
    return sta                                                                  # written Supplementary figure paths


# Supplementary figure specifications
# -----------------------------------------------------------------------------
if __name__ == '__main__':                                                      # render all validation figures
    obs       = SimpleNamespace()                                               # observed plotting state
    obs.d     = ddir                                                            # observed participant files
    obs.m     = ddir                                                            # design and stimulus metadata
    obs.p     = np.arange(1, 50)                                                # complete participant cohort
    obs.c     = 'PTD-RMC'                                                       # validated paradigm
    obs.f     = fdir                                                            # Supplementary figure output
    obs.plt   = abm_figure(plt)                                                 # configured plotting module
    obs.src   = 'experiment'                                                    # experimental output mapping
    obs       = abm_describe(obs)                                               # load observed condition data
    obs       = abm_count(obs)                                                  # observed transition frequencies
    obs       = abm_figure_S1(obs)                                              # Figures S1A and S1C

    sim       = SimpleNamespace()                                               # simulated plotting state
    sim.D     = pd.read_csv(                                                    # cached A0 validation trials
    vdir / 'abm_validation_A0.csv.gz',                                          # compressed simulated cohort
    sep = '\t')                                                                # derivative-compatible delimiter
    sim.m     = ddir                                                            # design and stimulus metadata
    sim.c     = 'PTD-RMC'                                                       # validated paradigm
    sim.f     = fdir                                                            # Supplementary figure output
    sim.plt   = obs.plt                                                         # configured plotting module
    sim.src   = 'simulation'                                                    # simulation output mapping
    sim       = abm_count(sim)                                                  # simulated transition frequencies
    sim       = abm_figure_S1(sim)                                              # Figures S1B and S1D
