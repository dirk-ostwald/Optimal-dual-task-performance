"""
Create the participant-behavior panels for Figure 2.
"""

# initialization
# -----------------------------------------------------------------------------
import sys                                                                      # system tools
from pathlib import Path                                                        # path management
from types import SimpleNamespace                                               # shared analysis state

import matplotlib as mpl                                                        # backend management
import numpy as np                                                              # numerical operations
from matplotlib.image import imread                                             # image loading
from matplotlib.patches import Rectangle                                        # target-frame drawing

mpl.use('Agg')                                                                  # non-interactive plotting backend
import matplotlib.pyplot as plt                                                 # plotting interface

# directory management
# -----------------------------------------------------------------------------
src  = Path(__file__).resolve().parent                                          # visualization directory
root = src.parent.parent                                                        # project root directory
ddir = root / 'data' / 'experiment' / 'derivatives'                             # experimental data directory
fdir = root / 'figures'                                                         # figure output directory
udir = src.parent / 'abm'                                                       # project utilities directory
sys.path.append(str(udir))                                                      # add project utilities to module search path

# project utility imports
# -----------------------------------------------------------------------------
from abm_describe import abm_describe                                           # descriptive statistics
from abm_figure import abm_figure                                               # shared figure defaults

def abm_plot_behavior(sta):                                                     # participant behavior figures
    """
    Plot and save participant behavior for one FCDT condition.

    Input
        sta : SimpleNamespace() object with required attributes:
            .D   : condition-specific trial data (pandas.DataFrame)
            .p   : participant indices (sequence of integers)
            .c   : condition label, e.g. 'PTD-RMC'
            .m   : experimental metadata directory (pathlib.Path)
            .f   : figure output directory (pathlib.Path)
            .plt : configured matplotlib.pyplot module
            .nt  : number of trials in the upper plot (integer)
            .dec : classification decision labels by task (dict)
            .a_* : participant statistics from abm_describe

    Output
        sta : input SimpleNamespace with the new attribute:
            .bf  : written participant PNG and PDF paths

    Optional .lab supplies Figure 2 panel labels and writes the files directly
    to .f. Optional .h supplies a simulation heading. Optional .fc controls
    condition subdirectories for files without panel labels (default True).
    """
    # shared plotting specifications
    # -------------------------------------------------------------------------
    plt    = sta.plt                                                            # configured pyplot module
    ddir   = sta.m                                                              # stimulus metadata directory
    nt     = sta.nt                                                             # upper-row trial window
    dec    = sta.dec                                                            # classification decisions
    c      = sta.c                                                              # condition label
    ta, tb = c.split('-')                                                       # component-task names
    lab    = getattr(sta, 'lab', None)                                          # optional Figure 2 panel labels
    odir   = sta.f if lab else sta.f / 'participants'                           # requested output directory
    if not lab and getattr(sta, 'fc', True):                                    # condition-specific legacy output
        odir = odir / c                                                         # condition subdirectory
    odir.mkdir(parents = True, exist_ok = True)                                 # create output directory
    sta.bf = []                                                                 # written participant figures
    head   = getattr(sta, 'h', '')                                              # optional simulation agent and parameters
    head   = head + r'\\[4pt]' if head else ''                                  # additional simulation heading line
    fda   = {'ha' : 'right' , 'va': 'center', 'fontsize': 10, 'color': 'blue'}  # task A row-label font
    fdb   = {'ha' : 'right' , 'va': 'center', 'fontsize': 10, 'color': 'red'}   # task B row-label font
    fds   = {'ha' : 'center', 'va': 'center', 'fontsize': 10}                   # stimulus font
    fdt   = {'ha' : 'center', 'va': 'center', 'fontsize': 10, 'color': 'green'} # target font
    fdr   = {'ha' : 'center', 'va': 'center', 'fontsize': 10, 'color': 'black'} # response font
    fdl   = {'ha' : 'right' , 'va': 'center', 'fontsize': 10}                   # response row-label font

    # behavior
    # -------------------------------------------------------------------------
    for j, p in enumerate(sta.p):                                               # participant iterations
        dpc = sta.D[sta.D['P'] == p]                                            # participant-condition data
        dpc = dpc.sort_values(['R', 'B', 't']).reset_index(drop = True)         # chronological condition data
        fig = plt.figure(figsize = (10, 6.7), constrained_layout = True)        # compact participant-condition figure
        fig.get_layout_engine().set(w_pad = 0.10, h_pad = 0.08)                 # horizontal and vertical figure margins
        gs  = fig.add_gridspec(nrows = 2, ncols = 1, hspace = 0.10)             # closely spaced subplot grid

        # upper row: first nt trials
        # ---------------------------------------------------------------------
        win = dpc.head(nt)                                                      # trial-window data
        n   = win.shape[0]                                                      # number of displayed trials
        x   = np.arange(n)                                                      # zero-based trial axis
        trg = str(win['T'].iloc[0]).split(' | ') if n > 0 else []               # memory-search targets
        mem = []                                                                # target title labels
        key = set()                                                             # target keys

        # target title labels and matching keys
        for v in trg:                                                           # target iterations
            v    = Path(v.strip()).stem                                         # target stem
            part = v.split()                                                    # target tokens
            pair = len(part) == 2                                               # pair flag
            unit = all(len(par) == 1 for par in part)                           # single-character token flag
            v    = ''.join(part) if pair and unit else v                        # compact target pair
            key.add(v)                                                          # collect target key
            v    = v.split('_')[0] if '_' in v else v                           # remove image filename suffix
            mem.append(v)                                                       # collect target title label
        mem      = ', '.join(mem) if n > 0 else 'n/a'                           # target title
        trg      = key                                                          # memory-search target keys
        ax       = fig.add_subplot(gs[0, 0])                                    # upper axis initialization
        a        = win['a1'].to_numpy()                                         # task choices
        at       = x[a == 'A']                                                  # task A trials
        bt       = x[a == 'B']                                                  # task B trials
        ax.vlines(at, 0, 0.48, color = 'blue', linewidth = 1.20)                # task A choice stems
        ax.vlines(bt, 0, -0.48, color = 'red', linewidth = 1.20)                # task B choice stems
        ax.axhline(0, color = 'black', linewidth = 0.90)                        # middle line

        # trial-window stimulus and response annotations
        for k, row in win.iterrows():                                           # trial annotation loop
            tx             = k                                                  # zero-based trial x-position
            col            = 'blue' if row['a1'] == 'A' else 'red'              # selected-task color
            for s, y in [(row['sA'], 0.84), (row['sB'], 0.66)]:                 # stimulus rows
                s          = str(s)                                             # stimulus string
                stem       = Path(s).stem                                       # stimulus stem
                part       = stem.split()                                       # stimulus tokens
                pair       = len(part) == 2                                     # pair flag
                unit       = all(len(par) == 1 for par in part)                 # single-character token flag
                key        = ''.join(part) if pair and unit else stem           # stimulus key
                img        = ddir / 'stimuli' / 'images' / f'{stem}.png'        # derivative stimulus image
                tar        = key in trg                                         # target flag

                if '_' in stem and s != 'n/a':                                  # image stimulus
                    im     = imread(img).astype(float)                          # image array
                    im     = im / 255 if im.max() > 1 else im                   # normalize image
                    if tar:                                                     # target image
                        im = 0.62 * im + 0.38 * np.array([0.25, 1.00, 0.25])    # target tint
                    ax.imshow(                                                  # image stimulus visualization
                    im,                                                         # image array
                    extent = [tx - 0.32, tx + 0.32, y - 0.07, y + 0.07],        # image bounds
                    aspect = 'auto',                                            # aspect ratio
                    zorder = 3)                                                 # z-order
                    if tar:                                                     # target image
                        ax.add_patch(Rectangle(                                 # target frame patch
                        (tx - 0.32, y - 0.07),                                  # lower-left corner
                        0.64,                                                   # width
                        0.14,                                                   # height
                        edgecolor = 'green',                                    # edge color
                        facecolor = 'none',                                     # face color
                        linewidth = 0.65,                                       # line width
                        zorder    = 4))                                         # z-order and target frame addition
                else:                                                           # text stimulus
                    ax.text(tx, y, key, fdt if tar else fds)                    # target or non-target label

            a2             = 'NY'[int(row['a2'])]                               # answer label
            rt             = str(row['r'])                                      # reward label
            fdr['color']   = col                                                # response font color
            ax.text(tx, -0.66, a2, fdr)                                         # selected answer
            ax.text(tx, -0.84, rt, fdr)                                         # selected-answer accuracy

        # trial-window labels and limits
        ax.text(-0.85, 0.84, ta, fda)                                           # task A label before the first trial
        ax.text(-0.85, 0.66, tb, fdb)                                           # task B label before the first trial
        ax.text(-0.85, -0.66, 'Action', fdl)                                    # answer label before the first trial
        ax.text(-0.85, -0.84, 'Accuracy', fdl)                                  # accuracy label before the first trial
        ax.set_xlim(-2.90, n)                                                   # reserve label space inside the y-axis
        ax.set_ylim(-1.00, 1.00)                                                # y-limits
        ax.set_yticks([])                                                       # hide y-axis ticks
        ax.set_xticks(np.arange(0, n, 5))                                       # zero-based trial ticks
        ax.tick_params('x', labelsize = 10)                                     # x-axis tick labels
        ax.set_xlabel('Trial', fontsize = 12)                                   # x-axis label
        ax.set_title(                                                           # trial-window title
        r'\shortstack[c]{'                                                      # centered title with LaTeX-controlled spacing
        f'{head}'                                                               # agent and parameter settings for simulations
        f'SUB-{str(p).zfill(3)} $|$ {c} $|$ First {n} trials'                   # participant, condition, and window
        r'\\[7pt]'                                                              # explicit vertical gap between title lines
        f'{ta} targets: {mem} $|$ {tb} decision: {dec[tb]}'                     # targets and classification decision
        r'}',                                                                   # close title stack
        fontsize = 13,                                                          # title font size
        pad      = 16)                                                          # title distance from plot

        # lower row: all trials
        # ---------------------------------------------------------------------
        n  = dpc.shape[0]                                                       # number of trials
        x  = np.arange(n)                                                       # zero-based trial axis
        a  = dpc['a1'].to_numpy()                                               # task choices
        r  = dpc['r'].to_numpy()                                                # rewards
        at = x[a == 'A']                                                        # task A trials
        bt = x[a == 'B']                                                        # task B trials
        rt = np.cumsum(r) / np.arange(1, n + 1)                                 # cumulative reward rate
        ax = fig.add_subplot(gs[1, 0])                                          # lower axis initialization
        ax.vlines(at, 0, 1, color = 'blue', linewidth = 0.45)                   # task A choices
        ax.vlines(bt, 0, -1, color = 'red', linewidth = 0.45)                   # task B choices
        ax.axhline(0, color = 'black', linewidth = 0.8)                         # middle line
        ax.plot(x, rt, color = 'gray', linewidth = 0.8)                         # cumulative rewards
        ax.set_ylim(-1.2, 1.2)                                                  # y-limits
        ax.set_xlim(-1, n)                                                      # participant-specific trial range
        ax.set_yticks([])                                                       # hide y-axis ticks
        ax.tick_params(axis = 'x', labelsize = 10)                              # x-axis tick labels
        ax.set_xlabel('Trial', fontsize = 12)                                   # x-axis label
        ax.set_title(                                                           # all-trials title
        r'\shortstack[c]{'                                                      # centered title with LaTeX-controlled spacing
        f'{head}'                                                               # agent and parameter settings for simulations
        f'SUB-{str(p).zfill(3)} $|$ {c} $|$ All trials'                         # participant and condition
        r'\\[7pt]'                                                              # explicit vertical gap after heading
        f'Task switches: {sta.a_swi[j]:.0f},  '                                 # number of task switches
        f'Trials: {n:.0f},  '                                                   # number of trials
        f'Time: {sta.a_tim[j]:.1f} s,  '                                        # total time spent on trials
        f'Reward: {sta.a_rew[j]:.0f},  '                                        # number of rewarded trials
        f'Reward/trial: {sta.a_rpt[j]:.3f}'                                     # reward rate per trial
        r'}',                                                                   # close title stack
        fontsize = 13,                                                          # title font size
        pad      = 16)                                                          # title distance from plot

        # one figure per participant and condition
        for ext in ['png', 'pdf']:                                              # raster and publication formats
            if lab:                                                             # manuscript Figure 2 output
                file = odir / f'abm_figure_2{lab[j]}.{ext}'                     # panel-specific filename
            else:                                                               # reusable simulation output
                file = odir / f'abm-S1-{str(p).zfill(3)}-{c}.{ext}'             # participant-condition filename
            fig.savefig(file, dpi = 400)                                        # figure printing
            sta.bf.append(file)                                                 # retain written figure path
        plt.close(fig)                                                          # figure closure
    
    # output specification
    # -------------------------------------------------------------------------
    return sta                                                                  # attach written participant figure paths


# selected experimental figures
# -----------------------------------------------------------------------------
if __name__ == '__main__':                                                      # direct script execution
    sta     = SimpleNamespace()                                                 # shared plotting state
    sta.d   = ddir                                                              # participant data directory
    sta.m   = ddir                                                              # stimulus metadata directory
    sta.f   = fdir                                                              # figure output directory
    sta.p   = [8, 46, 11]                                                       # selected participants
    sta.lab = ('A', 'B', 'C')                                                   # Figure 2 panel labels
    sta.c   = 'PTD-RMC'                                                         # experimental condition
    sta.nt  = 30                                                                # upper-panel trial window
    sta.dec = {'RMC': 'rotation only?', 'DPC': 'same parity?'}                  # task decisions
    sta.plt = abm_figure(plt)                                                   # configured pyplot module
    sta     = abm_describe(sta)                                                 # participant statistics
    sta     = abm_plot_behavior(sta)                                            # Figure 2 panels
