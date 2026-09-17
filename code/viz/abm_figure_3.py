"""
Visualize fixed-temperature A1 likelihood slices and parameter estimates.
"""

# initialization
# -----------------------------------------------------------------------------
import sys                                                                      # system tools
from pathlib import Path                                                        # path management

import matplotlib.pyplot as plt                                                 # plotting interface
import numpy as np                                                              # numerical operations
import pandas as pd                                                             # evaluation data loading

# directory management
# -----------------------------------------------------------------------------
src                   = Path(__file__).resolve().parent                         # visualization directory
udir                  = src.parent / 'abm'                                      # project utilities directory
root                  = src.parent.parent                                       # project root directory
ddir                  = root / 'data' / 'evaluation'                            # evaluation data directory
ldir                  = root / 'data' / 'likelihood'                            # cached conditional likelihood surfaces
fdir                  = root / 'figures'                                        # figure output directory
sys.path.append(str(udir))                                                      # add project utilities to module search path

# project utility import
# -----------------------------------------------------------------------------
from abm_figure import abm_figure                                               # shared figure defaults

# data preparation
# -----------------------------------------------------------------------------
eva                   = pd.read_pickle(ddir / 'sub-all-eva.pkl')                # aggregate model evaluation
mid                   = eva.M.index('A1')                                       # agentic model index
par                   = np.vstack([row[mid] for row in eva.mls])                # beta, sigma, and tau estimates
sel                   = np.all(np.isfinite(par), axis = 1)                      # participants with finite estimates
par                   = par[sel]                                                # finite parameter estimates

# likelihood surfaces and parameter projections
# -----------------------------------------------------------------------------
plt                   = abm_figure(plt)                                         # shared pyplot defaults
fig, axs              = plt.subplots(                                           # likelihood and parameter rows
2,                                                                              # two subplot rows
3,                                                                              # three subplot columns
figsize               = (10, 6.5),                                              # compact landscape figure
constrained_layout    = True)                                                   # automatic panel spacing
cmap                  = plt.get_cmap('Blues')                                   # shared blue color map
ax                    = axs[1]                                                  # lower-row parameter projections
col                   = cmap(0.70)                                              # participant marker color

# fixed-temperature log likelihood surfaces
# -----------------------------------------------------------------------------
for j, pid in enumerate((8, 46, 11)):                                           # participant column order
    file              = ldir / f'sub-{pid:03}-plot-llh.pkl'                     # independent plotting cache
    if not file.exists():                                                       # explain how to create missing surfaces
        raise FileNotFoundError(                                                # actionable cache error
        f'{file} is missing. Run code/abm_likelihood.py first.')                # generation entry
    pro               = pd.read_pickle(file)                                    # numeric axes and log likelihood matrix
    idx               = list(eva.p).index(pid)                                  # participant position in aggregate fits
    fit               = np.asarray(eva.mls[idx][mid])                           # current joint estimate
    if not np.array_equal(pro.fit, fit):                                        # prevent a stale fitted marker
        raise ValueError(f'Regenerate the likelihood cache for {pid}.')         # fit
    if not np.all(np.isfinite(pro.llh)):                                        # partial runs cannot form a surface
        raise ValueError(f'Incomplete likelihood cache for {pid}.')             # resume
    top               = axs[0, j]                                               # upper-row surface panel
    hi                = float(np.max(pro.llh))                                  # panel-specific likelihood maximum
    lo                = hi - 30                                                 # show likelihood structure near the maximum
    img               = top.pcolormesh(                                         # heatmap on physical parameter coordinates
    pro.b,                                                                      # increasing beta axis
    pro.s,                                                                      # increasing sigma axis
    pro.llh,                                                                    # rows are sigma and columns are beta
    shading           = 'nearest',                                              # cells centered on evaluated coordinates
    cmap              = 'coolwarm',                                             # reference likelihood color map
    vmin              = lo,                                                     # lower color limit
    vmax              = hi,                                                     # upper color limit
    rasterized        = True)                                                   # compact vector output
    top.contour(                                                                # likelihood differences reveal both parameter directions
    pro.b,                                                                      # beta support
    pro.s,                                                                      # sigma support
    pro.llh,                                                                    # conditional log likelihood
    levels            = hi - np.array((20, 10, 5, 2, 0.5)),                     # increasing likelihood levels
    colors            = 'black',                                                # visible against the heatmap
    linewidths        = 0.45,                                                   # fine contours
    alpha             = 0.55)                                                   # subdued contour lines
    pos               = np.unravel_index(np.argmax(pro.llh), pro.llh.shape)     # grid maximum
    top.plot(                                                                   # mark the maximum found on the evaluated grid
    pro.b[pos[1]],                                                              # beta at the grid maximum
    pro.s[pos[0]],                                                              # sigma at the grid maximum
    marker            = '+',                                                    # reference-style white cross
    color             = 'white',                                                # visible maximum marker
    markersize        = 9,                                                      # maximum marker size
    markeredgewidth   = 1.5,                                                    # maximum marker stroke
    zorder            = 4,                                                      # draw above the boundary spine
    clip_on           = False)                                                  # retain the sigma-zero boundary marker
    if pid == 8:                                                                # show transition and plateau on one beta axis
        top.set_xscale('log')                                                   # beta scale
        top.set_xticks((0.1, 0.3, 1, 3, 5))                                     # physical beta ticks
        top.set_xticklabels(('0.1', '0.3', '1', '3', '5'))                      # readable log-axis labels
        top.tick_params(axis = 'x', which = 'minor', labelbottom = False)       # suppress overlapping minor labels
    top.set_xlim(pro.b[0], pro.b[-1])                                           # evaluated beta range
    top.set_ylim(pro.s[0], pro.s[-1])                                           # evaluated sigma range
    lab               = r'$\beta$ (log scale)' if pid == 8 else r'$\beta$'      # axis scaling label
    top.set_xlabel(lab, fontsize = 14)                                          # imbalance cost
    top.set_ylabel(r'$\sigma$', fontsize = 14, rotation = 0, labelpad = 6)      # cost
    top.set_title(                                                              # participant and fixed nuisance parameter
    rf'sub-{pid:03}, $\hat{{\tau}} = {fit[2]:.3f}$',                            # fixed temperature
    fontsize          = 11,                                                     # title size
    pad               = 8)                                                      # compact panel title
    top.tick_params(labelsize = 9)                                              # physical parameter ticks
    if pid != 8:                                                                # linear beta axes with small parameter values
        top.ticklabel_format(axis = 'x', style = 'sci', scilimits = (-3, 3))    # scale
    cb                = fig.colorbar(                                           # horizontal log likelihood scale increasing rightward
    img,                                                                        # corresponding heatmap
    ax                = top,                                                    # aligned with the surface panel
    orientation       = 'horizontal',                                           # increasing likelihood from left to right
    location          = 'top',                                                  # scale above the surface
    shrink            = 0.9,                                                    # inset color-bar length
    pad               = 0.06,                                                   # separation from the panel title
    aspect            = 28,                                                     # slim color bar
    extend            = 'max')                                                  # arrow at the higher-likelihood end
    vals              = np.linspace(lo, hi, 4)                                  # include both color limits
    cb.set_ticks(vals)                                                          # evenly spaced likelihood ticks
    cb.set_ticklabels([f'{val:.0f}' for val in vals])                           # integer likelihood labels
    cb.set_label('Log likelihood', fontsize = 9)                                # absolute likelihood units
    cb.ax.tick_params(labelsize = 8)                                            # color-bar tick labels
    cb.outline.set_visible(False)                                               # reference-style unobtrusive color bar

# tau-sigma projection
# -----------------------------------------------------------------------------
ax[0].scatter(                                                                  # participant estimates
par[:, 2],                                                                      # tau estimates
par[:, 1],                                                                      # sigma estimates
s                     = 38,                                                     # marker size
color                 = col,                                                    # marker color
alpha                 = 0.85,                                                   # marker transparency
edgecolors            = 'white',                                                # marker edges
linewidths            = 0.6)                                                    # marker-edge width
ax[0].set_xlabel(r'$\hat{\tau}$', fontsize = 14)                                # estimated decision temperature axis
ax[0].set_ylabel(r'$\hat{\sigma}$', fontsize = 14, rotation = 0, labelpad = 6)  # estimated switching cost axis

# tau-beta projection
# -----------------------------------------------------------------------------
ax[1].scatter(                                                                  # participant estimates
par[:, 2],                                                                      # tau estimates
par[:, 0],                                                                      # beta estimates
s                     = 38,                                                     # marker size
color                 = col,                                                    # marker color
alpha                 = 0.85,                                                   # marker transparency
edgecolors            = 'white',                                                # marker edges
linewidths            = 0.6)                                                    # marker-edge width
ax[1].set_xlabel(r'$\hat{\tau}$', fontsize = 14)                                # estimated decision temperature axis
ax[1].set_ylabel(r'$\hat{\beta}$', fontsize = 14, rotation = 0, labelpad = 6)   # estimated imbalance cost axis

# sigma-beta projection
# -----------------------------------------------------------------------------
ax[2].scatter(                                                                  # participant estimates
par[:, 1],                                                                      # sigma estimates
par[:, 0],                                                                      # beta estimates
s                     = 38,                                                     # marker size
color                 = col,                                                    # marker color
alpha                 = 0.85,                                                   # marker transparency
edgecolors            = 'white',                                                # marker edges
linewidths            = 0.6)                                                    # marker-edge width
ax[2].set_xlabel(r'$\hat{\sigma}$', fontsize = 14)                              # estimated switching cost axis
ax[2].set_ylabel(r'$\hat{\beta}$', fontsize = 14, rotation = 0, labelpad = 6)   # estimated imbalance cost axis

# beta scaling
# -----------------------------------------------------------------------------
for i in (1, 2):                                                                # panels with beta ordinates
    ax[i].set_yscale('symlog', linthresh = 0.05, linscale = 0.8)                # compress beta while retaining zero

# shared panel formatting
# -----------------------------------------------------------------------------
for i, lab in enumerate(('D', 'E', 'F')):                                       # subplot iterations
    ax[i].text(                                                                 # subplot label
    -0.18,                                                                      # position left of the y-axis
    1.03,                                                                       # position above the axes
    lab,                                                                        # subplot letter
    transform         = ax[i].transAxes,                                        # axes-relative coordinates
    ha                = 'right',                                                # align toward the y-axis
    va                = 'bottom',                                               # align above the axes
    fontsize          = 16,                                                     # subplot-label font size
    clip_on           = False)                                                  # retain label outside the axes
    ax[i].tick_params(labelsize = 9)                                            # axis tick labels
    ax[i].grid(True, linewidth = 0.5, color = [0.9, 0.9, 0.9])                  # reference grid

# upper-row panel labels
# -----------------------------------------------------------------------------
for j, lab in enumerate(('A', 'B', 'C')):                                       # likelihood panel letters
    axs[0, j].text(                                                             # letter outside the surface
    -0.18,                                                                      # horizontal label position
    1.03,                                                                       # vertical label position
    lab,                                                                        # axes-relative position and panel letter
    transform         = axs[0, j].transAxes,                                    # panel coordinates
    ha                = 'right',                                                # horizontal alignment
    va                = 'bottom',                                               # vertical alignment
    fontsize          = 16,                                                     # panel letter size
    clip_on           = False)                                                  # label style

# figure saving
# -----------------------------------------------------------------------------
fdir.mkdir(parents = True, exist_ok = True)                                     # ensure output directory exists
for ext in ('png', 'pdf'):                                                      # raster and vector formats
    file              = fdir / f'abm_figure_3.{ext}'                            # output figure path
    fig.savefig(file, dpi = 400, format = ext, bbox_inches = 'tight')           # write publication figure

plt.close(fig)                                                                  # release figure resources
