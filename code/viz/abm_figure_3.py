"""
Visualize participant-specific A1 parameter estimates in two dimensions.
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
src  = Path(__file__).resolve().parent                                          # visualization directory
udir = src.parent / 'abm'                                                       # project utilities directory
root = src.parent.parent                                                        # project root directory
ddir = root / 'data' / 'evaluation'                                             # evaluation data directory
fdir = root / 'figures'                                                         # figure output directory
sys.path.append(str(udir))                                                      # add project utilities to module search path

# project utility import
# -----------------------------------------------------------------------------
from abm_figure import abm_figure                                               # shared figure defaults

# data preparation
# -----------------------------------------------------------------------------
eva = pd.read_pickle(ddir / 'sub-all-eva.pkl')                                  # aggregate model evaluation
mid = eva.M.index('A1')                                                         # agentic model index
par = np.vstack([row[mid] for row in eva.mls])                                  # beta, sigma, and tau estimates
sel = np.all(np.isfinite(par), axis = 1)                                        # participants with finite estimates
par = par[sel]                                                                  # finite parameter estimates

# two-dimensional scatter plots
# -----------------------------------------------------------------------------
plt = abm_figure(plt)                                                           # shared pyplot defaults
fig, ax = plt.subplots(                                                         # one-row parameter projections
1,                                                                              # one subplot row
3,                                                                              # three subplot columns
figsize            = (9, 3),                                                    # compact landscape figure
constrained_layout = True)                                                      # automatic panel spacing
cmap = plt.get_cmap('Blues')                                                    # shared blue color map
col  = cmap(0.70)                                                               # participant marker color

# tau-sigma projection
# -----------------------------------------------------------------------------
ax[0].scatter(                                                                  # participant estimates
par[:, 2],                                                                      # tau estimates
par[:, 1],                                                                      # sigma estimates
s          = 38,                                                                # marker size
color      = col,                                                               # marker color
alpha      = 0.85,                                                              # marker transparency
edgecolors = 'white',                                                           # marker edges
linewidths = 0.6)                                                               # marker-edge width
ax[0].set_xlabel(r'$\hat{\tau}$', fontsize = 14)                                # estimated decision temperature axis
ax[0].set_ylabel(r'$\hat{\sigma}$', fontsize = 14, rotation = 0, labelpad = 6)  # estimated switching cost axis

# tau-beta projection
# -----------------------------------------------------------------------------
ax[1].scatter(                                                                  # participant estimates
par[:, 2],                                                                      # tau estimates
par[:, 0],                                                                      # beta estimates
s          = 38,                                                                # marker size
color      = col,                                                               # marker color
alpha      = 0.85,                                                              # marker transparency
edgecolors = 'white',                                                           # marker edges
linewidths = 0.6)                                                               # marker-edge width
ax[1].set_xlabel(r'$\hat{\tau}$', fontsize = 14)                                # estimated decision temperature axis
ax[1].set_ylabel(r'$\hat{\beta}$', fontsize = 14, rotation = 0, labelpad = 6)   # estimated imbalance cost axis

# sigma-beta projection
# -----------------------------------------------------------------------------
ax[2].scatter(                                                                  # participant estimates
par[:, 1],                                                                      # sigma estimates
par[:, 0],                                                                      # beta estimates
s          = 38,                                                                # marker size
color      = col,                                                               # marker color
alpha      = 0.85,                                                              # marker transparency
edgecolors = 'white',                                                           # marker edges
linewidths = 0.6)                                                               # marker-edge width
ax[2].set_xlabel(r'$\hat{\sigma}$', fontsize = 14)                              # estimated switching cost axis
ax[2].set_ylabel(r'$\hat{\beta}$', fontsize = 14, rotation = 0, labelpad = 6)   # estimated imbalance cost axis

# beta scaling
# -----------------------------------------------------------------------------
for i in (1, 2):                                                                # panels with beta ordinates
    ax[i].set_yscale('symlog', linthresh = 0.05, linscale = 0.8)                # compress beta while retaining zero

# shared panel formatting
# -----------------------------------------------------------------------------
for i, lab in enumerate(('A', 'B', 'C')):                                       # subplot iterations
    ax[i].text(                                                                 # subplot label
    -0.18,                                                                      # position left of the y-axis
    1.03,                                                                       # position above the axes
    lab,                                                                        # subplot letter
    transform = ax[i].transAxes,                                                # axes-relative coordinates
    ha        = 'right',                                                        # align toward the y-axis
    va        = 'bottom',                                                       # align above the axes
    fontsize  = 16,                                                             # subplot-label font size
    clip_on   = False)                                                          # retain label outside the axes
    ax[i].tick_params(labelsize = 9)                                            # axis tick labels
    ax[i].grid(True, linewidth = 0.5, color = [0.9, 0.9, 0.9])                  # reference grid

# figure saving
# -----------------------------------------------------------------------------
fdir.mkdir(parents = True, exist_ok = True)                                     # ensure output directory exists
for ext in ('png', 'pdf'):                                                      # raster and vector formats
    file = fdir / f'abm_figure_3.{ext}'                                         # output figure path
    fig.savefig(                                                                # write publication figure
    file,                                                                       # output path
    dpi         = 400,                                                          # high raster resolution
    format      = ext,                                                          # requested output format
    bbox_inches = 'tight')                                                      # retain external subplot labels
plt.close(fig)                                                                  # release figure resources
