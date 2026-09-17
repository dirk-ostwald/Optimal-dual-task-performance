"""
Evaluate descriptive statistics and shared figures for experimental data.
"""

# initialization
# -----------------------------------------------------------------------------
import sys                                                                      # system tools
from pathlib import Path                                                        # path management
from types import SimpleNamespace                                               # shared analysis state

import matplotlib as mpl                                                        # backend management
import numpy as np                                                              # participant indices

mpl.use('Agg')                                                                  # non-interactive plotting backend
import matplotlib.pyplot as plt                                                 # visualization

# directory management
# -----------------------------------------------------------------------------
wdir = Path(__file__).resolve().parent                                          # script directory
rdir = wdir.parent                                                              # project root directory
ddir = rdir / 'data' / 'experiment' / 'derivatives'                             # experimental data and metadata
fdir = rdir / 'figures' / 'experiment'                                          # experimental figure directory
udir = wdir / 'abm'                                                             # project utilities directory

# ABM utility imports
# -----------------------------------------------------------------------------
sys.path.append(str(udir))                                                      # add utility directory
from abm_describe import abm_describe                                           # descriptive statistics
from abm_figure import abm_figure                                               # Matplotlib defaults

# analysis specifications
# -----------------------------------------------------------------------------
cond    = ['PTD-RMC']                                                           # evaluated conditions
sta     = SimpleNamespace()                                                     # shared analysis state
sta.d   = ddir                                                                  # participant data directory
sta.m   = ddir                                                                  # design and stimulus metadata directory
sta.f   = fdir                                                                  # figure output directory
sta.p   = np.arange(1, 50)                                                      # experimental participants
sta.nt  = 30                                                                    # upper-row trial window
sta.dec = {'RMC': 'rotation only?', 'DPC': 'same parity?'}                      # decisions
sta.plt = abm_figure(plt)                                                       # configured pyplot module

# condition-specific statistics and figures
# -----------------------------------------------------------------------------
for c in cond:                                                                  # condition iterations
    sta.c = c                                                                   # condition label
    sta   = abm_describe(sta)                                                   # descriptive statistics
