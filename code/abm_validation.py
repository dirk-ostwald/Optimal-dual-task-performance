"""
Validate experimental PTD-RMC state statistics with A0 simulations.
"""

# initialization
# -----------------------------------------------------------------------------
import json                                                                     # experimental design loading
import sys                                                                      # module search path management
from pathlib import Path                                                        # project paths
from types import SimpleNamespace                                               # shared analysis state

import numpy as np                                                              # numerical operations
import pandas as pd                                                             # trial tables and validation output

# directory management
# -----------------------------------------------------------------------------
wdir = Path(__file__).resolve().parent                                          # code directory
rdir = wdir.parent                                                              # project root directory
ddir = rdir / 'data' / 'experiment' / 'derivatives'                             # observed data and design metadata
odir = rdir / 'data' / 'validation'                                             # validation artifact directory
udir = wdir / 'abm'                                                             # shared ABM utilities

# ABM utility imports
# -----------------------------------------------------------------------------
sys.path.append(str(udir))                                                      # make shared ABM modules importable
from abm_count import abm_count                                                 # pooled transition frequencies
from abm_describe import abm_describe                                           # observed participant data loading
from abm_simulate import abm_simulate                                           # design-matched block simulation


def abm_validation(sta):                                                        # experimental-statistics validation
    """
    Compare observed PTD-RMC transition frequencies with A0 simulations.

    Input
        sta : SimpleNamespace() object with required attributes:
            .d    : observed derivative directory (pathlib.Path)
            .o    : validation summary directory (pathlib.Path)
            .p    : participant identifiers (sequence of integers)
            .c    : condition label, 'PTD-RMC'
            .seed : simulation random seed (integer)

    Output
        sta : input SimpleNamespace with the new attributes:
            .obs : observed analysis state (SimpleNamespace)
            .sim : simulated analysis state (SimpleNamespace)
            .tab : observed-simulated comparison table (pandas.DataFrame)

    Each observed block is simulated with its original stimulus pools, target
    sets, and trial count. A0 is the only action-generating agent. Its uniform
    random actions avoid fitting behavior while sampling the full MDP state
    transition law under the experimentally realized designs.
    """
    # observed states
    # -------------------------------------------------------------------------
    obs       = SimpleNamespace()                                               # observed analysis state
    obs.d     = sta.d                                                           # observed participant files
    obs.m     = sta.d                                                           # design and stimulus metadata
    obs.p     = sta.p                                                           # complete participant cohort
    obs.c     = sta.c                                                           # PTD-RMC condition
    obs       = abm_describe(obs)                                               # load observed condition data
    obs       = abm_count(obs)                                                  # observed transition frequencies
    sta.o.mkdir(parents = True, exist_ok = True)                                # validation output directory

    # design-matched A0 simulations
    # -------------------------------------------------------------------------
    des       = json.loads((sta.d / 'design.json').read_text())                 # participant-specific designs
    cat       = pd.read_csv(sta.d / 'stimuli' / 'stimuli.csv')                  # stimulus catalogue
    dfs       = []                                                              # simulated block data frames
    rng       = np.random.get_state()                                           # preserve caller random state
    np.random.seed(sta.seed)                                                    # reproducible validation sample
    try:                                                                        # restore the caller state after simulation
        for p in sta.p:                                                         # participant-specific designs
            pid = str(int(p))                                                   # design-database participant key
            dat = obs.D[obs.D.P == p]                                           # observed participant condition
            num = dat.groupby(['R', 'B']).size()                                # matched block trial counts
            run = des['participants'][pid][sta.c]                               # participant PTD-RMC runs
            for rid in sorted(run, key = int):                                  # experimental runs
                for bid in sorted(run[rid], key = int):                         # PTD-RMC blocks
                    cfg       = run[rid][bid]                                   # block-specific stimulus design
                    src       = int(cfg['source_block'])                        # derivative block identifier
                    sim       = SimpleNamespace()                               # block simulation settings
                    sim.X     = SimpleNamespace()                               # experimental design settings
                    sim.X.D   = des                                             # complete design database
                    sim.X.SL  = cat                                             # complete stimulus catalogue
                    sim.X.T1  = 'PTD'                                           # Task 1 identifier
                    sim.X.T2  = 'RMC'                                           # Task 2 identifier
                    sim.X.P   = pid                                             # participant identifier
                    sim.X.R   = rid                                             # run identifier
                    sim.X.B   = bid                                             # within-run block identifier
                    sim.X.n   = int(num.loc[int(rid), src])                     # observed block trial count
                    sim.O     = np.nan                                          # response onset unavailable
                    sim.agent = 'A0'                                            # uniform random action generator
                    sim.θ     = None                                            # A0 has no fitted parameters
                    sim       = abm_simulate(sim)                               # simulate the full MDP block
                    df        = sim.P.D.copy()                                  # simulated block trials
                    df.insert(4, 'T', ' | '.join(cfg['S1T']))                   # block-specific PTD targets
                    dfs.append(df)                                              # pooled simulated cohort
            print(f'Simulated sub-{pid.zfill(3)}', flush = True)                # participant-level progress
    finally:                                                                    # random-state isolation
        np.random.set_state(rng)                                                # restore caller random state

    sim       = SimpleNamespace()                                               # simulated analysis state
    sim.D     = pd.concat(dfs, ignore_index = True)                             # all design-matched A0 trials
    sim.m     = sta.d                                                           # design and stimulus metadata
    sim.c     = sta.c                                                           # PTD-RMC condition
    sim       = abm_count(sim)                                                  # simulated transition frequencies
    sim.D.to_csv(                                                               # reusable validation simulation
    sta.o / 'abm_validation_A0.csv.gz',                                         # compressed simulated cohort
    sep   = '\t',                                                               # derivative-compatible delimiter
    index = False)                                                              # omit data frame index

    # observed-simulated comparisons
    # -------------------------------------------------------------------------
    evt       = ('lno', 'lys', 'inn', 'inp', 'ipn', 'ipp')                      # six frequency estimands
    rows      = []                                                              # comparison records
    for tsk in ('PTD', 'RMC'):                                                  # validation tasks only
        for key in evt:                                                         # panel-specific frequency arrays
            x   = obs.F[tsk][key].ravel()                                       # observed frequencies
            y   = sim.F[tsk][key].ravel()                                       # simulated frequencies
            sel = np.isfinite(x) & np.isfinite(y)                               # shared estimable cells
            dif = x[sel] - y[sel]                                               # cellwise discrepancies
            cor = np.nan                                                        # undefined for constant arrays
            if sel.sum() > 1 and np.std(x[sel]) and np.std(y[sel]):             # variable paired arrays
                cor = float(np.corrcoef(x[sel], y[sel])[0, 1])                  # Pearson pattern correlation
            rows.append(dict(                                                   # one panel-family comparison
            task = tsk, event = key, n = int(sel.sum()),                        # task, event, and shared cells
            mae  = float(np.mean(np.abs(dif))),                                 # mean absolute discrepancy
            rmse = float(np.sqrt(np.mean(dif ** 2))),                           # root mean squared discrepancy
            cor  = cor))                                                        # observed-simulated correlation
    tab       = pd.DataFrame(rows)                                              # validation summary table
    tab.to_csv(sta.o / 'abm_validation_summary.csv', index = False)             # reproducible numerical comparison

    sta.obs   = obs                                                             # observed analysis results
    sta.sim   = sim                                                             # simulated analysis results
    sta.tab   = tab                                                             # numerical validation summary
    return sta                                                                  # completed experimental-statistics validation


# validation specifications
# -----------------------------------------------------------------------------
if __name__ == '__main__':                                                      # reproducible complete validation
    sta       = SimpleNamespace()                                               # validation settings
    sta.d     = ddir                                                            # observed data and metadata
    sta.o     = odir                                                            # numerical validation output
    sta.p     = np.arange(1, 50)                                                # all experimental participants
    sta.c     = 'PTD-RMC'                                                       # validated paradigm
    sta.seed  = 20260916                                                        # fixed simulation seed
    sta       = abm_validation(sta)                                             # run simulation and analyses
    print(sta.tab.to_string(index = False))                                     # concise validation summary
