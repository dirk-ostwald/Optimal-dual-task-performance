"""
Evaluate and cache fixed-temperature A1 log likelihood surfaces for Figure 3.

Run python code/abm_likelihood.py to generate or reuse the plotting caches.
ABM_LLH_STAGE selects coarse, pilot, or plot, with plot as the default.
ABM_LLH_PID optionally selects participant 8, 46, or 11.
ABM_LLH_RES overrides the base grid resolution, 7 for exploration and 25
for plotting. Including the fitted value can add one point to each axis.

All six PTD-RMC blocks enter the agentic likelihood, including errors.
Tau remains fixed at the saved joint estimate. These conditional slices
are not profiles that re-optimize the nuisance parameter at each point.

Coarse exploration supports these participant-specific plotting windows:
    Participant 8:  beta 0.1 to 5, sigma 0 to 0.03, logarithmic beta axis.
    Participant 46: beta 0 to 0.025, sigma 0.35 to 0.90.
    Participant 11: beta -0.00004 to 0.00030, sigma 0.85 to 1.65.

Participant 8 has a numerical beta plateau above about 2 at sigma zero.
The saved beta estimate marks that plateau, not an isolated maximum.
Increasing sigma to 0.00833 at the fitted beta reduces the log likelihood
by about 9.93. The other participants show curved likelihood ridges.
The saved joint log likelihoods are approximately -405.038239, -600.973186,
and -197.783238, respectively. Tau is always read at full saved precision.
For participant 46, the 25-point base grid improves on the saved fit by
about 0.671 at beta 0.01041667 and sigma 0.625, with tau unchanged.
The saved fit is therefore not the best value found on this slice.
Figure 3 marks the maximum found on each evaluated grid.

The serialized fits retain the legacy condition label PTD+RMC, whereas
the prepared observations and evaluation script use PTD-RMC. Generation
uses PTD-RMC and verifies the fitted likelihood before evaluating a grid.

Separate coarse, pilot, and plot pickle files reside in data/likelihood.
Rows index sigma and columns index beta. Each file retains the joint fit,
reference likelihood, block metadata, dependency fingerprint, and compute
time. Atomic row checkpoints support resuming interrupted calculations.
Changing source inputs or grid settings invalidates the associated cache.
Figure 3 only reads the plotting caches and does not compute likelihoods.
"""

import hashlib                                                                  # cache provenance
import json                                                                     # experimental block designs
import os                                                                       # optional execution settings
import pickle as pkl                                                            # likelihood cache serialization
import sys                                                                      # project utility imports
import time                                                                     # evaluation timing
from ast import literal_eval                                                    # observed action tuples
from pathlib import Path                                                        # filesystem paths
from types import SimpleNamespace                                               # agentic likelihood inputs
import numpy as np                                                              # likelihood grids
import pandas as pd                                                             # fitted parameters and observations

root                  = Path(__file__).resolve().parent.parent                  # project root directory
ddir                  = root / 'data' / 'experiment' / 'derivatives'            # observed data
edir                  = root / 'data' / 'evaluation'                            # saved joint estimates
tdir                  = root / 'data' / 'likelihood'                            # reusable likelihood surfaces
udir                  = root / 'code' / 'abm'                                   # agentic utilities
sys.path.insert(0, str(udir))                                                   # project import path
from abm_abm import abm_abm                                                     # agentic sequential log likelihood
from abm_mdp import abm_mdp                                                     # full block-specific task model
from abm_task import Task                                                       # observed-history reconstruction

eva                   = pd.read_pickle(edir / 'sub-all-eva.pkl')                # aggregate fits
mid                   = eva.M.index('A1')                                       # fitted agent index
cond                  = 'PTD-RMC'                                               # condition used by abm_evaluation.py
step                  = os.environ.get('ABM_LLH_STAGE', 'plot')                 # generation stage
if step not in ('coarse', 'pilot', 'plot'):                                     # supported stages
    raise ValueError('ABM_LLH_STAGE must be coarse, pilot, or plot.')           # invalid stage
base                  = 25 if step == 'plot' else 7                             # stage-specific resolution
res                   = int(os.environ.get('ABM_LLH_RES', base))                # requested grid resolution
if res < 3:                                                                     # require a two-dimensional surface
    raise ValueError('ABM_LLH_RES must be at least 3.')                         # invalid resolution
pids                  = (8, 46, 11)                                             # figure column order
pick                  = os.environ.get('ABM_LLH_PID')                           # optional participant selection
if pick is not None:                                                            # single-participant execution
    pids              = (int(pick),)                                            # selected participant
if any(pid not in (8, 46, 11) for pid in pids):                                 # supported figure participants
    raise ValueError('Select participant 8, 46, or 11.')                        # invalid participant
rngs                  = {8: ((0.0, 6.0), (0.0, 2.0)),                           # broad beta and sigma supports
        46: ((-0.003, 0.025), (0.0, 1.2)),                                      # small imbalance cost
        11: ((-0.0002, 0.001), (0.6, 1.7))}                                     # nearly zero imbalance cost
wins                  = {8: ((0.1, 5.0), (0.0, 0.03)),                          # plateau and boundary transition
        46: ((0.0, 0.025), (0.35, 0.9)),                                        # curved likelihood ridge
        11: ((-0.00004, 0.0003), (0.85, 1.65))}                                 # near-zero beta peak
tdir.mkdir(parents = True, exist_ok = True)                                     # cache output directory

for pid in pids:                                                                # participant-specific surfaces
    idx               = list(eva.p).index(pid)                                  # identifier lookup independent of row order
    fit               = np.asarray(eva.mls[idx][mid], dtype = float)            # beta, sigma, tau
    rng               = (rngs if step == 'coarse' else wins)[pid]               # axis windows
    bx                = np.unique(np.r_[np.linspace(*rng[0], res), fit[0]])     # include fitted beta
    if pid == 8 and step != 'coarse':                                           # resolve the low-beta transition
        vals          = np.geomspace(*rng[0], res)                              # positive beta support
        bx            = np.unique(np.r_[vals, fit[0]])                          # support including fitted beta
    sy                = np.unique(np.r_[np.linspace(*rng[1], res), fit[1]])     # include fitted sigma
    srcs              = [edir / 'sub-all-eva.pkl', ddir / f'sub-{pid:03}.csv']  # data inputs
    srcs += [ddir / 'design.json', ddir / 'stimuli' / 'stimuli.csv']            # designs
    srcs += sorted(udir.glob('*.py'))                                           # agentic implementation
    dig               = hashlib.sha256()                                        # source fingerprint
    for src in srcs:                                                            # all likelihood dependencies
        dig.update(src.name.encode())                                           # dependency identity
        dig.update(src.read_bytes())                                            # dependency content
    key               = dig.hexdigest()                                         # reproducible cache provenance
    file              = tdir / f'sub-{pid:03}-{step}-llh.pkl'                   # stage-specific cache
    old               = pd.read_pickle(file) if file.exists() else None         # saved evaluations
    good              = ( old is not None and old.key == key                    # unchanged inputs
                        and np.array_equal(old.b, bx)                           # unchanged beta support
                        and np.array_equal(old.s, sy)                           # unchanged sigma support
                        and np.array_equal(old.fit, fit))                       # unchanged fixed temperature
    if good and np.all(np.isfinite(old.llh)):                                   # completed matching cache
        print(f'sub-{pid:03}: reusing {file.name}', flush = True)               # cache reuse
        continue                                                                # avoid costly evaluation

    p                 = int(pid)                                                # participant identifier
    print(f'Evaluating sub-{p:03}: {cond}')                                     # participant progress
    src               = ddir / f'sub-{p:03}.csv'                                # participant data filename
    df                = pd.read_csv(                                            # load observations once
    src,                                                                        # prepared participant table
    sep               = '\t',                                                   # tab-separated fields
    converters        = {'a_t': literal_eval})                                  # restore task-response tuples
    df                = df.loc[df.C == cond].copy()                             # all condition blocks
    cfg               = json.loads((ddir / 'design.json').read_text())          # designs
    cfg               = cfg['participants'][str(p)][cond]                       # participant condition
    cat               = pd.read_csv(ddir / 'stimuli' / 'stimuli.csv')           # catalogue
    blks              = []                                                      # independent observed blocks
    info              = []                                                      # compact block provenance

    # block-specific designs and observed histories
    # -------------------------------------------------------------------------
    for (r, bid), grp in df.groupby(['R', 'B'], sort = True):                   # independent blocks
        b              = SimpleNamespace()                                      # block preparation namespace
        b.R            = int(r)                                                 # experimental run
        b.B            = int(bid)                                               # source block within the run
        b.df           = grp.sort_values('t').reset_index(drop = True)          # stages
        b.n            = len(b.df)                                              # observed horizon including errors
        run            = cfg[str(b.R)]                                          # run-specific experimental designs
        src            = {int(v['source_block']): k for k, v in run.items()}    # IDs
        par            = SimpleNamespace()                                      # full-model experimental settings
        par.P          = str(p)                                                 # participant identifier
        par.R          = str(b.R)                                               # run identifier
        par.B          = src[b.B]                                               # within-condition repetition identifier
        par.T1, par.T2 = cond.split('-')                                        # constituent tasks
        par.TP         = cond                                                   # condition identifier
        par.BLK        = run[par.B]                                             # block-specific design
        par.n          = b.n                                                    # block-specific horizon
        s1             = cat.loc[cat.task_type == par.T1]                       # Task 1 catalogue
        s2             = cat.loc[cat.task_type == par.T2]                       # Task 2 catalogue
        par.S1L        = s1.sort_values('stimulus_id').stimulus_name.tolist()   # IDs
        par.S2L        = s2.sort_values('stimulus_id').stimulus_name.tolist()   # IDs
        par.S1T        = par.BLK['S1T']                                         # Task 1 positive labels
        par.S2T        = par.BLK['S2T']                                         # Task 2 positive labels
        b.mdp          = abm_mdp(par)                                           # full task model built once
        par            = SimpleNamespace()                                      # observed-task input
        par.mode       = 'llh'                                                  # observed-history replay
        par.df         = b.df                                                   # ordered block observations
        par.mdp        = b.mdp                                                  # block-specific label functions
        task           = Task(par)                                              # fresh history with no previous task
        b.s            = task.s                                                 # observed states before each action
        blks.append(b)                                                          # preserve independent blocks
        row            = dict(R = b.R, B = b.B, n = b.n)                        # block provenance
        row['errors']  = int((b.df.r == 0).sum())                               # retained incorrect responses
        info.append(row)                                                        # compact results metadata
    par               = SimpleNamespace()                                       # participant-level agentic input
    par.mode          = 'llh'                                                   # observed-action evaluation
    par.agent         = 'A1'                                                    # fitted agent
    par.blocks        = blks                                                    # six independent observed histories
    par.θ             = fit.copy()                                              # fitted temperature remains fixed
    chk               = float(abm_abm(par))                                     # independently reproduce the fitted likelihood
    if not np.isclose(chk, eva.mll[idx, mid], atol = 1e-7, rtol = 0):           # fit check
        raise ValueError('Agentic likelihood differs from saved fit.')          # stale fit
    out               = old if good else SimpleNamespace()                      # cache namespace
    out.key           = key                                                     # dependency fingerprint
    out.pid           = pid                                                     # participant identifier
    out.cond          = cond                                                    # observed condition
    out.fit           = fit                                                     # joint beta, sigma, tau estimate
    out.ref           = chk                                                     # log likelihood at the joint estimate
    out.b             = bx                                                      # increasing beta axis
    out.s             = sy                                                      # increasing sigma axis
    out.res           = res                                                     # requested grid resolution
    out.step          = step                                                    # exploration or visualization
    out.info          = info                                                    # block identities, horizons, and error counts
    out.llh           = old.llh if good else np.full((len(sy), len(bx)), np.nan)# matrix
    out.secs          = old.secs if good else 0.0                               # cumulative evaluation time
    for r, sig in enumerate(sy):                                                # rows follow increasing sigma
        tic           = time.perf_counter()                                     # row evaluation timer
        for c, bet in enumerate(bx):                                            # columns follow increasing beta
            if np.isfinite(out.llh[r, c]):                                      # retain completed cells
                continue                                                        # resume interrupted work
            par.θ         = np.array((bet, sig, fit[2]))                        # fixed fitted temperature
            out.llh[r, c] = abm_abm(par)                                        # sum agentic block log likelihoods
        out.secs += time.perf_counter() - tic                                   # accumulated compute time
        tmp           = file.with_suffix('.tmp')                                # atomic row checkpoint
        with tmp.open('wb') as f:                                               # serialize numerical results and provenance
            pkl.dump(out, f)                                                    # reusable namespace
        tmp.replace(file)                                                       # publish complete checkpoint
        print(                                                                  # row progress
        f'sub-{pid:03} {step}: row {r + 1}/{len(sy)}, '                         # participant and stage
        f'{out.secs:.1f} s',                                                    # cumulative elapsed time
        flush = True)                                                           # cumulative elapsed time
    pos               = np.unravel_index(np.argmax(out.llh), out.llh.shape)     # grid maximum
    print(                                                                      # report grid maximum and fitted value
    f'sub-{pid:03}: grid maximum {out.llh[pos]:.9f}, '                          # best sampled likelihood
    f'beta = {bx[pos[1]]:.9g}, sigma = {sy[pos[0]]:.9g}, '                      # maximizing grid point
    f'joint fit = {out.ref:.9f}',                                               # reference likelihood
    flush = True)                                                               # reference likelihood
