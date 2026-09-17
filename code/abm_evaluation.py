"""
Evaluate A0 and A1 across all six PTD-RMC blocks of each participant.

A0 is evaluated without optimization. A1 parameters (beta, sigma, tau) are
fitted jointly using single-start Nelder-Mead. All observed responses are
retained. Participants are evaluated in parallel using Windows spawn.
Results and descriptive statistics cover all selected blocks.
"""

# general utility imports
# -----------------------------------------------------------------------------
import sys                                                                      # system tools
from multiprocessing import cpu_count, get_context                              # Windows process pool
import pickle as pkl                                                            # evaluation output serialization
from pathlib import Path                                                        # path management
from types import SimpleNamespace                                               # shared evaluation state
import numpy as np                                                              # participant and result arrays

# directory management
# -----------------------------------------------------------------------------
wdir    = Path(__file__).resolve().parent                                       # script directory
udir    = wdir / 'abm'                                                          # project utilities directory
rdir    = wdir.parent                                                           # project root directory
ddir    = rdir / 'data' / 'experiment' / 'derivatives'                          # experimental data and metadata
tdir    = rdir / 'data' / 'evaluation'                                          # evaluation output directory
sys.path.append(str(udir))                                                      # add project utilities to the module search path

# project utility imports
# -----------------------------------------------------------------------------
from abm_evaluate import abm_evaluate                                           # participant fitting through the agentic likelihood
from abm_describe import abm_describe                                           # condition-specific descriptive statistics

# parameters
# -----------------------------------------------------------------------------
p       = np.arange(1, 50, dtype = int)                                         # all 49 participant identifiers
mods    = ('A0', 'A1')                                                          # analysis model set

# model evaluation specifications
# -----------------------------------------------------------------------------
eva     = SimpleNamespace()                                                     # evaluation structure
eva.m   = 'exp'                                                                 # experimental data mode
eva.d   = ddir                                                                  # participant data and design metadata directory
eva.t   = tdir                                                                  # participant and aggregate evaluation outputs
eva.M   = mods                                                                  # analysis model set
eva.c   = 'PTD-RMC'                                                             # condition selected within each participant file
eva.p   = p                                                                     # participant identifiers
eva.n   = len(eva.p)                                                            # number of participants
eva.i   = range(eva.n)                                                          # participant iteration indices

# participant evaluation
# -----------------------------------------------------------------------------
if __name__ == '__main__':                                                      # execute fitting only when run as a script

    eva.t.mkdir(parents = True, exist_ok = True)                                # preserve existing evaluation files
    eva.mll             = np.full((eva.n, len(eva.M)), np.nan)                  # maximized log likelihoods
    eva.mln             = np.full((eva.n, len(eva.M)), np.nan)                  # numbers of valid observed actions
    eva.mlk             = np.full((eva.n, len(eva.M)), np.nan)                  # numbers of fitted parameters
    eva.mls             = [None for i in eva.i]                                 # participant-specific parameter estimates
    eva.res             = [None for i in eva.i]                                 # model-wise optimizer diagnostics

    jobs                = []                                                    # independent participant inputs
    for i in eva.i:                                                             # participant iterations
        evp             = SimpleNamespace()                                     # participant evaluation namespace
        evp.d           = eva.d                                                 # participant data directory
        evp.p           = eva.p                                                 # participant identifiers
        evp.c           = eva.c                                                 # condition to evaluate
        evp.M           = eva.M                                                 # requested agent models
        evp.i           = i                                                     # participant index supplied inside the namespace
        jobs.append(evp)                                                        # submit only required participant fields

    print(f'Number of CPUs: {cpu_count()}')                                     # available logical CPUs
    ctx                 = get_context('spawn')                                  # Windows multiprocessing context
    nproc               = max(1, cpu_count() // 2)                              # half the CPUs to limit resource use
    print(f'Using {nproc} parallel processes')                                  # configured worker count
    with ctx.Pool(processes = nproc, maxtasksperchild = 10) as pool:            # recycle workers after ten participants
        for evp in pool.imap_unordered(abm_evaluate, jobs):                     # collect participants as they complete
            i           = evp.i                                                 # restore the participant's aggregate row
            eva.mll[i]  = evp.mll                                               # collect model-wise maximized log likelihoods
            eva.mln[i]  = evp.mln                                               # collect model-wise observation counts
            eva.mlk[i]  = evp.mlk                                               # collect model-wise parameter counts
            eva.mls[i]  = evp.mls                                               # collect model-wise parameter estimates
            eva.res[i]  = evp.res                                               # aggregate participant optimizer diagnostics
            file        = eva.t / f'sub-{int(eva.p[i]):03}-eva.pkl'             # participant evaluation filename
            with file.open('wb') as f:                                          # participant evaluation output
                pkl.dump(evp, f)                                                # serialize participant results

            for j, mod in enumerate(eva.M):                                     # print model-wise parameter estimates and fit
                print(                                                          # participant model result
                f'sub-{int(eva.p[i]):03} {mod}: '                               # participant and model identifiers
                f'parameters = {evp.mls[j]}, log likelihood = {evp.mll[j]:.9f}',# beta, sigma, tau for A1, empty vector for A0
                flush = True)                                                   # display completed results immediately

    print('All parallel evaluations completed successfully.')                   # all participant results collected

    # descriptive statistics and result aggregation
    # -------------------------------------------------------------------------
    sta                 = SimpleNamespace()                                     # descriptive statistics namespace
    sta.d               = eva.d                                                 # participant data directory
    sta.p               = eva.p                                                 # evaluated participants
    sta.c               = eva.c                                                 # evaluated condition
    sta                 = abm_describe(sta)                                     # evaluate condition-specific descriptive statistics
    eva.sta             = sta                                                   # attach descriptive statistics

    # data saving
    # -------------------------------------------------------------------------
    file                = eva.t / 'sub-all-eva.pkl'                             # aggregate evaluation filename
    with file.open('wb') as f:                                                  # aggregate evaluation output
        pkl.dump(eva, f)                                                        # serialize evaluation settings and results
