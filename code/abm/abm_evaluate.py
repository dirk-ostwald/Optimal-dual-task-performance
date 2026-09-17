"""
Evaluate agent models across independent blocks of one participant's condition.

All blocks share one parameter vector. Observed histories reset at each block.
"""

import json                                                                     # block-specific experimental designs
from ast import literal_eval                                                    # deserialize prepared action tuples
from types import SimpleNamespace                                               # block and estimation namespaces
import numpy as np                                                              # numerical result and lookup arrays
import pandas as pd                                                             # experimental observations
from abm_estimate import abm_estimate                                           # participant-level parameter estimation
from abm_mdp import abm_mdp                                                     # block-specific full task model
from abm_task import Task                                                       # reconstruction of observed block histories


def abm_evaluate(eva):                                                          # participant-specific model evaluation
    """
    Fit the specified models jointly across a participant's condition blocks.

    Input
        eva : SimpleNamespace() object with required attributes:
            .d : data and metadata directory (pathlib.Path)
            .p : participant identifiers (sequence of integers)
            .i : index of the participant to evaluate (integer)
            .c : condition label, e.g. 'PTD-RMC'
            .M : analysis model identifiers (sequence of strings)

    Output
        eva : input SimpleNamespace with the new attributes:
            .mll  : model-wise fitted log likelihoods (numpy.ndarray)
            .mln  : model-wise total observed action counts (numpy.ndarray)
            .mlk  : model-wise fitted parameter counts (numpy.ndarray)
            .mls  : model-wise parameter estimates (list)
            .bic  : log likelihood minus half the BIC penalty (numpy.ndarray)
            .res  : model-wise estimation diagnostics (list)
            .info : block identities, horizons, and observation counts (list)

    Existing result fields are replaced with participant-specific results.
    Each (R, B) block has its own design, horizon, and observed history.
    Correct and incorrect responses are retained. A1 requires the answered
    stimulus label, but an unobserved other-task label is allowed.

    The estimator receives .blocks, a list of namespaces with .R, .B, .n,
    .df, .mdp, and .s, containing block identifiers, action counts, ordered
    observations, full task models, and observed states. The model identifier
    is supplied in .M. Participant and action counts remain local.
    The block collection is passed to abm_abm during estimation, using
    the same physical parameter vector and separate observed histories.

    One estimator call per model fits a shared parameter vector over blocks.
    A1 estimates are ordered as (beta, sigma, tau). A0 has no parameters.
    The .bic convention is log likelihood minus half the BIC penalty, so
    larger values indicate better fit after the complexity penalty.
    """
    # participant observations
    # -------------------------------------------------------------------------
    p              = int(eva.p[eva.i])                                          # participant identifier
    print(f'Evaluating sub-{p:03}: {eva.c}')                                    # participant progress
    file           = eva.d / f'sub-{p:03}.csv'                                  # participant data filename
    df             = pd.read_csv(                                               # load observations once
    file,                                                                       # prepared participant table
    sep            = '\t',                                                      # tab-separated fields
    converters     = {'a_t': literal_eval})                                     # restore task-response tuples
    df             = df.loc[df.C == eva.c].copy()                               # all condition blocks
    cfg            = json.loads((eva.d / 'design.json').read_text())            # designs
    cfg            = cfg['participants'][str(p)][eva.c]                         # participant condition
    cat            = pd.read_csv(eva.d / 'stimuli' / 'stimuli.csv')             # catalogue
    blocks         = []                                                         # independent observed blocks
    eva.info       = []                                                         # compact block provenance

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
        P              = SimpleNamespace()                                      # full-model experimental settings
        P.P            = str(p)                                                 # participant identifier
        P.R            = str(b.R)                                               # run identifier
        P.B            = src[b.B]                                               # within-condition repetition identifier
        P.T1, P.T2     = eva.c.split('-')                                       # constituent tasks
        P.TP           = eva.c                                                  # condition identifier
        P.BLK          = run[P.B]                                               # block-specific design
        P.n            = b.n                                                    # block-specific horizon
        s1             = cat.loc[cat.task_type == P.T1]                         # Task 1 catalogue
        s2             = cat.loc[cat.task_type == P.T2]                         # Task 2 catalogue
        P.S1L          = s1.sort_values('stimulus_id').stimulus_name.tolist()   # IDs
        P.S2L          = s2.sort_values('stimulus_id').stimulus_name.tolist()   # IDs
        P.S1T          = P.BLK['S1T']                                           # Task 1 positive labels
        P.S2T          = P.BLK['S2T']                                           # Task 2 positive labels
        b.mdp          = abm_mdp(P)                                             # full task model built once
        P              = SimpleNamespace()                                      # observed-task input
        P.mode         = 'llh'                                                  # observed-history replay
        P.df           = b.df                                                   # ordered block observations
        P.mdp          = b.mdp                                                  # block-specific label functions
        task           = Task(P)                                                # fresh history with no previous task
        b.s            = task.s                                                 # observed states before each action
        blocks.append(b)                                                        # preserve independent blocks
        row            = dict(R = b.R, B = b.B, n = b.n)                        # block provenance
        row['errors']  = int((b.df.r == 0).sum())                               # retained incorrect responses
        eva.info.append(row)                                                    # compact results metadata

    # participant-level likelihood inputs
    # -------------------------------------------------------------------------
    n              = len(df)                                                    # total observed actions

    # analysis model iterations
    # -------------------------------------------------------------------------
    nm             = len(eva.M)                                                 # number of requested models
    eva.mll        = np.full(nm, np.nan)                                        # fitted log likelihoods
    eva.mln        = np.full(nm, np.nan)                                        # total observed action counts
    eva.mlk        = np.full(nm, np.nan)                                        # fitted parameter counts
    eva.bic        = np.full(nm, np.nan)                                        # penalized fit scores
    eva.mls        = [None for _ in range(nm)]                                  # physical parameter estimates
    eva.res        = [None for _ in range(nm)]                                  # estimation diagnostics
    for i, mod in enumerate(eva.M):                                             # one joint fit per model
        mle        = SimpleNamespace()                                          # model-specific estimation inputs
        mle.blocks = blocks                                                     # independent observed blocks
        mle.M      = mod                                                        # model identifier
        mle        = abm_estimate(mle)                                          # joint participant-level estimation
        eva.mll[i] = mle.llh                                                    # sum of block log likelihoods
        eva.mln[i] = n                                                          # all observed actions across blocks
        eva.mlk[i] = mle.k                                                      # number of shared fitted parameters
        eva.bic[i] = mle.llh - 0.5 * mle.k * np.log(n)                          # comparison score
        eva.mls[i] = mle.mle                                                    # physical parameter estimates
        eva.res[i] = mle.res                                                    # model-specific estimation diagnostics
    
    return eva                                                                  # participant-level results and compact block metadata
