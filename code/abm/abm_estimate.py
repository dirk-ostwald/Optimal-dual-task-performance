"""
Evaluate A0 or jointly fit A1 across independent observed blocks.

A1 uses one Nelder-Mead run in physical parameter coordinates and the
agentic block log likelihoods.
"""

from types import SimpleNamespace                                               # parameter and block namespaces
import numpy as np                                                              # parameter coordinates
from scipy.optimize import minimize                                             # single-start joint optimization
from abm_define import abm_define                                               # model parameter space
from abm_abm import abm_abm                                                     # agentic block log likelihood


def abm_estimate(mle):                                                          # participant-level joint maximum likelihood estimation
    """
    Evaluate A0 or jointly estimate A1 parameters across observed blocks.

    Input
        mle : SimpleNamespace() object with required attributes:
            .M      : analysis model identifier, 'A0' or 'A1'
            .blocks : prepared blocks with .df, .mdp, and .n (list)

    Output
        mle : input SimpleNamespace with the new attributes:
            .mle : physical (beta, sigma, tau), or empty numpy.ndarray for A0
            .llh : fitted sequential conditional log likelihood (float)
            .k   : number of fitted parameters, zero or three
            .res : augmented scipy.optimize.OptimizeResult, or None for A0

    Each block is evaluated by abm_abm with a fresh task, agent, and recorder.
    The agent supplies observed-action probabilities, and the recorder
    accumulates the block log likelihood. Only parameters are shared.
    One abm_abm call evaluates and sums the complete block collection.
    A1 constructs its reduced task model within each agent initialization.

    A0 assigns probability one quarter to each observed action. A1 uses one
    Nelder-Mead run starting from physical parameters (0.01, 0.1, 0.3).
    All three parameters are optimized jointly. There are no restarts or
    separate boundary searches. Temperature is numerically bounded by
    exp(-15) and exp(15), with endpoints included. These safeguards are not
    restrictions of the theoretical parameter space.

    The .res output retains physical parameters in .x and .par, and the
    physical starting point in .ini. The coordinate tolerance applies to
    physical parameters. Its .guard flag identifies proximity to a
    temperature bound. No parameter transformations are used.
    Convergence does not establish a global maximum or a finite MLE.
    """
    # model parameter space
    # -------------------------------------------------------------------------
    mod               = SimpleNamespace()                                       # model definition input
    mod.M             = mle.M                                                   # requested model
    mod               = abm_define(mod)                                         # estimation domain
    mle.k             = mod.Θ.k                                                 # shared fitted parameter count

    # block-specific likelihood arguments
    # -------------------------------------------------------------------------
    P                 = SimpleNamespace()                                       # multi-block agentic input
    P.mode            = 'llh'                                                   # observed-data likelihood mode
    P.agent           = mle.M                                                   # requested agent
    P.blocks          = mle.blocks                                              # complete collection of observed blocks
    if mle.M == 'A0':                                                           # parameter-free agent evaluated through the same pipeline
        P.θ           = None                                                    # parameter-free uniform agent
        mle.mle       = np.empty(0)                                             # no fitted parameters
        mle.llh       = float(abm_abm(P))                                       # block sum
        mle.res       = None                                                    # no optimizer required
        return mle                                                              # evaluated agentic baseline

    # physical starting point and numerical bounds
    # -------------------------------------------------------------------------
    θ0                = np.array((0.01, 0.1, 0.3))                              # fixed physical starting point
    bnd               = list(mod.Θ.b)                                           # physical parameter bounds
    bnd[-1]           = (np.exp(-15), np.exp(15))                               # strictly positive numerical temperature interval

    # joint nonlinear optimization
    # -------------------------------------------------------------------------
    def nell(θ): P.θ = θ; return -abm_abm(P)                                    # negative log likelihood at the physical candidate
    res               = minimize(nell, θ0,                                      # joint maximum likelihood in physical coordinates
    method            = 'Nelder-Mead',                                          # derivative-free simplex search
    bounds            = bnd,                                                    # numerical physical domain
    options           = dict(                                                   # optimizer convergence settings
    maxiter           = 2500,                                                   # maximum simplex iterations
    xatol             = 1e-8,                                                   # physical-coordinate convergence tolerance
    fatol             = 1e-9,                                                   # likelihood convergence tolerance
    adaptive          = True))                                                  # dimension-dependent simplex coefficients
    res.ini           = θ0.copy()                                               # retain the physical starting point
    res.par           = res.x.copy()                                            # fitted physical beta, sigma, and temperature
    P.θ               = res.par                                                 # fitted parameters for final evaluation
    res.fun           = -float(abm_abm(P))                                      # negative likelihood at the final candidate
    lo                = np.exp(-14.9)                                           # lower temperature guard threshold
    hi                = np.exp(14.9)                                            # upper temperature guard threshold
    res.guard         = bool(res.x[2] < lo or res.x[2] > hi)                    # numerical-bound proximity

    # estimation results
    # -------------------------------------------------------------------------
    mle.mle           = res.par.copy()                                          # physical parameter estimates
    mle.llh           = -float(res.fun)                                         # participant-level fitted log likelihood
    mle.res           = res                                                     # single-run optimizer diagnostics
    return mle                                                                  # numerical fit and optimizer diagnostics
