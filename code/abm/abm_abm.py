"""
Run independent FCDT blocks in generative or agentic log likelihood mode.

Each block has its own horizon, initial state, task, agent, and recorder.
"""

from types import SimpleNamespace                                               # block-specific constructor inputs
import pandas as pd                                                             # combined simulated trial data
from abm_task import Task                                                       # task class
from abm_recorder import Recorder                                               # recorder class
from abm_construct import abm_construct                                         # agent construction


def abm_abm(P):                                                                 # multiple-block simulation or likelihood evaluation
    """
    Run independent blocks with shared agent parameters and no learning step.

    Input
        P : SimpleNamespace() object with required attributes:
            .agent  : agent identifier, e.g. 'A1'
            .blocks : block input namespaces (list), each supplying .mdp

    The optional .mode is 'gen' or 'llh' and defaults to 'gen'. Each block
    supplies .df, either ordered observations in likelihood mode or a
    trial template in generative mode. The template contains t, P, R, B,
    C, and O. Simulated data are collected in memory without file output.
    A1 constructs its reduced task model internally for each block.

    Both simulation and estimation supply the shared physical parameter
    vector (beta, sigma, tau) in P.θ for A1 and P.θ = None for A0.
    Agent parameters come from P, not individual blocks.
    This function forwards physical parameters to each agent. The agent
    computes action probabilities without a separate likelihood model.

    Output
        P : generative output namespace with shared .mode, .agent, .θ,
            .blocks, and combined simulated trials in .D (DataFrame)
        Likelihood mode returns the summed block log likelihood (float).

    Shared inputs are extracted before P is reused for block constructors.
    Generative mode assembles the output namespace after all blocks.
    Every block constructs a fresh task, agent, and recorder. Trials start
    at zero with the block's own initial state and horizon. No state passes
    between blocks. In likelihood mode, the task supplies observed states,
    the agent returns observed actions and their probabilities, and the
    recorder accumulates log probabilities. Only abm_abm sums across blocks.
    """
    # shared specifications
    # -------------------------------------------------------------------------
    mode                = getattr(P, 'mode', 'gen')                             # default generative mode
    mod                 = P.agent                                               # shared agent model
    θ                   = P.θ                                                   # shared physical parameters, or None for A0
    blocks              = P.blocks                                              # independent block inputs
    m                   = len(blocks)                                           # number of independent blocks
    if mode == 'llh':                                                           # reset total for every parameter evaluation
        llh             = 0.0                                                   # sum of block log likelihoods
    else:                                                                       # reset records for every simulation
        D               = pd.DataFrame()                                        # combined simulated blocks

    # independent blocks
    # -------------------------------------------------------------------------
    for b in range(m):                                                          # experimental block index
        P               = SimpleNamespace()                                     # block-specific constructor inputs
        P.mode          = mode                                                  # shared execution mode
        P.agent         = mod                                                   # shared agent model
        P.θ             = θ                                                     # shared physical parameters, or None for A0
        P.mdp           = blocks[b].mdp                                         # block-specific full task model
        P.df            = blocks[b].df                                          # observed data or simulation trial template

        # block initialization
        # ---------------------------------------------------------------------
        task            = Task(P)                                               # fresh block-specific task
        agent           = abm_construct(P.agent, P, task)                       # fresh block-specific agent
        record          = Recorder(P, task, agent)                              # fresh block-specific recorder
        task.t          = 0                                                     # first trial of this block
        task.get_s_0()                                                          # block-specific initial state

        # trials within the block
        # ---------------------------------------------------------------------
        while task.t < task.n:                                                  # observed or designed block horizon
            agent.t     = task.t                                                # current trial index
            agent.s_t   = task.s_t                                              # observed or generated state
            agent.action()                                                      # observed-action probability or generated action
            task.a_t    = agent.a_t                                             # realized action, not the intended decision
            task.get_ϱ()                                                        # observed or generated task reward
            record.rec(task, agent)                                             # log probability or generated trial record
            task.get_s_tt()                                                     # observed or generated successor state
            task.t      = task.t + 1                                            # next trial within this block

        # block output
        # ---------------------------------------------------------------------
        if mode == 'gen':                                                       # independently recorded simulated block
            D           = pd.concat((D, record.D), ignore_index = True)         # append simulated trials in block order
        else:                                                                   # likelihood contribution of the observed block
            llh         = llh + record.llh                                      # sum independent block log likelihoods

    # output
    # -------------------------------------------------------------------------
    if mode == 'gen':                                                           # simulation specifications and block collection
        P               = SimpleNamespace()                                     # combined simulation output
        P.mode          = mode                                                  # shared execution mode
        P.agent         = mod                                                   # shared agent model
        P.θ             = θ                                                     # shared physical parameters, or None for A0
        P.blocks        = blocks                                                # complete block collection
        P.D             = D                                                     # combined simulated trials
        return P                                                                # simulation output namespace
    return llh                                                                  # participant-level agentic conditional log likelihood
