"""
Construct reusable arrays for the reduced FCDT-MDP.
"""

import numpy as np                                                              # numerical arrays

def abm_mdp_chi(sta):                                                           # reduced FCDT-MDP structure
    """
    Prepare reduced states, transitions, and reward arrays.

    Input
        sta : SimpleNamespace() object with required attributes:
            .n   : positive integer horizon and absolute imbalance bound

    Output
        sta : input SimpleNamespace with the new attributes:
            .J   : integer array of decision and terminal stages
            .T   : integer array of regular previous-task values
            .B   : integer array of imbalance values
            .S   : integer array of previous-task and imbalance pairs
            .A   : integer array of abstract task actions
            .p0  : initial probability masses over reduced states
            .p   : unit transition masses for each state and action
            .k   : integer successor indices corresponding to those masses
            .r_b : negative absolute proposed imbalance
            .r_s : negative task-switch indicator
            .r_a : unit accuracy reward for each state and action

    Row zero of .S represents the initial state. Previous-task value zero
    encodes the absence of a previous task. Columns zero and one represent
    the mathematical components s^3 and s^4.

    Columns of .p, .k, .r_a, .r_b, and .r_s correspond to actions one and
    two. Each state-action pair has one successor with probability one.

    The parameter-independent reward components combine as
    .r_a + beta * .r_b + sigma * .r_s for the supplied cost parameters.
    """
    
    # index set, state space, and action set
    # -------------------------------------------------------------------------
    n                 = int(sta.n)                                              # horizon and imbalance bound
    n_b               = 2 * n + 1                                               # imbalance-set cardinality
    n_s               = 1 + 2 * n_b                                             # reduced state-space cardinality
    sta.J             = np.arange(n + 1, dtype = np.int64)                      # decision and terminal stages
    sta.T             = np.array((1, 2), dtype = np.int64)                      # task-state set T for s^3
    sta.B             = np.arange(-n, n + 1, dtype = np.int64)                  # imbalance set B for s^4
    sta.A             = np.array((1, 2), dtype = np.int64)                      # abstract task actions
    sta.S             = np.zeros((n_s, 2), dtype = np.int64)                    # initial and regular states (s^3, s^4)
    sta.S[1:, 0]      = np.repeat(sta.T, n_b)                                   # previous-task component s^3
    sta.S[1:, 1]      = np.tile(sta.B, 2)                                       # imbalance component s^4

    # initial state distribution
    # -------------------------------------------------------------------------
    sta.p0            = np.zeros(n_s, dtype = np.float64)                       # initial probability masses
    sta.p0[0]         = 1.0                                                     # unit mass at (s^3, s^4) = (empty, 0), encoded (0, 0)

    # transition distribution and reward arrays
    # -------------------------------------------------------------------------
    sta.p             = np.ones((n_s, 2), dtype = np.float64)                   # unit mass at each successor
    sta.k             = np.zeros((n_s, 2), dtype = np.int64)                    # successor row k with S[k] = (s_{t+1}^3, s_{t+1}^4) and p_χ(S[k] | s_t, a_t) = 1
    sta.r_b           = np.full((n_s, 2), np.nan, dtype = np.float64)           # parameter-independent r_b(s,a), contributing β * r_b(s,a) to ρ_χ(s,a)
    sta.r_s           = np.full((n_s, 2), np.nan, dtype = np.float64)           # parameter-independent r_s(s,a), contributing σ * r_s(s,a) to ρ_χ(s,a)
    sta.r_a           = np.ones((n_s, 2), dtype = np.float64)                   # parameter-independent r_a(s,a) = 1 for either correct task action
    s_t               = sta.S                                                   # current reduced states (s_t^3, s_t^4)
    s_tt              = np.full_like(s_t, np.nan, dtype = np.float64)           # successor states (s_{t+1}^3, s_{t+1}^4)
    abound            = np.abs(s_t[:, 1]) == n                                  # absorbing boundary |s_t^4| = n

    for a_t in sta.A:                                                           # abstract task actions
        a             = a_t - 1                                                 # action column
        s_tt[:, 0]    = a_t                                                     # successor task component s_{t+1}^3 = a_t
        s_tt[:, 1]    = s_t[:, 1] - (-1) ** int(a_t)                            # proposed s_{t+1}^4 = s_t^4 - (-1)^a_t
        sta.r_b[:, a] = -np.abs(s_tt[:, 1])                                     # r_b(s_t,a_t) = -|s_t^4 - (-1)^a_t|, including boundary states, without β
        sta.r_s[:, a] = -((s_t[:, 0] != 0) & (s_t[:, 0] != a_t)).astype(float)  # r_s(s_t,a_t) = -Iverson((s_t^3 = 1 and a_t = 2) or (s_t^3 = 2 and a_t = 1)), without σ
        s_tt[abound]  = s_t[abound]                                             # preserve both s^3 and s^4 at the boundary
        sta.k[:, a]   = 1 + (s_tt[:, 0] - 1) * n_b + s_tt[:, 1] + n             # k = 1 + (s_{t+1}^3 - 1)(2n + 1) + s_{t+1}^4 + n, so S[k] is the unique successor

    return sta                                                                  # reusable reduced MDP arrays
