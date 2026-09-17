"""
Compute an optimal task policy for the reduced FCDT-MDP.
"""

import numpy as np                                                              # numerical arrays

def abm_dp(mdp_chi):                                                            # finite-horizon optimal policy
    """
    Apply backward induction to the reduced FCDT-MDP with gamma = 1.

    Input
        mdp_chi : SimpleNamespace() object with required attributes:
            .n   : positive integer horizon
            .A   : ordered task-action array containing one and two
            .k   : integer successor indices, one column per task action
            .r_a : parameter-independent accuracy reward array
            .r_b : parameter-independent imbalance reward array
            .r_s : parameter-independent switching reward array
            .β   : finite signed imbalance cost parameter
            .σ   : nonnegative finite switching cost parameter

    Output
        mdp_chi : input SimpleNamespace with the new attributes:
            .ρ   : reward array for the supplied cost parameters
            .d   : uint8 decision rules representing the optimal policy pi*
            .v   : optimal values at stage zero for all reduced states
            .qχ  : action values indexed by stage, state row, and task action

    Output arrays are allocated and results are recomputed on each call.
    Two state-value buffers are reused across the backward stages, with
    the second buffer kept local to the function. The array .d is indexed by
    zero-based decision stage and state row. Entry d[t, i] stores the value
    d_t*(s_(i)) of the optimal decision function. These decision functions
    represent the deterministic optimal policy pi*: pi_t*(s_(i), a) equals one
    if a = d[t, i] and zero otherwise. A1 can select tasks directly from .d,
    while .qχ supplies stage-specific action values to its data model. Exact
    ties select the first action in .A, namely Task 1. The initial optimal value
    is .v at row zero. The full policy is stored, while only two successive 
    stages of state values are retained. Parameters satisfy beta finite and
    sigma finite and nonnegative, as specified by the estimation domain.
    """
    # numerical arrays
    # -------------------------------------------------------------------------
    n                 = int(mdp_chi.n)                                          # number of decision stages
    n_s               = mdp_chi.k.shape[0]                                      # reduced state-space cardinality
    mdp_chi.ρ         = np.full((n_s, 2), np.nan, dtype = np.float64)           # ρ_χ(s,a)
    mdp_chi.d         = np.zeros((n, n_s), dtype = np.uint8)                    # decision rules d_t*(s) representing π*
    mdp_chi.v         = np.full(n_s, np.nan, dtype = np.float64)                # first state-value buffer

    # reward evaluation
    # -------------------------------------------------------------------------
    mdp_chi.ρ[:]      = (mdp_chi.r_a + mdp_chi.β * mdp_chi.r_b                  # accuracy and imbalance rewards
                         + mdp_chi.σ * mdp_chi.r_s)                             # switching reward

    # terminal initialization
    # -------------------------------------------------------------------------
    mdp_chi.v[:]      = 0.0                                                     # v_n(s) = 0 for every state
    v_tt              = mdp_chi.v                                               # continuation values v_{t+1}
    v_t               = np.full(n_s, np.nan, dtype = np.float64)                # local buffer for current state values v_t
    mdp_chi.qχ        = np.empty((n, n_s, 2), dtype = np.float64)               # reduced action values q_{χ,t}*(s,a)
    q_t               = np.full((n_s, 2), np.nan, dtype = np.float64)           # current action values q_{χ,t}*(s,a)

    # backward induction
    # -------------------------------------------------------------------------
    for t in range(n - 1, -1, -1):                                              # stages n-1 through zero
        q_t[:]        = mdp_chi.ρ + v_tt[mdp_chi.k]                             # q_{χ,t}*(s,a) = ρ_χ(s,a) + v_{t+1}*(s_{t+1})
        mdp_chi.qχ[t] = q_t                                                     # retain the current reduced action values
        mdp_chi.d[t]  = mdp_chi.A[np.argmax(q_t, axis = 1)]                     # optimal decision function values d_t*(s)
        v_t[:]        = np.max(q_t, axis = 1)                                   # v_t*(s) = max_a q_t*(s,a)
        v_tt, v_t     = v_t, v_tt                                               # exchange buffers for the preceding stage

    # output
    # -------------------------------------------------------------------------
    mdp_chi.v         = v_tt                                                    # optimal state values v_0
    return mdp_chi                                                              # optimal policy π* represented by .d, and values
