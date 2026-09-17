"""
Define the FCDT task for one simulated or observed block.

The task reward varrho records completed and correct responses for each task.
The agent's subjective reward rho and its cost parameters are separate.
"""


class Task:                                                                     # single-block task environment
    """
    Supply full states and the experimenter's four-component task reward.
    """

    # constructor
    # -------------------------------------------------------------------------
    def __init__(self, P):                                                      # initialize the task in generative or likelihood mode
        """
        Initialize one block from its full MDP and optional observations.

        Input
            P : SimpleNamespace() object with required attributes:
                .mdp  : full task model (abm_mdp)

        The optional .mode is 'gen' or 'llh' and defaults to 'gen'. In 'llh'
        mode, .df is a pandas.DataFrame containing one complete block with
        columns t, sA, sB, a_t, and r. Trial indices run from
        zero to .mdp.n - 1. Stimulus names and labels use the block-specific
        .mdp.S1L, .mdp.S2L, .mdp.κ1, and .mdp.κ2 definitions. Task choices
        and responses are stored as tuples a_t = (j, k), with j in (1, 2)
        and k in (0, 1). Correctness r is binary.

        Output
            self : task object with initialized attributes:
                .mode : execution mode
                .mdp  : full task model
                .A    : task-response action space
                .n    : block horizon
                .t    : trial counter
                .s_t  : current full state
                .a_t  : current action tuple (j, k)
                .ϱ    : task reward tuple in the order (N1, C1, N2, C2)

        In likelihood mode, .df holds observations in trial order and .s
        holds their full states. Neither states nor actions are sampled.
        Previous-task and imbalance components use observed task choices,
        irrespective of correctness. An unobserved stimulus is represented
        by (None, None), without imputation or removal of the action. A0's
        likelihood does not depend on this component. A1 evaluation requires
        complete stimulus observations. No agent reward parameters are needed.
        """
        # task specifications
        # ---------------------------------------------------------------------
        self.mode    = getattr(P, 'mode', 'gen')                                # generative or log likelihood mode
        if self.mode not in ('gen', 'llh'):                                     # supported execution modes
            raise ValueError("Mode must be 'gen' or 'llh'.")                    # invalid mode
        self.mdp     = P.mdp                                                    # full state and transition model
        self.A       = self.mdp.A                                               # full task-response action space
        self.n       = self.mdp.n                                               # block-specific horizon

        # observed block
        # ---------------------------------------------------------------------
        if self.mode == 'llh':                                                  # observation-conditioned task
            self.df  = P.df.sort_values('t').reset_index(drop = True)           # trial-ordered observations

            # stimulus labels and observed history
            mdp      = self.mdp                                                 # block-specific stimulus label functions
            s1       = dict(zip(mdp.S1L, zip(mdp.κ1, mdp.I1)))                  # labeled Task 1 stimuli
            s2       = dict(zip(mdp.S2L, zip(mdp.κ2, mdp.I2)))                  # labeled Task 2 stimuli
            self.s   = []                                                       # full states preceding the observed actions
            j        = None                                                     # no task choice before the first trial
            b        = 0                                                        # zero initial task imbalance
            for row in self.df.itertuples():                                    # retain correct and incorrect responses
                l1   = s1.get(row.sA, (None, None))                             # observed Task 1 stimulus or unknown
                l2   = s2.get(row.sB, (None, None))                             # observed Task 2 stimulus or unknown
                s    = (l1, l2, j, b)                                           # observed stimuli and action history
                self.s.append(s)                                                # state before the current observed action
                j    = row.a_t[0]                                               # current observed task choice
                b    = b + (1 if j == 1 else -1)                                # imbalance after the observed choice

        # dynamic variables
        # ---------------------------------------------------------------------
        self.t       = 0                                                        # trial counter
        self.s_t     = []                                                       # state supplied before acting
        self.a_t     = None                                                     # action supplied by the agent
        self.ϱ       = (0, 0, 0, 0)                                             # completed and correct responses for each task

    # initial state
    # -------------------------------------------------------------------------
    def get_s_0(self):                                                          # sample or read the initial full state
        """
        Set self.s_t to the sampled or observed initial state.
        """
        if self.mode == 'gen':                                                  # generative mode
            self.s_t = self.mdp.r_p0()                                          # sample the initial state distribution
        else:                                                                   # log likelihood mode
            self.s_t = self.s[0]                                                # first observed stimulus pair and empty history

    # task reward
    # -------------------------------------------------------------------------
    def get_ϱ(self):                                                            # evaluate or read the experimenter's reward
        """
        Set self.ϱ to varrho in the order (N1, C1, N2, C2).
        """
        if self.mode == 'gen':                                                  # generative mode
            j, k     = self.a_t                                                 # selected task and binary response
            r        = int(k == self.s_t[j - 1][0])                             # correct response to the current stimulus
        else:                                                                   # log likelihood mode
            row      = self.df.iloc[self.t]                                     # current observed trial
            j        = row.a_t[0]                                               # observed task choice
            r        = int(row.r)                                               # recorded response correctness
        self.ϱ       = ( int(j == 1), int(j == 1) * r,                          # Task 1 completed and correct response
                         int(j == 2), int(j == 2) * r)                          # Task 2 completed and correct response

    # successor state
    # -------------------------------------------------------------------------
    def get_s_tt(self):                                                         # sample or read the successor full state
        """
        Update self.s_t before the caller increments the trial counter.
        """
        if self.mode == 'gen':                                                  # generative mode
            self.s_t = self.mdp.r_p(self.s_t, self.a_t)                         # sample the successor state distribution
        elif self.t + 1 < self.n:                                               # observed successor exists within this block
            self.s_t = self.s[self.t + 1]                                       # next observed stimulus pair and choice history
