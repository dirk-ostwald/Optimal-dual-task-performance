"""
Implement uniform A0 actions for one simulated or observed FCDT block.

Each of the four action tuples has probability one quarter. A0 has no free
parameters and requires no numerical optimization.
"""

import numpy as np                                                              # numerical initialization and action sampling


class Agent:                                                                    # uniformly random baseline agent
    """
    Generate uniform actions or evaluate their observed probabilities.
    """

    # constructor
    # -------------------------------------------------------------------------
    def __init__(self, P, task):                                                # initialize A0 for one block
        """
        Initialize the parameter-free uniform action model.

        Input
            P  : SimpleNamespace() object retained for the shared agent
                constructor interface, supplied with .θ = None and unused
            task : Task object with .mode and action space .A, and .df
                in likelihood mode, containing prepared action tuples a_t

        Output
            self : agent with initialized attributes:
                .name : agent denomination
                .task : task reference
                .mode : generative or log likelihood mode
                .D    : full task-response action space
                .t    : zero-based trial index
                .s_t  : state supplied by the task, unused by A0
                .a_t  : generated or observed action tuple (j, k)
                .pa   : observed-action probability
                .lpa  : observed-action log probability

        Likelihood mode also stores the task's ordered observations in .df.
        A0 neither uses subjective reward parameters nor estimates tau.
        """
        # agent specifications
        # ---------------------------------------------------------------------
        self.name   = 'Agent A0'                                                # agent denomination
        self.task   = task                                                      # full-MDP environment
        self.mode   = task.mode                                                 # generative or log likelihood mode
        self.D      = task.A                                                    # four full task-response actions
        if self.mode == 'llh':                                                  # observed block
            self.df = task.df                                                   # prepared observations in the task's trial order

        # dynamic variables
        # ---------------------------------------------------------------------
        self.t      = 0                                                         # initial trial index
        self.s_t    = None                                                      # state supplied before acting
        self.a_t    = None                                                      # generated or observed action tuple
        self.lpa    = np.nan                                                    # stable selected-action log probability
        self.pa     = np.nan                                                    # observed-action probability

    # action realization
    # -------------------------------------------------------------------------
    def r_a(self):                                                              # sample the uniform action distribution
        """
        Set self.a_t to a uniformly sampled full action tuple.
        """
        i        = np.random.choice(len(self.D))                                # uniform action index
        self.a_t = self.D[i]                                                    # corresponding task-response tuple

    # action probability
    # -------------------------------------------------------------------------
    def p_a(self):                                                              # evaluate the uniform observed-action probability
        """
        Set self.pa to one quarter for any observed full action.
        """
        self.lpa = -np.log(len(self.D))                                         # stable selected-action log probability
        self.pa  = 1 / len(self.D)                                              # equal probability for each of the four actions

    # ABM action
    # -------------------------------------------------------------------------
    def action(self):                                                           # generate an action or evaluate its probability
        """
        Sample an action or evaluate the prepared observation at trial t.
        """
        if self.mode == 'gen':                                                  # generative mode
            self.r_a()                                                          # uniform action realization
        else:                                                                   # log likelihood mode
            self.a_t = self.df.iloc[self.t].a_t                                 # prepared observed task-response tuple
            self.p_a()                                                          # observed-action probability without sampling
