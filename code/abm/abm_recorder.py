"""
Record simulated FCDT trials or accumulate the agentic log likelihood.
"""

import pandas as pd                                                             # pandas


class Recorder:

    """
    This is the definition file for the Recorder object.
    """

    # constructor
    # -------------------------------------------------------------------------
    def __init__(self, P, task, agent):                                         # initialize a simulation record or log likelihood
        """
        Initialize the recorder for the task's execution mode.

        Generative mode requires a trial-template DataFrame P.df
        with columns P, R, B, C, O, and t. The template is copied and each
        row's metadata is retained when generated outcomes are recorded.
        Likelihood mode initializes only the scalar log likelihood.
        """
        self.mode        = task.mode                                            # generative or log likelihood mode
        if self.mode == 'gen':                                                  # simulation output
            self.df      = P.df.copy()                                          # independent copy of the trial template
            self.D       = pd.DataFrame()                                       # simulated trial records
        else:                                                                   # conditional likelihood output
            self.llh     = 0.0                                                  # sum of observed-action log probabilities

    # recording function
    def rec(self, task, agent):                                                 # record a trial or its log probability
        """
        Record the simulated trial or add agent.lpa to self.llh.

        An impossible observed action contributes minus infinity, without
        clipping its probability. Log probabilities are supplied directly,
        avoiding probability underflow. Likelihood mode creates no trial table.
        """
        if self.mode == 'llh':                                                  # agentic conditional likelihood
            self.llh     = self.llh + agent.lpa                                 # accumulate stable agentic log probabilities
            return                                                              # no simulated-data record in likelihood mode

        j, k            = agent.a_t                                             # action task and response
        D_t             = self.df.iloc[[task.t]].copy()                         # current template row and its metadata

        # shared experimental-data fields
        D_t["t"]        = [task.t]                                              # zero-based trial index
        D_t["sA"]       = [task.mdp.S1L[task.s_t[0][1] - 1]]                    # Task 1 stimulus
        D_t["sB"]       = [task.mdp.S2L[task.s_t[1][1] - 1]]                    # Task 2 stimulus
        D_t["a1"]       = ["A" if j == 1 else "B"]                              # selected task
        D_t["a2"]       = [k]                                                   # binary response
        D_t["a_t"]      = [agent.a_t]                                           # action
        D_t["r"]        = [int(k == task.s_t[j - 1][0])]                        # response accuracy

        # simulation-specific fields
        D_t["s11_t"]    = [task.s_t[0][0]]                                      # Task 1 stimulus label
        D_t["s12_t"]    = [task.s_t[0][1]]                                      # Task 1 stimulus identifier
        D_t["s21_t"]    = [task.s_t[1][0]]                                      # Task 2 stimulus label
        D_t["s22_t"]    = [task.s_t[1][1]]                                      # Task 2 stimulus identifier
        D_t["s3_t"]     = [task.s_t[2]]                                         # previously selected task
        D_t["s4_t"]     = [task.s_t[3]]                                         # task imbalance
        D_t["varrho"]   = [task.ϱ]                                              # task reward varrho
        self.D           = pd.concat([self.D, D_t], ignore_index = True)        # dataframe concatenation
