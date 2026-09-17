"""
Implement A1 decisions and actions for one FCDT block.
"""

from types import SimpleNamespace                                               # reduced-model construction input
import numpy as np                                                              # numerical arrays and action sampling
from abm_mdp_chi import abm_mdp_chi                                             # reduced task-model construction
from scipy.special import logsumexp                                             # stable softmax normalization
from abm_dp import abm_dp                                                       # reduced-MDP backward induction


class Agent:                                                                    # optimal-value agent with softmax actions
    """
    Implement the agent and data models of A1.
    """

    # constructor
    # -------------------------------------------------------------------------
    def __init__(self, P, task):                                                # initialize A1 and its optimal policy
        """
        Initialize the agent for generative or log likelihood mode.

        Input
            P  : SimpleNamespace() object with required attributes:
                .θ : physical parameter vector (beta, sigma, tau)
            task : Task object with .mode, .A, and block horizon .n

        Both modes require .θ = (beta, sigma, tau). Likelihood mode also
        reads the ordered observations in task.df.

        Output
            self : agent with initialized attributes:
                .mode : generative or log likelihood mode
                .θ    : physical parameter vector (beta, sigma, tau)
                .τ    : strictly positive softmax temperature
                .D    : full decision space, identical to the action space
                .δ    : reduced optimal decision-function values
                .qχ   : reduced action values for every stage and state
                .q_t  : current correct/incorrect values for Tasks 1 and 2
                .lp   : log probabilities in the same four-action order
                .lpa  : log probability of the generated or observed action
                .t    : zero-based decision stage
                .s_t  : perceived full state
                .d_t  : action-lifted decision
                .a_t  : generated or observed action
                .pa   : conditional probability of the observed action

        The task reference is stored in .task, the reduced imbalance-set
        size in .nb, and likelihood observations in .df. The deterministic
        decision distribution requires no separate log likelihood term.
        """
        # agent specifications
        # ---------------------------------------------------------------------
        self.name   = 'Agent A1'                                                # agent denomination
        self.task   = task                                                      # full-MDP environment
        self.mode   = task.mode                                                 # generative or log likelihood mode
        self.D      = task.A                                                    # full decision space D = A
        self.θ      = P.θ                                                       # shared physical parameter vector
        self.τ      = self.θ[2]                                                 # response-distribution parameter
        if self.mode == 'llh':                                                  # observed block inputs
            self.df = task.df                                                   # observed block in the task's trial order

        # optimal policy construction
        # ---------------------------------------------------------------------
        mdp_chi     = SimpleNamespace()                                         # reduced task-model input
        mdp_chi.n   = task.n                                                    # block-specific horizon
        mdp_chi     = abm_mdp_chi(mdp_chi)                                      # reduced states, transitions, and rewards
        self.nb     = mdp_chi.B.size                                            # reduced imbalance-set cardinality
        mdp_chi.β   = self.θ[0]                                                 # imbalance cost beta
        mdp_chi.σ   = self.θ[1]                                                 # switching cost sigma
        mdp_chi     = abm_dp(mdp_chi)                                           # reduced optimal policy by backward induction
        self.qχ     = mdp_chi.qχ                                                # stage-specific reduced action values
        self.δ      = mdp_chi.d                                                 # optimal reduced decision functions delta_t*

        # dynamic variables
        # ---------------------------------------------------------------------
        self.t      = 0                                                         # initial decision stage
        self.s_t    = None                                                      # full state supplied before deciding
        self.d_t    = None                                                      # intended full action
        self.a_t    = None                                                      # generated or observed full action
        self.q_t    = None                                                      # full-action values in correct/incorrect order
        self.lp     = None                                                      # normalized full-action log probabilities
        self.lpa    = np.nan                                                    # stable selected-action log probability
        self.pa     = np.nan                                                    # observed-action probability

    # full-decision lifting
    # -------------------------------------------------------------------------
    def λ(self, s, a):                                                          # full-decision lifting lambda(s, a)
        """
        Return the correct full decision (a, k) for reduced task choice a.

        An unobserved stimulus label remains None in likelihood mode.
        The observed task has a known label. The unchosen label only
        permutes its correct/incorrect response values in the normalizer.
        """
        k           = s[a - 1][0]                                               # selected stimulus label, possibly unobserved
        if k is not None:                                                       # known binary stimulus label
            k       = int(k)                                                    # integer response component
        return (a, k)                                                           # full decision as a task-response tuple

    # decision function
    # -------------------------------------------------------------------------
    def delta(self):                                                            # deterministic full decision function delta_t
        """
        Set the optimal decision and full-action softmax log probabilities.

        Slots are Task 1 correct, Task 1 incorrect, Task 2 correct, and
        Task 2 incorrect. Incorrect responses lose one accuracy-reward unit
        but have the same reduced successor state. Unknown unchosen labels
        do not affect this normalizer. Center before dividing by tau.
        """
        s3          = self.s_t[2]                                               # previous-task component
        s4          = self.s_t[3]                                               # task imbalance before the current action
        if s3 is None:                                                          # initial reduced state
            k       = 0                                                         # row for the empty task history
        else:                                                                   # regular reduced state
            k       = 1 + (s3 - 1) * self.nb + s4 + self.task.n                 # row corresponding to chi(s_t)
        self.q_t    = np.repeat(self.qχ[self.t, k], 2)                          # full-action values in correct/incorrect order
        self.q_t[1::2] -= 1.0                                                   # one-unit penalty for incorrect responses
        q           = self.qχ[self.t, k] - np.max(self.qχ[self.t, k])           # center reduced task values before scaling
        lp          = np.repeat(q, 2)                                           # four centered action logits
        lp[1::2]   -= 1.0                                                       # incorrect-response accuracy penalty
        lp          = lp / self.τ                                               # four centered action logits
        self.lp     = lp - logsumexp(lp)                                        # normalized full-action log probabilities
        a           = int(self.δ[self.t, k])                                    # optimal reduced task decision at stage t
        self.d_t    = self.λ(self.s_t, a)                                       # action-lifted full decision delta_t(s_t)

    # decision realization
    # -------------------------------------------------------------------------
    def decide(self):                                                           # realize the deterministic conditional decision distribution
        """
        Realize d_t at the unit point mass defined by delta_t(s_t).
        """
        self.delta()                                                            # deterministic decision, identical in both modes

    # action realization
    # -------------------------------------------------------------------------
    def r_a(self):                                                              # sample the optimal-value softmax distribution
        """
        Sample a full action from the four softmax probabilities.
        """
        i           = np.random.choice(4, p = np.exp(self.lp))                  # selected full-action slot
        j           = int(i // 2 + 1)                                           # selected task
        k           = int(self.s_t[j - 1][0])                                   # correct response label for the selected task
        self.a_t    = (j, k if i % 2 == 0 else 1 - k)                           # generated task-response tuple
        self.p_a()                                                              # evaluate the generated action probability

    # action probability
    # -------------------------------------------------------------------------
    def p_a(self):                                                              # evaluate the observed-action softmax probability
        """
        Set the probability and stable log probability of self.a_t.
        """
        j, k        = self.a_t                                                  # observed or generated action components
        err         = int(k != self.s_t[j - 1][0])                              # incorrect-response indicator
        i           = 2 * (j - 1) + err                                         # selected full-action slot
        self.lpa    = float(self.lp[i])                                         # stable selected-action log probability
        self.pa     = float(np.exp(self.lpa))                                   # selected-action probability

    # ABM action
    # -------------------------------------------------------------------------
    def action(self):                                                           # generate an action or evaluate its conditional probability
        """
        Decide, then generate an action or evaluate the observed action.
        """
        self.decide()                                                           # action-lifted deterministic decision
        if self.mode == 'gen':                                                  # generative mode
            self.r_a()                                                          # full-action softmax realization
        else:                                                                   # log likelihood mode
            self.a_t = self.df.iloc[self.t].a_t                                 # prepared observed task-response tuple
            self.p_a()                                                          # stable observed-action log probability
