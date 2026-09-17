"""
Define FCDT states and transitions with full action tuples (j, k).
"""

import numpy as np                                                              # numpy

class abm_mdp:

    """
    This class defines the free-concurrent dual-task MDP.
    """

    # MDP instantiation
    # -------------------------------------------------------------------------
    def __init__(self, P):

        """
        This function is the instantiation operation of the MDP class.

        Input
            P       : experimental design structure

        Output
            Initialized MDP object

        """
        
        # attributes
        # ---------------------------------------------------------------------
        # experimental identity and provenance
        self.P      = P.P                                                       # participant ID
        self.run    = P.R                                                       # run number
        self.blk    = P.B                                                       # task repetition block
        self.T1     = P.T1                                                      # task 1
        self.T2     = P.T2                                                      # task 2
        self.TP     = P.TP                                                      # dual-task condition
        self.SB     = P.BLK['source_block']                                     # source local block number

        # concrete stimulus specifications
        self.S1L    = P.S1L                                                     # task 1 stimulus labels
        self.S2L    = P.S2L                                                     # task 2 stimulus labels
        self.S1T    = P.S1T                                                     # task 1 target labels
        self.S2T    = P.S2T                                                     # task 2 target labels
        self.n1     = len(self.S1L)                                             # number of task 1 stimuli
        self.n2     = len(self.S2L)                                             # number of task 2 stimuli
        self.I1     = np.arange(1, self.n1 + 1)                                 # task 1 stimulus identifier set
        self.I2     = np.arange(1, self.n2 + 1)                                 # task 2 stimulus identifier set
        self.κ1     = np.isin(self.S1L, self.S1T).astype(int)                   # task 1 stimulus label function
        self.κ2     = np.isin(self.S2L, self.S2T).astype(int)                   # task 2 stimulus label function
        self.I10    = self.I1[self.κ1 == 0]                                     # task 1 label 0 identifier subset
        self.I11    = self.I1[self.κ1 == 1]                                     # task 1 label 1 identifier subset
        self.I20    = self.I2[self.κ2 == 0]                                     # task 2 label 0 identifier subset
        self.I21    = self.I2[self.κ2 == 1]                                     # task 2 label 1 identifier subset
        self.n10    = len(self.I10)                                             # task 1 label 0 subset cardinality
        self.n11    = len(self.I11)                                             # task 1 label 1 subset cardinality
        self.n20    = len(self.I20)                                             # task 2 label 0 subset cardinality
        self.n21    = len(self.I21)                                             # task 2 label 1 subset cardinality

        # action specifications       
        self.na     = 4                                                         # number of actions
        self.A      = ((1, 0), (1, 1), (2, 0), (2, 1))                          # full action space of task-response tuples

        # state-index specifications
        self.n      = P.n                                                       # maximum number of trials
        self.nb     = 2 * self.n + 1                                            # number of task imbalance values
        self.S0     = [((int(self.κ1[i1 - 1]), int(i1)),                        # task 1 labeled stimulus
                        (int(self.κ2[i2 - 1]), int(i2)),                        # task 2 labeled stimulus
                        None, 0)                                                # initial task and imbalance values
                        for i1 in self.I1 for i2 in self.I2]                    # initial state layer
        self.n0     = len(self.S0)                                              # number of initial states
        self.nsp    = self.n1 * self.n2 * 2 * self.nb                           # number of regular states
        self.ns     = self.n0 + self.nsp                                        # number of states

    # methods
    # -------------------------------------------------------------------------
    # initial state distribution methods
    def p0(self, s_0):

        """
        This function evaluates the initial state distribution p₀(s₀).

        Input
            self    : MDP object
            s_0     : initial state

        Output
            Initial state probability mass

        """
        # initial state components
        s1_0, s2_0, s3_0, s4_0  = s_0                                           # initial state components
        l1_0, i1_0              = s1_0                                          # task 1 label and identifier
        l2_0, i2_0              = s2_0                                          # task 2 label and identifier

        # label-specific identifier subsets and cardinalities
        I1l = self.I10 if l1_0 == 0 else self.I11                               # I_{1,s₀^{1,1}}
        I2l = self.I20 if l2_0 == 0 else self.I21                               # I_{2,s₀^{2,1}}
        n1l = self.n10 if l1_0 == 0 else self.n11                               # n_{1,s₀^{1,1}}
        n2l = self.n20 if l2_0 == 0 else self.n21                               # n_{2,s₀^{2,1}}

        # initial state probability mass terms
        T1  = (1 / 2) * int(i1_0 in I1l) / n1l                                  # ½ ⟦s₀^{1,2} ∈ I_{1,s₀^{1,1}}⟧ / n_{1,s₀^{1,1}}
        T2  = (1 / 2) * int(i2_0 in I2l) / n2l                                  # ½ ⟦s₀^{2,2} ∈ I_{2,s₀^{2,1}}⟧ / n_{2,s₀^{2,1}}
        T3  = int(s3_0 is None)                                                 # ⟦s₀³ = ∅⟧
        T4  = int(s4_0 == 0)                                                    # ⟦s₀⁴ = 0⟧
        return T1 * T2 * T3 * T4                                                # initial state probability mass

    def p0_pmf(self):

        """
        This function evaluates the initial state probability mass function p₀.

        Input
            self    : MDP object

        Output
            Initial state probability masses

        """ 
        return np.array([self.p0(s_0) for s_0 in self.S0])                      # initial state probability masses
 
    def r_p0(self):

        """
        This function samples the initial state distribution p₀(s).

        Input
            self    : MDP object

        Output
            Initial state realization

        """
        i = np.random.choice(self.n0, p = self.p0_pmf())                        # initial state index realization
        return self.S0[i]                                                       # initial state realization

    # state transition methods
    def p(self, s_tt, s_t, a_t):

        """
        Evaluate the state transition probability p(s_tt|s_t,a_t).

        Input
            self    : MDP object
            s_tt    : successor state
            s_t     : current state
            a_t     : current action

        Output
            State transition probability mass

        """
        # state components
        s1_t, s2_t, s3_t, s4_t      = s_t                                       # current state components
        s1_tt, s2_tt, s3_tt, s4_tt  = s_tt                                      # successor state components

        # absorbing boundary transition
        if abs(s4_t) == self.n:                                                 # |s⁴| = n
            return int(s_tt == s_t)                                             # ⟦s̃ = s⟧
        if abs(s4_t) > self.n:                                                  # |s⁴| > n
            return 0.0                                                          # state outside S

        # selected and unselected task components
        j, _            = a_t                                                   # selected task j
        Δa              = 1 if j == 1 else -1                                   # task imbalance increment Δ(a)
        sj_t            = s1_t if j == 1 else s2_t                              # selected current task state sʲ
        sj_tt           = s1_tt if j == 1 else s2_tt                            # selected successor task state s̃ʲ
        su_t            = s2_t if j == 1 else s1_t                              # unselected current task state s³⁻ʲ
        su_tt           = s2_tt if j == 1 else s1_tt                            # unselected successor task state s̃³⁻ʲ
        lj_tt, ij_tt    = sj_tt                                                 # successor label and identifier
        _, ij_t         = sj_t                                                  # current identifier

        # successor label-specific identifier subset and cardinality
        if j == 1:                                                              # Task 1 selected
            Ijl = self.I10 if lj_tt == 0 else self.I11                          # I_{1,s̃^{1,1}}
            njl = self.n10 if lj_tt == 0 else self.n11                          # n_{1,s̃^{1,1}}
        else:                                                                   # Task 2 selected
            Ijl = self.I20 if lj_tt == 0 else self.I21                          # I_{2,s̃^{2,1}}
            njl = self.n20 if lj_tt == 0 else self.n21                          # n_{2,s̃^{2,1}}

        # state transition probability mass terms
        T1  = 1 / 2                                                             # equal successor label probability
        T2  = int(ij_tt != ij_t)                                                # ⟦s̃^{j,2} ≠ s^{j,2}⟧
        T3  = int(ij_tt in Ijl)                                                 # ⟦s̃^{j,2} ∈ I_{j,s̃^{j,1}}⟧
        T4  = int(su_tt == su_t)                                                # ⟦s̃³⁻ʲ = s³⁻ʲ⟧
        T5  = int(s3_tt == j)                                                   # ⟦s̃³ = j⟧
        T6  = int(s4_tt == s4_t + Δa)                                           # ⟦s̃⁴ = s⁴ + Δ(a)⟧
        D   = njl - int(ij_t in Ijl)                                            # n_{j,s̃^{j,1}} - ⟦s^{j,2} ∈ I_{j,s̃^{j,1}}⟧
        return T1 * T2 * T3 * T4 * T5 * T6 / D                                  # state transition probability mass

    def p_pmf(self, s_t, a_t):

        """
        This function evaluates the sparse state transition probability mass
        function p(.|s_t,a_t).

        Input
            self    : MDP object
            s_t     : current state
            a_t     : current action

        Output
            p       : successor state probability masses
            S       : nonzero successor state support

        """
        # state and action components
        s1_t, s2_t, s3_t, s4_t  = s_t                                           # current state components
        j, _                    = a_t                                           # selected task j
        Δa                      = 1 if j == 1 else -1                           # task imbalance increment

        # nonzero successor state support
        if abs(s4_t) == self.n:                                                 # absorbing boundary state
            S = [s_t]                                                           # singleton successor support
        elif j == 1:                                                            # Task 1 selected
            S = [((int(self.κ1[i - 1]), int(i)),                                # successor Task 1 stimulus
                  s2_t, j, s4_t + Δa)                                           # unchanged Task 2, task, and balance
                  for i in self.I1 if i != s1_t[1]]                             # direct-repetition exclusion
        else:                                                                   # Task 2 selected
            S = [(s1_t,                                                         # unchanged Task 1 stimulus
                  (int(self.κ2[i - 1]), int(i)),                                # successor Task 2 stimulus
                  j, s4_t + Δa)                                                 # task and balance
                  for i in self.I2 if i != s2_t[1]]                             # direct-repetition exclusion

        p   = np.array([self.p(s_tt, s_t, a_t) for s_tt in S])                  # nonzero successor probability masses
        return p, S                                                             # sparse state transition distribution

    def r_p(self, s_t, a_t):

        """
        This function samples the state transition distribution p(.|s_t,a_t).

        Input
            self    : MDP object
            s_t     : current state
            a_t     : current action

        Output
            Successor state realization

        """
        p, S = self.p_pmf(s_t, a_t)                                             # sparse state transition distribution
        i       = np.random.choice(len(S), p = p)                               # successor state index realization
        return S[i]                                                             # successor state realization
    
    # reward function methods
    def ρ_β_σ(self, s_t, a_t, β, σ):

        """
        Evaluate the agent's subjective reward ρ_{β,σ}(s_t,a_t).

        Input
            self    : MDP object
            s_t     : current state
            a_t     : current action
            β       : task balance cost parameter
            σ       : task switch cost parameter

        Output
            Reward

        """
        # state components
        s1_t, s2_t, s3_t, s4_t  = s_t                                           # current state components
        l1_t, _                 = s1_t                                          # task 1 stimulus label
        l2_t, _                 = s2_t                                          # task 2 stimulus label
        j, ra_t                 = a_t                                           # current action task and response

        # task accuracy component
        r   = int((l1_t == 0 and (j,ra_t) == (1,0)) or                          # ⟦s_t^{1,1} = 0 ∧ a_t = a^{1,0} ∨
                  (l1_t == 1 and (j,ra_t) == (1,1)) or                          #  s_t^{1,1} = 1 ∧ a_t = a^{1,1} ∨
                  (l2_t == 0 and (j,ra_t) == (2,0)) or                          #  s_t^{2,1} = 0 ∧ a_t = a^{2,0} ∨
                  (l2_t == 1 and (j,ra_t) == (2,1)))                            #  s_t^{2,1} = 1 ∧ a_t = a^{2,1}⟧
        
        # task balance component
        Δa_t = 1 if j == 1 else -1                                              # Δ(a_t) := +1 (a_t ∈ A¹), -1 (a_t ∈ A²)
        r_β  = -β * abs(s4_t + Δa_t)                                            # r_β(s_t,a_t) := -β·|s_t⁴ + Δ(a_t)|

        # task switch cost comoponent
        r_σ = -σ * int((s3_t == 1 and j == 2) or (s3_t == 2 and j == 1))        # r_σ(s_t,a_t) := -σ·⟦(s_t³ = 1 ∧ a_t ∈ A²) ∨ (s_t³ = 2 ∧ a_t ∈ A¹)⟧
        return r + r_β + r_σ                                                    # parameterized reward
