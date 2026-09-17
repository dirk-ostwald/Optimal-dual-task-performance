"""
Evaluate FCDT descriptive statistics by condition and optional block.
"""

import numpy as np                                                              # numerical operations
import pandas as pd                                                             # data frame management


def abm_describe(sta):                                                          # condition-specific descriptive statistics
    """
    Evaluate descriptive statistics for one FCDT condition.

    Input
        sta : SimpleNamespace() object with required attributes:
            .d   : data directory (pathlib.Path)
            .p   : participant indices (sequence of integers)
            .c   : condition label, e.g. 'PTD-RMC'

    Output
        sta : input SimpleNamespace with the new attributes:
            .D   : trial-wise data for the requested condition only
            .a_* : participant statistics and group summaries

    Optional .R and .B restrict the data to one run and source block.
    Optional .x selects a simulation filename suffix, defaulting to ''.
    """
    # data loading
    # -------------------------------------------------------------------------
    n        = len(sta.p)                                                       # number of participants
    suf      = getattr(sta, 'x', '')                                            # optional simulation filename suffix
    dfs      = []                                                               # data frame collection
    for p in sta.p:                                                             # participant iterations
        file = sta.d / f'sub-{int(p):03}{suf}.csv'                              # participant filename
        di   = pd.read_csv(file, sep = '\t', header = 0)                        # participant data
        if 'RT' not in di:                                                      # unavailable simulated response times
            di.insert(12, 'RT', np.nan)                                         # common descriptive-data schema
        dfs.append(di[di['C'] == sta.c])                                        # retain requested condition only

    df       = pd.concat(dfs, ignore_index = True)                              # concatenate condition data
    if hasattr(sta, 'R'):                                                       # optional run selection
        df   = df.loc[df.R == sta.R]                                            # selected run
    if hasattr(sta, 'B'):                                                       # optional source-block selection
        df   = df.loc[df.B == sta.B]                                            # selected block
    sta.D    = df                                                               # attach trial-wise condition data

    # condition-specific descriptive statistics
    # -------------------------------------------------------------------------
    sta.a_trl = np.full([n], np.nan)                                            # number of trials
    sta.a_tim = np.full([n], np.nan)                                            # total time spent on trials
    sta.a_rew = np.full([n], np.nan)                                            # number of rewarded trials
    sta.a_rrt = np.full([n], np.nan)                                            # reward rate per second
    sta.a_rpt = np.full([n], np.nan)                                            # reward rate per trial
    sta.a_swi = np.full([n], np.nan)                                            # number of task switches
    sta.a_rst = np.full([n], np.nan)                                            # response times

    # participant iterations
    # -------------------------------------------------------------------------
    for i, p in enumerate(sta.p):                                               # participant iterations
        dpc          = df[df['P'] == p]                                         # participant-condition data
        dpc          = dpc.sort_values(['R', 'B', 't'])                         # chronological condition data
        end          = dpc.groupby(['R', 'B'], sort = False).tail(1)            # run-block terminal trials
        sta.a_trl[i] = dpc.shape[0]                                             # trials across all runs
        sta.a_tim[i] = (end['O'] + end['RT']).sum(min_count = 1)                # total time spent on trials
        sta.a_rew[i] = dpc['r'].sum()                                           # number of rewarded trials
        sta.a_rrt[i] = sta.a_rew[i] / sta.a_tim[i]                              # rewards per second
        sta.a_rpt[i] = sta.a_rew[i] / sta.a_trl[i]                              # rewards per trial
        sta.a_swi[i] = dpc['a1'].ne(dpc['a1'].shift()).iloc[1:].sum()           # number of task switches
        sta.a_rst[i] = dpc['RT'].mean()                                         # mean response time

    # group-level summaries (population SD and SD / sqrt(n), as before)
    # -------------------------------------------------------------------------
    sta.a_trl_mean = pd.Series(sta.a_trl).mean()                                # mean number of trials
    sta.a_trl_sd   = pd.Series(sta.a_trl).std(ddof = 0)                         # trial-count standard deviation
    sta.a_trl_sem  = sta.a_trl_sd / np.sqrt(n)                                  # trial-count standard error
    sta.a_tim_mean = pd.Series(sta.a_tim).mean()                                # mean total time spent on trials
    sta.a_tim_sd   = pd.Series(sta.a_tim).std(ddof = 0)                         # total time standard deviation
    sta.a_tim_sem  = sta.a_tim_sd / np.sqrt(n)                                  # total time standard error
    sta.a_rew_mean = pd.Series(sta.a_rew).mean()                                # mean rewarded trials
    sta.a_rew_sd   = pd.Series(sta.a_rew).std(ddof = 0)                         # rewarded trials standard deviation
    sta.a_rew_sem  = sta.a_rew_sd / np.sqrt(n)                                  # rewarded trials standard error
    sta.a_rrt_mean = pd.Series(sta.a_rrt).mean()                                # mean reward rate per second
    sta.a_rrt_sd   = pd.Series(sta.a_rrt).std(ddof = 0)                         # reward-rate standard deviation
    sta.a_rrt_sem  = sta.a_rrt_sd / np.sqrt(n)                                  # reward-rate standard error
    sta.a_rpt_mean = pd.Series(sta.a_rpt).mean()                                # mean reward rate per trial
    sta.a_rpt_sd   = pd.Series(sta.a_rpt).std(ddof = 0)                         # per-trial reward-rate standard deviation
    sta.a_rpt_sem  = sta.a_rpt_sd / np.sqrt(n)                                  # per-trial reward-rate standard error
    sta.a_swi_mean = pd.Series(sta.a_swi).mean()                                # mean number of task switches
    sta.a_swi_sd   = pd.Series(sta.a_swi).std(ddof = 0)                         # task-switch standard deviation
    sta.a_swi_sem  = sta.a_swi_sd / np.sqrt(n)                                  # task-switch standard error
    sta.a_rst_mean = pd.Series(sta.a_rst).mean()                                # mean response time
    sta.a_rst_sd   = pd.Series(sta.a_rst).std(ddof = 0)                         # response time standard deviation
    sta.a_rst_sem  = sta.a_rst_sd / np.sqrt(n)                                  # response time standard error

    # output specification
    # -------------------------------------------------------------------------
    return sta                                                                  # output specification
