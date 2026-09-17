"""
Count selected-stream transitions for all four FCDT conditions.
"""

# initialization
# -----------------------------------------------------------------------------
import json                                                                     # JSON metadata loading
import numpy as np                                                              # numerical operations
import pandas as pd                                                             # stimulus catalogue loading


def abm_count(sta):                                                             # observed label and identifier frequencies
    """
    Count experimental FCDT transitions and normalize observed frequencies.

    Input
        sta : SimpleNamespace() object with required attributes:
            .m : experimental metadata directory (pathlib.Path), containing
                 design.json and stimuli/stimuli.csv
            .D : trial data (pandas.DataFrame), with columns
                 P, R, B, C, t, sA, sB, a1, and a2

    Output
        sta : input SimpleNamespace with the new attributes:
            .F : dict of task dicts containing normalized frequencies
                 lno: label-1 probability after No, by previous identifier
                 lys: label-1 probability after Yes, by previous identifier
                 inn: identifier probabilities for negative -> negative
                 inp: identifier probabilities for negative -> positive
                 ipn: identifier probabilities for positive -> negative
                 ipp: identifier probabilities for positive -> positive

    Negative/positive mean stimulus labels 0/1, not response correctness.
    lno/lys have shape (n,). Identifier arrays have shape (n, n), with
    previous identifiers in rows and successor identifiers in columns.

    The conditions in .D['C'] determine which tasks are evaluated. Under .m,
    design.json supplies task_1/task_2 and S1T/S2T for each participant,
    run, source block, and condition. S1T/S2T list the stimuli with label 1
    in each stream; all other catalogue stimuli receive label 0.

    stimuli/stimuli.csv supplies the complete stimulus_name catalogue for
    each task_type, ordered by stimulus_id. The number of catalogue entries
    defines n, including stimuli not observed in .D. This is the size of the
    stimulus set, not a count of observed events. Missing fixed_label values
    identify tasks whose target sets vary between blocks. The actual event
    counts used for the frequencies are computed from transitions in .D.

    Array index i represents catalogue identifier i + 1. For tasks with varying
    target sets, inp/ipp use blockwise target-first indices: sorted targets,
    then sorted non-targets. LTD letter pairs are treated as unordered.

    All panels pool participants, runs, blocks, other-stream states, prior
    task choice, and imbalance. If .D contains multiple conditions, these
    are pooled within each task. lno/lys pool previous labels and successor
    identifiers; inn/inp/ipn/ipp pool answers. Incorrect responses remain.
    Each update contributes to one label and one identifier distribution.
    Only consecutive trials within a block with known stimulus pairs count.
    Rows without observed transitions yield NaN frequencies, not zeros.
    Raw counts are local intermediates; no output files are produced.
    """
    # metadata and block definitions
    # -------------------------------------------------------------------------
    file = sta.m / 'design.json'                                                # participant-specific experimental design
    spec = json.loads(file.read_text(encoding = 'utf-8'))                       # load block metadata
    cats = pd.read_csv(sta.m / 'stimuli' / 'stimuli.csv')                       # complete stimulus catalogue
    blocks = {}                                                                 # lookup by participant, run, source block, and condition
    for p, cond in spec['participants'].items():                                # participant definitions
        for c, runs in cond.items():                                            # all available conditions
            for r, reps in runs.items():                                        # run definitions
                for cfg in reps.values():                                       # local block definitions
                    key       = (int(p), int(r), cfg['source_block'], c)        # derivative block identity
                    blocks[key] = cfg                                           # task names and positive-label stimulus sets

    # task definitions, chronological order, and count allocation
    # -------------------------------------------------------------------------
    lbl = ('lno', 'lys')                                                        # label probabilities after No and Yes
    idt = (('inn', 'inp'), ('ipn', 'ipp'))                                      # identifier probabilities by old/new label
    key = ['P', 'R', 'B', 'C']                                                  # independent experimental blocks
    col = key + ['t', 'sA', 'sB', 'a1', 'a2']                                   # required trial columns
    df  = sta.D[col].sort_values(key + ['t'])                                   # ordered copy; preserve sta.D
    tsk = sorted({v for c in df['C'].unique() for v in c.split('-')})           # tasks present
    ids = {}                                                                    # original stimulus identifiers by task
    lut = {}                                                                    # stimulus-name lookups by task
    var = {}                                                                    # variable versus fixed label sets
    cnt = {}                                                                    # task-specific event-count dictionaries
    for task in tsk:                                                            # task iterations
        cat       = cats[cats['task_type'] == task]                             # task-specific metadata
        cat       = cat.sort_values('stimulus_id')                              # original identifier order
        ids[task] = cat['stimulus_name'].str.split().str.join(' ').to_numpy()   # normalized whitespace
        lut[task] = dict(zip(ids[task], range(len(cat))))                       # zero-based stimulus indices
        var[task] = cat['fixed_label'].isna().all()                             # metadata marks varying target sets

        n                  = len(cat)                                           # all task stimuli, including unobserved identifiers
        cnt[task]          = {}                                                 # six independent event-count arrays
        for evt in lbl + idt[0] + idt[1]:                                       # event-group iterations
            shp            = (n, 2) if evt in lbl else (n, n)                   # minimal per-distribution axes
            cnt[task][evt] = np.zeros(shp, dtype = np.int64)                    # raw integer counts

    # blockwise reconstruction and hard-coded pooling
    # -------------------------------------------------------------------------
    for bid, blk in df.groupby(key, sort = False):                              # never join across blocks
        cfg         = blocks[bid]                                               # block-specific task and target definitions
        act         = blk['a1'].to_numpy()                                      # selected stream before each update
        ans         = blk['a2'].to_numpy(dtype = int)                           # answer before each update
        adj         = np.diff(blk['t'].to_numpy()) == 1                         # never bridge missing trials
        for j, s in enumerate('AB', start = 1):                                 # selected-stream updates
            task    = cfg[f'task_{j}']                                          # task assigned to this stream
            n       = len(ids[task])                                            # catalogue size from metadata
            trg     = [' '.join(v.split()) for v in cfg[f'S{j}T']]              # positive-label names
            raw     = blk['s' + s].str.split().str.join(' ')                    # normalize verbal spacing
            if task == 'LTD':                                                   # letter-pair order is irrelevant
                trg = [' '.join(sorted(v.split())) for v in trg]                # canonical target pairs
                raw = raw.str.split().map(sorted, na_action = 'ignore')         # sort observed letters
                raw = raw.str.join(' ')                                         # canonical stimulus pairs
            lab     = np.isin(ids[task], trg).astype(int)                       # labels from design metadata
            idx     = raw.map(lut[task]).fillna(-1).to_numpy(dtype = int)       # missing sentinel
            pos     = np.flatnonzero((act[:-1] == s) & adj)                     # nonterminal selections
            pos     = pos[(idx[pos] >= 0) & (idx[pos + 1] >= 0)]                # observed stimulus pairs
            pre     = idx[pos]                                                  # previous original identifiers
            nxt     = idx[pos + 1]                                              # successor original identifiers
            old     = lab[pre]                                                  # previous labels
            new     = lab[nxt]                                                  # successor labels

            # label frequencies: retain answer, previous ID, and successor label
            for a, evt in enumerate(lbl):                                       # No and Yes label distributions
                sel = ans[pos] == a                                             # pool previous labels and successor identifiers
                np.add.at(cnt[task][evt], (pre[sel], new[sel]), 1)              # count both labels

            # identifier frequencies: retain label and ID pairs, pooling answers
            seq      = np.r_[np.flatnonzero(lab), np.flatnonzero(lab == 0)]     # targets first
            rec      = np.empty(n, dtype = int)                                 # blockwise identifier recoding
            rec[seq] = np.arange(n)                                             # sorted target and non-target ranks
            for o in [0, 1]:                                                    # previous labels
                for v in [0, 1]:                                                # successor labels
                    evt     = idt[o][v]                                         # identifier distribution for this label pair
                    sel     = (old == o) & (new == v)                           # both answers contribute
                    bef     = pre[sel]                                          # previous original identifiers
                    aft     = nxt[sel]                                          # successor original identifiers
                    if var[task] and v == 1:                                    # align varying target sets for positive successors
                        bef = rec[bef]                                          # previous target-first identifier
                        aft = rec[aft]                                          # successor target-first identifier
                    np.add.at(cnt[task][evt], (bef, aft), 1)                    # raw transition counts

    # normalization
    # -------------------------------------------------------------------------
    sta.F = {}                                                                  # task-specific frequency dictionaries
    for task in cnt:                                                            # task iterations
        sta.F[task] = {}                                                        # normalized frequency arrays
        for evt in lbl + idt[0] + idt[1]:                                       # event-group iterations
            mat             = cnt[task][evt]                                    # raw event-count matrix
            den             = mat.sum(axis = 1, keepdims = True)                # row-specific event totals
            frq             = np.divide(                                        # normalize only observed rows
            mat,                                                                # raw event counts
            den,                                                                # matching conditional denominators
            out   = np.full(mat.shape, np.nan),                                 # unobserved rows remain undefined
            where = den > 0)                                                    # avoid division by zero
            sta.F[task][evt] = frq[:, 1] if evt in lbl else frq                 # bar vectors or transition matrices

    # output specification
    # -------------------------------------------------------------------------
    return sta                                                                  # return normalized frequencies
