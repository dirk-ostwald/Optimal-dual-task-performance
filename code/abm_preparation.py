"""
Prepare BIDS responses, trial states, and task metadata for the FCDT experiment.
"""

# initialization
# -----------------------------------------------------------------------------
import json                                                                     # JSON metadata
import re                                                                       # stimulus-name parsing
import shutil as shu                                                            # stimulus asset copying
from itertools import combinations as comb                                      # unordered letter-pair catalogue
from pathlib import Path                                                        # directory management
from types import SimpleNamespace                                               # shared preparation state

import numpy as np                                                              # Boolean scalar types
import pandas as pd                                                             # response and trial tables
from PIL import Image                                                           # stimulus image conversion


def abm_bids(pre):                                                              # source-to-BIDS conversion
    """
    Convert the OSF response table and stimulus assets to a BIDS dataset.

    Input
        pre : SimpleNamespace() object with required attributes:
            .s : OSF source directory (pathlib.Path)
            .b : BIDS output directory (pathlib.Path)

    Output
        pre : input SimpleNamespace with the new attributes:
            .p : sorted participant indices (list of integers)
            .n : number of retained dual-task response rows (integer)

        Writes participant/run TSV files, metadata, and stimulus assets to .b.
        Existing generated files are replaced; unrelated files are retained.
    """
    # source data and path normalization
    # -------------------------------------------------------------------------
    file            = pre.s / 'raw data' / 'Data_Exp1_raw.csv'                  # source response file
    df              = pd.read_csv(                                              # semicolon-delimited source table
    file,                                                                       # source CSV path
    sep             = ';',                                                      # source field separator
    decimal         = ',',                                                      # source decimal separator
    na_values       = ['NA'],                                                   # missing-value marker
    low_memory      = False)                                                    # infer types from complete columns
    df              = df[df['trialType'] != 'ST'].copy()                        # retain dual-task rows
    df              = df.drop(columns = ['trialType'])                          # omit source row-type flag
    vals            = []                                                        # normalized stimulus references
    for val in df['stimulus']:                                                  # source stimulus values
        if pd.notna(val):                                                       # preserve missing values
            val     = str(val).replace('\\', '/')                               # portable path separators
            if '/res/list/' in val:                                             # absolute source-machine path
                val = val.split('/res/list/', 1)[1]                             # relative stimulus reference
            elif val.startswith('res/list/'):                                   # source-relative path
                val = val[len('res/list/'):]                                    # remove source prefix
            elif val.startswith('stimulus material/'):                          # OSF-relative path
                val = val[len('stimulus material/'):]                           # remove OSF prefix
        vals.append(val)                                                        # retain text stimuli and missing values
    df['stimulus']  = vals                                                      # replace source-machine references

    # dataset directories and participant table
    # -------------------------------------------------------------------------
    pre.p = [int(p) for p in sorted(df['VP'].unique())]                         # available participants
    pre.n = len(df)                                                             # retained response count
    blocks = sorted(df['blockNo'].dropna().unique())                            # source run order
    runs  = {int(b): i + 1 for i, b in enumerate(blocks)}                       # one-based BIDS runs
    sdir  = pre.b / 'stimuli'                                                   # BIDS stimulus directory
    sdir.mkdir(parents = True, exist_ok = True)                                 # create output directories
    shu.copytree(pre.s / 'stimulus material', sdir, dirs_exist_ok = True)       # copy assets
    part  = pd.DataFrame(                                                       # BIDS participant table
    {                                                                           # record fields
        'participant_id': [f'sub-{p:03d}' for p in pre.p],                      # BIDS identifiers
        'VP'            : pre.p})                                               # source participant numbers
    file  = pre.b / 'participants.tsv'                                          # participant table path
    part.to_csv(                                                                # write BIDS participant table
    file,                                                                       # output TSV path
    sep = '\t',                                                                 # BIDS field separator
    index = False,                                                              # omit dataframe index
    na_rep = 'n/a',                                                             # BIDS missing-value marker
    lineterminator = '\n')                                                      # stable line endings

    # BIDS metadata
    # -------------------------------------------------------------------------
    meta                             = {}                                       # shared dataset metadata
    meta['dataset_description.json'] = {                                        # dataset metadata
        'Name'             : 'FCDT - Experiment 1 of Brüning et al. (2020)',    # Name metadata
        'BIDSVersion'      : '1.11.1',                                          # BIDSVersion metadata
        'DatasetType'      : 'raw',                                             # DatasetType metadata
        'Authors'          : [                                                  # Authors metadata
            'Jovita Brüning',                                                   # Authors metadata
            'Marie Mückstein',                                                  # Authors metadata
            'Dietrich Manzey'],                                                 # Authors metadata
        'HowToAcknowledge' : ( 'Please cite Brüning, Mückstein, and Manzey '    # HowToAcknowledge metadata
                               '(2020).'),                                      # HowToAcknowledge metadata
        'GeneratedBy'      : [                                                  # GeneratedBy metadata
            {                                                                   # GeneratedBy metadata
                'Name'        : 'abm_preparation.py',                           # Name metadata
                'Description' : ( 'Local conversion from the OSF '              # Description metadata
                                  'semicolon-delimited CSV file for '           # Description metadata
                                  'Experiment 1 of Brüning et al. (2020) to '   # Description metadata
                                  'BIDS behavioral TSV files.')}]}              # Description metadata
    meta['participants.json']        = {                                        # dataset metadata
        'participant_id' : {                                                    # participant_id metadata
            'Description' : 'BIDS participant identifier.'},                    # Description metadata
        'VP'             : {                                                    # VP metadata
            'Description' : 'Participant number in the OSF source file.'}}      # Description metadata
    meta['task-fcdt_beh.json']       = {                                        # dataset metadata
        'TaskName'     : 'fcdt',                                                # TaskName metadata
        'Description'  : ( 'Behavioral FCDT data from Experiment 1 of '         # Description metadata
                           'Brüning et al. (2020). Original source '            # Description metadata
                           'variable names are preserved.'),                    # Description metadata
        'Columns'      : [                                                      # Columns metadata
            'idx',                                                              # Columns metadata
            'VP',                                                               # Columns metadata
            'resource',                                                         # Columns metadata
            'condition',                                                        # Columns metadata
            'blockNo',                                                          # Columns metadata
            'trial',                                                            # Columns metadata
            'task',                                                             # Columns metadata
            'stimulus',                                                         # Columns metadata
            'value',                                                            # Columns metadata
            'key',                                                              # Columns metadata
            'response',                                                         # Columns metadata
            'stim_pres_ms',                                                     # Columns metadata
            'response_ms'],                                                     # Columns metadata
        'idx'          : {                                                      # idx metadata
            'Description' : ( 'Number of the current row in the OSF '           # Description metadata
                              'source file.')},                                 # Description metadata
        'VP'           : {                                                      # VP metadata
            'Description' : 'Participant number in the OSF source file.'},      # Description metadata
        'resource'     : {                                                      # resource metadata
            'Description' : ( 'Type of resource competition. Level uni '        # Description metadata
                              'involves task pairs competing for the same '     # Description metadata
                              'perceptual-cognitive resources; level mix '      # Description metadata
                              'involves task pairs competing for different '    # Description metadata
                              'perceptual-cognitive resources.'),               # Description metadata
            'Levels'      : {                                                   # Levels metadata
                'mix' : ( 'Task pairs competing for different '                 # mix metadata
                          'perceptual-cognitive resources.'),                   # mix metadata
                'uni' : ( 'Task pairs competing for the same '                  # uni metadata
                          'perceptual-cognitive resources.')}},                 # uni metadata
        'condition'    : {                                                      # condition metadata
            'Description' : ( 'Experimental condition code in the OSF '         # Description metadata
                              'source file. The OSF workbook documents '        # Description metadata
                              'UniR, UniV, VgRk, and VkRg as the four '         # Description metadata
                              'combinations of verbal/spatial '                 # Description metadata
                              'memory-search and classification tasks. The '    # Description metadata
                              'acronym components are read as Uni = '           # Description metadata
                              'uniform/same processing code, V = verbal, R '    # Description metadata
                              '= spatial/raeumlich, g = '                       # Description metadata
                              'memory-search/Gedaechtnis, and k = '             # Description metadata
                              'classification/Klassifikation.'),                # Description metadata
            'Levels'      : {                                                   # Levels metadata
                'UniR' : ( 'Uniform spatial condition: spatial '                # UniR metadata
                           'memory-search task and spatial classification '     # UniR metadata
                           'task; high resource competition because both '      # UniR metadata
                           'tasks require spatial processing-code '             # UniR metadata
                           'resources.'),                                       # UniR metadata
                'UniV' : ( 'Uniform verbal condition: verbal memory-search '    # UniV metadata
                           'task and verbal classification task; high '         # UniV metadata
                           'resource competition because both tasks '           # UniV metadata
                           'require verbal processing-code resources.'),        # UniV metadata
                'VgRk' : ( 'Mixed verbal-memory/spatial-classification '        # VgRk metadata
                           'condition: verbal memory-search task and '          # VgRk metadata
                           'spatial classification task; low resource '         # VgRk metadata
                           'competition because verbal and spatial '            # VgRk metadata
                           'processing-code resources differ.'),                # VgRk metadata
                'VkRg' : ( 'Mixed spatial-memory/verbal-classification '        # VkRg metadata
                           'condition: spatial memory-search task and '         # VkRg metadata
                           'verbal classification task; low resource '          # VkRg metadata
                           'competition because spatial and verbal '            # VkRg metadata
                           'processing-code resources differ.')}},              # VkRg metadata
        'blockNo'      : {                                                      # blockNo metadata
            'Description' : ( 'Number of the current block. A task-pair '       # Description metadata
                              'block entails two subblocks separated by a '     # Description metadata
                              'short break. This value was mapped to BIDS '     # Description metadata
                              'run labels by sorted order while the '           # Description metadata
                              'original value was preserved.')},                # Description metadata
        'trial'        : {                                                      # trial metadata
            'Description' : ( 'Continuously numbered trial within the '         # Description metadata
                              'current source (sub-)block.')},                  # Description metadata
        'task'         : {                                                      # task metadata
            'Description' : ( 'Currently performed component task. A '          # Description metadata
                              'denotes one of the two memory-search tasks, '    # Description metadata
                              'depending on condition; B denotes one of '       # Description metadata
                              'the two classification tasks, depending on '     # Description metadata
                              'condition.'),                                    # Description metadata
            'Levels'      : {                                                   # Levels metadata
                'A' : 'Memory-search component task, depending on condition.',  # A metadata
                'B' : ( 'Classification component task, depending on '          # B metadata
                        'condition.')}},                                        # B metadata
        'stimulus'     : {                                                      # stimulus metadata
            'Description' : ( 'Stimulus perceived by the participant. Image '   # Description metadata
                              'stimulus paths are normalized to paths '         # Description metadata
                              'relative to the BIDS stimuli directory; '        # Description metadata
                              'verbal stimuli are represented directly as '     # Description metadata
                              'text.')},                                        # Description metadata
        'value'        : {                                                      # value metadata
            'Description' : ( 'Expected response. TRUE denotes memory '         # Description metadata
                              'target, same parity, or not mirrored; FALSE '    # Description metadata
                              'denotes the respective no response; missing '    # Description metadata
                              'values denote VoiceKey errors.')},               # Description metadata
        'key'          : {                                                      # key metadata
            'Description' : ( 'Response given by the participant. TRUE '        # Description metadata
                              'denotes keys K/S or vocal response green; '      # Description metadata
                              'FALSE denotes keys L/A or vocal response red.')},# Description metadata
        'response'     : {                                                      # response metadata
            'Description' : ( 'Match of value with key. TRUE means the '        # Description metadata
                              'given response equals the expected response; '   # Description metadata
                              'FALSE means key and value differ.')},            # Description metadata
        'stim_pres_ms' : {                                                      # stim_pres_ms metadata
            'Description' : ( 'Task-specific timestamp of the stimulus '        # Description metadata
                              'presentation. Inter-response intervals '         # Description metadata
                              'rather than simple reaction times are '          # Description metadata
                              'needed for the two task streams.'),              # Description metadata
            'Units'       : 'ms'},                                              # Units metadata
        'response_ms'  : {                                                      # response_ms metadata
            'Description' : ( 'Timestamp of the response button or vocal '      # Description metadata
                              'response.'),                                     # Description metadata
            'Units'       : 'ms'}}                                              # Units metadata
    for name, obj in meta.items():                                              # metadata file iterations
        file = pre.b / name                                                     # metadata output path
        text = json.dumps(obj, indent = 2, ensure_ascii = False) + '\n'         # JSON
        file.write_text(text, encoding = 'utf-8')                               # write shared metadata

    text = ( '                                                                  # FCDT BIDS behavioral dataset\n\nThis dataset contains the '     # BIDS conversion notes
             'FCDT data from Experiment 1 of Brüning et al. (2020).\n\nIt '     # BIDS conversion notes
             'was converted from the OSF source file '                          # BIDS conversion notes
             '`Data_Exp1_raw.csv`.\n\nThe original OSF variable names were '    # BIDS conversion notes
             'preserved. No derived BIDS-friendly columns\nsuch as `onset`, '   # BIDS conversion notes
             '`duration`, `trial_type`, or `response_time` were added.\n\n'     # BIDS conversion notes
             'Source `blockNo` values were mapped to BIDS run labels by '       # BIDS conversion notes
             'sorted order:\n`2 -> run-01`, `3 -> run-02`, `4 -> '              # BIDS conversion notes
             'run-03`.\n\nImage stimulus paths in the `stimulus` column '       # BIDS conversion notes
             'were normalized to paths relative\nto this dataset\'s '           # BIDS conversion notes
             '`stimuli/` directory, for example `images/13_50_k.jpg`. Text\n'   # BIDS conversion notes
             'stimuli remain represented directly in the `stimulus` column. '   # BIDS conversion notes
             'The OSF stimulus\nmaterial is copied into the `stimuli/` '        # BIDS conversion notes
             'directory.\n\n')                                                  # BIDS conversion notes
    (pre.b / 'README').write_text(text, encoding = 'utf-8')                     # write dataset notes
    text = ( 'Stimulus assets copied from `data/osf/stimulus material`. '       # stimulus path documentation
             'Image paths in behavioral files are relative to this BIDS '       # stimulus path documentation
             '`stimuli/` directory.\n')                                         # stimulus path documentation
    (sdir / 'README.md').write_text(text, encoding = 'utf-8')                   # write asset notes

    # participant/run response tables
    # -------------------------------------------------------------------------
    for p in pre.p:                                                             # participant iterations
        subj                 = f'sub-{p:03d}'                                   # BIDS participant identifier
        odir                 = pre.b / subj / 'beh'                             # behavioral output directory
        data                 = df[df['VP'] == p]                                # participant response rows
        odir.mkdir(parents = True, exist_ok = True)                             # create behavior directory
        for b in blocks:                                                        # source run iterations
            out              = data[data['blockNo'] == b].copy()                # participant/run table
            if out.empty:                                                       # absent participant/run combination
                continue                                                        # omit empty behavioral files
            for col in out.columns:                                             # Boolean column normalization
                vals         = out[col].dropna()                                # observed column values
                flag         = pd.api.types.is_bool_dtype(out[col])             # Boolean dtype
                if not vals.empty:                                              # object columns may also contain Booleans
                    flag     = flag or all(                                     # recognize Boolean scalar objects
                    isinstance(v, (bool, np.bool_)) for v in vals)              # scalar types
                if flag:                                                        # BIDS lowercase Boolean representation
                    out[col] = out[col].map({True: 'true', False: 'false'})     # values
            run              = f'run-{runs[int(b)]:02d}'                        # BIDS run identifier
            file             = odir / f'{subj}_task-fcdt_{run}_beh.tsv'         # response output
            out.to_csv(                                                         # write participant/run BIDS responses
            file,                                                               # output TSV path
            sep = '\t',                                                         # BIDS field separator
            index = False,                                                      # omit dataframe index
            na_rep = 'n/a',                                                     # BIDS missing-value marker
            lineterminator = '\n')                                              # stable line endings
    return pre                                                                  # retain the input namespace and attach participant coverage


def abm_trials(pre):                                                            # trial-state reconstruction
    """
    Reconstruct the visible task streams at each participant response.

    Input
        pre : SimpleNamespace() object with required attributes:
            .b : BIDS input directory (pathlib.Path)
            .d : derivative output directory (pathlib.Path)
            .p : participant indices (sequence of integers)
            .c : source-condition to analysis-condition mapping (dict)

    Output
        pre : input SimpleNamespace with the new attributes:
            .f : written participant derivative paths (list of pathlib.Path)

        Writes one tab-separated sub-                                           ###.csv per participant, plus metadata.
        Rows retain both visible stimuli before the selected response; a stream
        becomes missing after its final response. RT is the selected stimulus's
        response time, while O advances with the preceding response event.
        The derivative trial index t runs from zero to n - 1 within each block.
        Column a_t stores the action tuple (j, k) translated from a1 and a2.
        CSV readers deserialize this tuple before passing it to the model.
    """
    # participant metadata
    # -------------------------------------------------------------------------
    pre.d.mkdir(parents = True, exist_ok = True)                                # create derivative directory
    pre.f = []                                                                  # generated participant file paths
    file  = pre.b / 'participants.tsv'                                          # BIDS participant table
    part  = pd.read_csv(file, sep = '\t')                                       # participant identifiers
    part  = part.rename(columns = {'VP': 'P'})                                  # compact participant field
    file  = pre.d / 'participants.csv'                                          # derivative participant table
    part.to_csv(                                                                # write derivative participant metadata
    file,                                                                       # output table path
    sep = '\t',                                                                 # tab-separated derivative format
    index = False,                                                              # omit dataframe index
    na_rep = 'n/a',                                                             # missing-value marker
    lineterminator = '\n')                                                      # stable line endings
    meta                             = {}                                       # shared dataset metadata
    meta['dataset_description.json'] = {                                        # dataset metadata
        'Name'             : ( 'FCDT trial-by-trial ABM derivatives - '         # Name metadata
                               'Experiment 1 of Brüning et al. (2020)'),        # Name metadata
        'BIDSVersion'      : '1.11.1',                                          # BIDSVersion metadata
        'DatasetType'      : 'derivative',                                      # DatasetType metadata
        'Authors'          : [                                                  # Authors metadata
            'Jovita Brüning',                                                   # Authors metadata
            'Marie Mückstein',                                                  # Authors metadata
            'Dietrich Manzey'],                                                 # Authors metadata
        'HowToAcknowledge' : ( 'Please cite Brüning, Mückstein, and Manzey '    # HowToAcknowledge metadata
                               '(2020).'),                                      # HowToAcknowledge metadata
        'GeneratedBy'      : [                                                  # GeneratedBy metadata
            {                                                                   # GeneratedBy metadata
                'Name'        : 'abm_preparation.py',                           # Name metadata
                'Description' : ( 'Local conversion from FCDT BIDS '            # Description metadata
                                  'response-level TSV files for Experiment '    # Description metadata
                                  '1 of Brüning et al. (2020) to flat '         # Description metadata
                                  'participant-level tab-separated '            # Description metadata
                                  'derivative CSV files.')}],                   # Description metadata
        'SourceDatasets'   : [                                                  # SourceDatasets metadata
            {                                                                   # SourceDatasets metadata
                'URL'         : '../bids',                                      # URL metadata
                'Description' : ( 'Local BIDS behavioral dataset generated '    # Description metadata
                                  'from the OSF source files.')}],              # Description metadata
        'FileOrganization' : ( 'Participant-level tab-separated derivative '    # FileOrganization metadata
                               'CSV files are stored directly in this '         # FileOrganization metadata
                               'Derivatives directory as sub-                   ###.csv. The '     # FileOrganization metadata
                               'generated design.json file documents '          # FileOrganization metadata
                               'participant- and block-specific task '          # FileOrganization metadata
                               'configurations. Stimulus material inherited '   # FileOrganization metadata
                               'from the source BIDS dataset and the '          # FileOrganization metadata
                               'generated stimuli.csv catalogue are stored '    # FileOrganization metadata
                               'in stimuli/.')}                                 # FileOrganization metadata
    meta['participants.json']        = {                                        # dataset metadata
        'participant_id' : {                                                    # participant_id metadata
            'Description' : ( 'BIDS participant identifier inherited from '     # Description metadata
                              'the source BIDS dataset.')},                     # Description metadata
        'P'              : {                                                    # P metadata
            'Description' : 'Participant number without BIDS zero padding.'}}   # Description metadata
    meta['dataset_codebook.json']    = {                                        # dataset metadata
        'TaskName'       : 'trial-by-trial',                                    # TaskName metadata
        'Description'    : ( 'Trial-by-trial derivative reconstructed from '    # Description metadata
                             'the FCDT BIDS response-level files for '          # Description metadata
                             'Experiment 1 of Brüning et al. (2020). Each '     # Description metadata
                             'row is one response event with the visible '      # Description metadata
                             'stimulus in task stream A and task stream B '     # Description metadata
                             'before that response.'),                          # Description metadata
        'Columns'        : [                                                    # Columns metadata
            'P',                                                                # Columns metadata
            'R',                                                                # Columns metadata
            'B',                                                                # Columns metadata
            'C',                                                                # Columns metadata
            'T',                                                                # Columns metadata
            'O',                                                                # Columns metadata
            't',                                                                # Columns metadata
            'sA',                                                               # Columns metadata
            'sB',                                                               # Columns metadata
            'a1',                                                               # Columns metadata
            'a2',
            'a_t',                                                              # prepared task-response tuple                                                               # Columns metadata
            'r',                                                                # Columns metadata
            'RT'],                                                              # Columns metadata
        'ParadigmNaming' : ( 'Paradigm names combine the target-detection '     # ParadigmNaming metadata
                             'task (PTD or LTD) and classification task '       # ParadigmNaming metadata
                             '(RMC or DPC).'),                                  # ParadigmNaming metadata
        'P'              : 'Participant number without BIDS zero padding.',     # P metadata
        'R'              : 'BIDS run number.',                                  # R metadata
        'B'              : ( 'Continuous local block within a run, '            # B metadata
                             'incremented when condition, trial counter, '      # B metadata
                             'or local response clock resets.'),                # B metadata
        'C'              : {                                                    # C metadata
            'Description' : 'Free-concurrent dual-task paradigm.',              # Description metadata
            'Levels'      : {                                                   # Levels metadata
                'PTD-RMC' : ( '2D pattern target detection and 3D '             # PTD-RMC metadata
                              'rotation/mirror classification.'),               # PTD-RMC metadata
                'PTD-DPC' : ( '2D pattern target detection and '                # PTD-DPC metadata
                              'digit-parity classification.'),                  # PTD-DPC metadata
                'LTD-RMC' : ( 'Letter-pair target detection and 3D '            # LTD-RMC metadata
                              'rotation/mirror classification.'),               # LTD-RMC metadata
                'LTD-DPC' : ( 'Letter-pair target detection and '               # LTD-DPC metadata
                              'digit-parity classification.')}},                # LTD-DPC metadata
        'T'              : ( 'Inferred target stimulus set for the '            # T metadata
                             'memory-search task in this block. Inferred '      # T metadata
                             'from positive target trials (source value '       # T metadata
                             'true) in the same block. Verbal pairs are '       # T metadata
                             'canonicalized because order was irrelevant '      # T metadata
                             'in the task.'),                                   # T metadata
        'O'              : {                                                    # O metadata
            'Description' : ( 'Trial onset in seconds, relative to the '        # Description metadata
                              'first stimulus onset of the current '            # Description metadata
                              'continuous block. The first trial starts at '    # Description metadata
                              '0.000; each following trial starts at the '      # Description metadata
                              'previous response time.'),                       # Description metadata
            'Units'       : 's'},                                               # Units metadata
        't'              : 'Zero-based trial index within block.',              # t metadata
        'sA'             : ( 'Visible stimulus in task stream A '               # sA metadata
                             'immediately before the selected response. '       # sA metadata
                             'Image stimuli are compacted from '                # sA metadata
                             'BIDS-relative rimages/... paths to filename '     # sA metadata
                             'stems and can be resolved against PNG files '     # sA metadata
                             'in the derivative stimuli/images directory. '     # sA metadata
                             'Terminal block states may be n/a when the '       # sA metadata
                             'source data do not contain the final '            # sA metadata
                             'unresponded stimulus for this stream.'),          # sA metadata
        'sB'             : ( 'Visible stimulus in task stream B '               # sB metadata
                             'immediately before the selected response. '       # sB metadata
                             'Image stimuli are compacted from '                # sB metadata
                             'BIDS-relative rimages/... paths to filename '     # sB metadata
                             'stems and can be resolved against PNG files '     # sB metadata
                             'in the derivative stimuli/images directory. '     # sB metadata
                             'Terminal block states may be n/a when the '       # sB metadata
                             'source data do not contain the final '            # sB metadata
                             'unresponded stimulus for this stream.'),          # sB metadata
        'a1'             : ( 'First action: participant task choice, coded '    # a1 metadata
                             'as the selected task stream A or B.'),            # a1 metadata
        'a2'             : ( 'Second action: participant task response, '       # a2 metadata
                             'mapped to 1 for yes/true and 0 for no/false.'),   # a2 metadata
        'a_t'            : ( 'Action tuple (j, k), with task j = 1 or 2 '       # action metadata
                             'and binary response k = 0 or 1. Stored as '       # action metadata
                             'tuple text in CSV, e.g. (1, 0).'),                # action metadata
        'r'              : ( 'Reward feedback, coded as 1 for correct '         # r metadata
                             'responses and 0 for incorrect responses.'),       # r metadata
        'RT'             : {                                                    # RT metadata
            'Description' : ( 'Post-stimulus response time for the '            # Description metadata
                              'selected task stimulus in seconds, '             # Description metadata
                              'reconstructed as response_ms minus '             # Description metadata
                              'stim_pres_ms.'),                                 # Description metadata
            'Units'       : 's'}}                                               # Units metadata
    for name, obj in meta.items():                                              # metadata file iterations
        file = pre.d / name                                                     # metadata output path
        text = json.dumps(obj, indent = 2, ensure_ascii = False) + '\n'         # JSON
        file.write_text(text, encoding = 'utf-8')                               # write shared metadata

    text = ( '                                                                  # FCDT trial-by-trial derivative dataset\n\nThis derivative '     # trial-state documentation
             'dataset contains the trial-by-trial FCDT data from\n'             # trial-state documentation
             'Experiment 1 of Brüning et al. (2020).\n\nIt was generated '      # trial-state documentation
             'from the local FCDT BIDS behavioral dataset by\n'                 # trial-state documentation
             '`abm_preparation.py`.\n\nParticipant-level trial-by-trial '       # trial-state documentation
             'derivative files are stored directly in this\ndirectory as '      # trial-state documentation
             'tab-separated `sub-                                               ###.csv` files. The files use compact '        # trial-state documentation
             'analysis\ncolumn names:\n\n`P`, `R`, `B`, `C`, `T`, `O`, `t`, '   # trial-state documentation
             '`sA`, `sB`, `a1`, `a2`, `a_t`, `r`, `RT`.\n\nThe shared file '    # trial-state documentation
             '`dataset_codebook.json` documents these columns. The files\n'     # trial-state documentation
             '`participants.csv` and `participants.json` document '             # trial-state documentation
             'participant identifiers.\nStimulus files inherited from the '     # trial-state documentation
             'source BIDS dataset are available in\n`stimuli/`; compact '       # trial-state documentation
             'image stems in `sA`, `sB`, and `T` resolve against\nPNG files '   # trial-state documentation
             'in `stimuli/images/`.\n\nThe generated file '                     # trial-state documentation
             '`stimuli/stimuli.csv` maps task-specific integer identifiers\n'   # trial-state documentation
             'to the concrete PTD, LTD, DPC, and RMC stimuli and records '      # trial-state documentation
             'fixed binary labels\nwhere these are defined independently of '   # trial-state documentation
             'participant and block.\n\nThe generated file `design.json` '      # trial-state documentation
             'documents the target configuration for each\nparticipant, '       # trial-state documentation
             'task pair, run, and first or second block repetition. It '        # trial-state documentation
             'also\nretains the corresponding local source block number.\n\n'   # trial-state documentation
             'Column `C` identifies the paradigm as `PTD-RMC`, `PTD-DPC`, '     # trial-state documentation
             '`LTD-RMC`, or\n`LTD-DPC`.\n\n')                                   # trial-state documentation
    (pre.d / 'README').write_text(text, encoding = 'utf-8')                     # write dataset notes

    # response-to-trial reconstruction
    # -------------------------------------------------------------------------
    cols = ( 'P', 'R', 'B', 'C', 'T', 'O', 't', 'sA', 'sB',                     # trial-state fields
             'a1', 'a2', 'a_t', 'r', 'RT')                                      # selected action, reward, and response time
    for p in pre.p:                                                             # participant iterations
        subj              = f'sub-{p:03d}'                                      # BIDS participant identifier
        sdir              = pre.b / subj / 'beh'                                # participant BIDS responses
        rows              = []                                                  # participant trial-state records
        for file in sorted(sdir.glob(f'{subj}_task-fcdt_run-*_beh.tsv')):       # runs
            run           = int(file.stem.split('_')[2][4:])                    # run index
            df            = pd.read_csv(file, sep = '\t')                       # source response table
            df['RT']      = df['response_ms'] - df['stim_pres_ms']              # selected-task RT
            prev          = df.shift()                                          # preceding response in source order
            rst           = ( df['condition'].ne(prev['condition'])             # condition change
                              | df['trial'].le(prev['trial'])                   # local trial reset
                              | df['response_ms'].lt(prev['response_ms']))      # clock reset
            df['episode'] = rst.cumsum()                                        # one-based continuous block index

            # compact stimulus paths and binary response values
            vals            = []                                                # compact stimulus identifiers
            for val in df['stimulus']:                                          # visible stimulus values
                if pd.isna(val):                                                # missing source stimulus
                    val     = 'n/a'                                             # explicit terminal/missing marker
                else:                                                           # image reference or text stimulus
                    val     = str(val)                                          # source stimulus text
                    if val.startswith(('res/list/', 'images/', 'rimages/')):    # image
                        val = Path(val).stem                                    # image filename without suffix
                vals.append(val)                                                # collect compact stimulus
            df['stimulus']  = vals                                              # compact visible-stimulus column
            for col in ('key', 'response'):                                     # answer and correctness columns
                vals    = df[col].fillna('n/a').astype(str)                     # response strings
                vals    = vals.str.strip().str.lower()                          # normalized Boolean text
                code    = {'true': 1, 'false': 0}                               # binary response mapping
                df[col] = [code.get(v, v) for v in vals]                        # retain other values

            for b, data in df.groupby('episode', sort = False):                 # local blocks
                # memory targets and initially visible task streams
                cond = pre.c[str(data['condition'].iloc[0])]                    # analysis condition
                keep = ( data['task'].eq('A')                                   # memory-search stream
                         & data['value'].astype(str).str.lower().eq('true')     # targets
                         & data['stimulus'].ne('n/a'))                          # observed target stimuli
                vals = data.loc[keep, 'stimulus'].tolist()                      # positive target trials
                if cond.startswith('LTD'):                                      # order-independent letter pairs
                    for i, val in enumerate(vals):                              # positive letter stimuli
                        pair        = val.split()                               # separate letter tokens
                        if len(pair) == 2 and all(len(x) == 1 for x in pair):   # pair
                            vals[i] = ' '.join(sorted(pair))                    # canonical order
                mem            = ' | '.join(sorted(set(vals))) or 'n/a'         # targets
                evts           = {}                                             # response rows within each task stream
                for task, grp in data.groupby('task', sort = False):            # streams
                    grp        = grp.sort_values('response_ms')                 # stream response order
                    evts[task] = list(grp.itertuples(index = False))            # stream rows
                pos            = {task: 0 for task in evts}                     # visible stream positions
                vis            = {task: seq[0] for task, seq in evts.items()}   # first stimuli
                base           = data['stim_pres_ms'].min()                     # block onset in milliseconds
                now            = base                                           # current response-state onset
                data           = data.sort_values('response_ms')                # chronological responses
                seq            = data.itertuples(index = False)                 # response events
                for n, evt in enumerate(seq):                                   # visible response states
                    task        = evt.task                                      # responded task stream
                    stim        = {}                                            # visible stimuli before this response
                    for t in ('A', 'B'):                                        # both task streams
                        row     = vis.get(t)                                    # currently visible stream row
                        stim[t] = row.stimulus if row is not None else 'n/a'    # ID
                    j           = 1 if task == 'A' else 2                       # numerical task in the full action
                    k           = int(evt.key)                                  # binary response in the full action
                    rows.append(                                                # complete trial record
                    {                                                           # record fields
                        'P' : p,                                                # participant index
                        'R' : run,                                              # run index
                        'B' : b,                                                # continuous local block
                        'C' : cond,                                             # analysis condition
                        'T' : mem,                                              # block memory set
                        'O' : f'{(now - base) / 1000:.3f}',                     # local onset in seconds
                        't' : n,                                                # zero-based response-state index
                        'sA': stim['A'],                                        # visible memory-task stimulus
                        'sB': stim['B'],                                        # visible classification stimulus
                        'a1': task,                                             # selected task stream
                        'a2': evt.key,                                          # binary response
                        'a_t': (j, k),                                          # prepared full action tuple
                        'r' : evt.response,                                     # binary correctness
                        'RT': f'{evt.RT / 1000:.3f}'})                          # selected-stimulus RT
                    pos[task] += 1                                              # advance only the selected stream
                    idx           = pos[task]                                   # next selected-stream row
                    if idx < len(evts[task]):                                   # successor observed in source
                        vis[task] = evts[task][idx]                             # reveal next stimulus
                    else:                                                       # no further response in the selected stream
                        vis[task] = None                                        # final unobserved stimulus is unknown
                    now           = evt.response_ms                             # next trial starts at this response

        # participant derivative output
        out  = pd.DataFrame(rows, columns = cols)                               # ordered trial table
        file = pre.d / f'{subj}.csv'                                            # participant derivative path
        out.to_csv(                                                             # write reconstructed trial states
        file,                                                                   # output CSV path
        sep = '\t',                                                             # tab-separated derivative format
        index = False,                                                          # omit dataframe index
        na_rep = 'n/a',                                                         # missing-value marker
        lineterminator = '\n')                                                  # stable line endings
        pre.f.append(file)                                                      # retain generated participant paths
        print(f'{subj}: {len(out):,} trial states', flush = True)               # progress
    return pre                                                                  # retain the input namespace and attach derivative paths


def abm_design(pre):                                                            # stimulus catalogue and experimental design
    """
    Build stimulus identities and participant/block target configurations.

    Input
        pre : SimpleNamespace() object with required attributes:
            .b : BIDS input directory (pathlib.Path)
            .d : derivative output directory (pathlib.Path)
            .f : participant derivative paths (sequence of pathlib.Path)
            .c : source-condition to analysis-condition mapping (dict)

    Output
        pre : input SimpleNamespace with the new attributes:
            .cat : generic stimulus catalogue (pandas.DataFrame)
            .des : participant/block target configuration database (dict)

        Writes normalized PNG assets, stimuli/stimuli.csv, catalogue notes,
        and design.json. Target-detection labels remain block-specific.
    """
    # derivative stimulus assets
    # -------------------------------------------------------------------------
    sdir     = pre.b / 'stimuli'                                                # BIDS stimulus assets
    odir     = pre.d / 'stimuli'                                                # derivative stimulus assets
    idir     = odir / 'images'                                                  # normalized PNG directory
    idir.mkdir(parents = True, exist_ok = True)                                 # create stimulus directories
    for file in sdir.iterdir():                                                 # stimulus lists and non-image assets
        dest = odir / file.name                                                 # derivative asset path
        if file.name == 'images':                                               # images converted separately
            continue                                                            # omit source image directory copy
        if file.is_dir():                                                       # non-image asset directory
            shu.copytree(file, dest, dirs_exist_ok = True)                      # copy nested assets
        else:                                                                   # stimulus list or documentation file
            shu.copy2(file, dest)                                               # copy individual asset
    for file in (sdir / 'images').iterdir():                                    # source stimulus images
        if file.suffix.lower() in ('.png', '.jpg', '.jpeg'):                    # supported images
            with Image.open(file) as img:                                       # close source image after conversion
                img.convert('RGB').save(idir / f'{file.stem}.png')              # PNG stimulus

    # task-specific stimulus sets and fixed classification labels
    # -------------------------------------------------------------------------
    keys           = []                                                         # natural image-name sorting keys
    for file in idir.glob('*.png'):                                             # available derivative images
        key        = [                                                          # numeric runs sort by value rather than lexical order
            int(v) if v.isdigit() else v.lower()                                # numeric or text token
            for v in re.split(r'(\d+)', file.stem)]                             # filename tokenization
        keys.append((key, file))                                                # pair sortable key with asset path
    imgs           = [file for key, file in sorted(keys)]                       # naturally ordered images
    lsts           = {}                                                         # source text stimulus lists
    for name in ('verbalKlafi_true', 'verbalKlafi_false', 'verbalGed'):         # stimulus lists
        file       = odir / f'{name}.txt'                                       # source list path
        vals       = file.read_text(encoding = 'utf-8').splitlines()            # source lines
        lsts[name] = [                                                          # retain only stimulus entries
            v.strip() for v in vals                                             # remove surrounding whitespace
            if v.strip() and not v.lstrip().startswith('                        #')]                    # omit comments
    yes         = {'  '.join(v.split()) for v in lsts['verbalKlafi_true']}      # same parity
    no          = {'  '.join(v.split()) for v in lsts['verbalKlafi_false']}     # mixed parity
    dpc         = [tuple(int(x) for x in v.split()) for v in yes | no]          # digit pairs
    dpc         = sorted(dpc)                                                   # natural digit-pair order
    name        = {}                                                            # stable task-specific stimulus names
    name['PTD'] = [                                                             # pattern-target stimuli in natural image order
        f.stem for f in imgs if re.fullmatch(r'[3-6][a-e]_z(?:_2)?', f.stem)]   # PTD
    name['LTD'] = [                                                             # unordered pairs from the source letter pool
        f'{a} {b}' for a, b in comb(lsts['verbalGed'], 2)]                      # LTD
    name['DPC'] = ['  '.join(str(x) for x in pair) for pair in dpc]             # DPC
    name['RMC'] = [                                                             # rotation and mirror source filename convention
        f.stem for f in imgs if re.fullmatch(r'\d+_50_(?:R_)?k', f.stem)]       # RMC
    rule        = {                                                             # binary stimulus-label definitions
        'PTD': 'block_target_membership',                                       # block-specific pattern targets
        'LTD': 'block_target_membership',                                       # block-specific letter-pair targets
        'DPC': 'same_parity',                                                   # fixed digit-parity category
        'RMC': 'rotation_only'}                                                 # fixed object-transformation category
    rows        = []                                                            # generic stimulus catalogue records
    for task, vals in name.items():                                             # catalogue task order
        for n, val in enumerate(vals, start = 1):                               # task-specific identifiers
            file    = f'images/{val}.png' if task in ('PTD', 'RMC') else ''     # asset
            lab     = pd.NA                                                     # target-detection labels depend on the block
            if task == 'DPC':                                                   # fixed digit-parity classification
                lab = int(val in yes)                                           # positive for same parity
            elif task == 'RMC':                                                 # fixed rotation/mirror classification
                lab = int('_R_' in val)                                         # positive for rotation only
            rows.append(                                                        # stimulus identity and label rule
            {                                                                   # record fields
                'task_type'    : task,                                          # component task
                'stimulus_id'  : n,                                             # one-based within-task identifier
                'stimulus_name': val,                                           # concrete stimulus identity
                'stimulus_file': file,                                          # relative image asset or empty text path
                'label_rule'   : rule[task],                                    # label interpretation
                'fixed_label'  : lab})                                          # binary label or block-specific missing
    pre.cat                = pd.DataFrame(rows)                                 # complete generic catalogue
    pre.cat['fixed_label'] = pre.cat['fixed_label'].astype('Int64')             # nullable labels
    file                   = odir / 'stimuli.csv'                               # catalogue output path
    pre.cat.to_csv(file, index = False, na_rep = '', lineterminator = '\n')     # CSV
    text = ( '                                                                  # FCDT stimulus catalogue\n\n`stimuli.csv` maps stable '          # stimulus catalogue documentation
             'task-specific integer identifiers to the concrete\nstimuli '      # stimulus catalogue documentation
             'used in the four component tasks. Image paths are relative to '   # stimulus catalogue documentation
             'this\ndirectory. Text stimuli have an empty `stimulus_file` '     # stimulus catalogue documentation
             'value.\n\nColumn `fixed_label` implements the binary stimulus '   # stimulus catalogue documentation
             'label when it is fixed by\nthe task. Label 1 denotes the '        # stimulus catalogue documentation
             'positive stimulus category and label 0 denotes the\n'             # stimulus catalogue documentation
             'complementary category. The value is empty for PTD and LTD '      # stimulus catalogue documentation
             'because target\nmembership is participant- and '                  # stimulus catalogue documentation
             'block-specific.\n\nThe catalogue is generated by '                # stimulus catalogue documentation
             '`code/abm_preparation.py` from the source stimulus\nmaterial. '   # stimulus catalogue documentation
             'It contains 18 PTD patterns, 190 unordered LTD letter pairs, '    # stimulus catalogue documentation
             '8 DPC\ndigit pairs, and 8 RMC object pairs.\n')                   # stimulus catalogue documentation
    (odir / 'README.md').write_text(text, encoding = 'utf-8')                   # write catalogue notes

    # participant/block target configurations
    # -------------------------------------------------------------------------
    fix  = {}                                                                   # fixed classification-task positive categories
    for task in ('DPC', 'RMC'):                                                 # classification component tasks
        data      = pre.cat[pre.cat['task_type'] == task]                       # task-specific stimuli
        data      = data.sort_values('stimulus_id')                             # stable identifier order
        vals      = data.loc[data['fixed_label'] == 1, 'stimulus_name']         # positive IDs
        fix[task] = vals.tolist()                                               # fixed positive-category stimulus names
    cols     = ['P', 'R', 'B', 'C', 'T']                                        # design-relevant trial fields
    tabs     = []                                                               # participant block configurations
    for file in pre.f:                                                          # only participants generated in this run
        data = pd.read_csv(file, sep = '\t', usecols = cols)                    # participant targets
        tabs.append(data[cols].drop_duplicates())                               # one record per local block
    blocks   = pd.concat(tabs, ignore_index = True)                             # complete block-design table
    blocks   = blocks.sort_values(['P', 'R', 'B']).reset_index(drop = True)     # source order
    part     = {}                                                               # nested participant design database
    for p, data in blocks.groupby('P', sort = True):                            # participant designs
        part[str(int(p))] = {}                                                  # task-pair designs for this participant
        for cond in pre.c.values():                                             # stable condition order
            cdat   = data[data['C'] == cond]                                    # condition-specific local blocks
            if cdat.empty:                                                      # unavailable participant condition
                continue                                                        # omit absent condition
            ta, tb = cond.split('-')                                            # component task names
            runs   = {}                                                         # condition-specific run designs
            for run, rdat in cdat.groupby('R', sort = True):                    # experimental runs
                rdat     = rdat.sort_values('B')                                # source local-block order
                reps     = {}                                                   # first/second repetition designs
                seq      = rdat.itertuples(index = False)                       # local block records
                for n, row in enumerate(seq):                                   # condition repetitions
                    vals = [] if pd.isna(row.T) else str(row.T).split('|')      # targets
                    vals = [v.strip() for v in vals if v.strip()]               # target names
                    reps[str(n)] = {                                            # simulator-ready block configuration
                        'source_block': int(row.B),                             # local source block index
                        'task_1'      : ta,                                     # target-detection component
                        'task_2'      : tb,                                     # classification component
                        'S1T'         : vals,                                   # block-specific memory targets
                        'S2T'         : fix[tb]}                                # fixed classification targets
                runs[str(int(run))] = reps                                      # attach ordered block repetitions
            part[str(int(p))][cond] = runs                                      # attach condition design

    # incomplete run documentation and design output
    # -------------------------------------------------------------------------
    runs          = sorted({str(int(r)) for r in blocks['R'].unique()})         # expected runs
    miss          = []                                                          # incomplete condition/run records
    for p, data in part.items():                                                # participant completeness
        for cond in pre.c.values():                                             # expected task pairs
            for run in runs:                                                    # expected experimental runs
                n = len(data.get(cond, {}).get(run, {}))                        # available repetitions
                if n != 2:                                                      # incomplete repetition count
                    miss.append(                                                # explicit design coverage record
                    {                                                           # record fields
                        'participant'     : int(p),                             # participant index
                        'task_pair'       : cond,                               # analysis condition
                        'run'             : int(run),                           # BIDS run index
                        'available_blocks': n})                                 # observed repetition count
    pre.des = {                                                                 # complete participant/block design metadata
        'description': ( 'Participant-specific FCDT target configurations '     # provenance
                         'derived from Experiment 1 of Brüning et al. (2020).'),# study
        'lookup': ( 'participants[participant][task_pair][run][block], where '  # lookup
                    'block is 1 or 2 in source local-block order.'),            # repetitions
        'incomplete_task_pair_runs': miss,                                      # coverage exceptions
        'participants': part}                                                   # participant-specific designs
    file    = pre.d / 'design.json'                                             # design database output
    text    = json.dumps(pre.des, indent = 2, ensure_ascii = False) + '\n'      # stable JSON
    file.write_text(text, encoding = 'utf-8')                                   # write target configurations
    return pre                                                                  # retain the input namespace and attach catalogue and design


def abm_prepare(pre):                                                           # complete experimental preparation
    """
    Prepare all participants and all four experimental conditions from OSF data.

    Input
        pre : SimpleNamespace() object with required attributes:
            .s : OSF source directory (pathlib.Path)
            .o : experiment output directory (pathlib.Path)

    Output
        pre : input SimpleNamespace with the new attributes:
            .b   : BIDS output directory (pathlib.Path)
            .d   : derivative output directory (pathlib.Path)
            .c   : source-condition to analysis-condition mapping (dict)
            .p   : sorted participant indices (list of integers)
            .n   : number of retained dual-task response rows (integer)
            .f   : written participant derivative paths (list of pathlib.Path)
            .cat : generic stimulus catalogue (pandas.DataFrame)
            .des : participant/block target configuration database (dict)

        Source data are read only. The three stages replace generated output
        files without deleting unrelated directories or other analysis outputs.
    """
    # preparation directories and condition definitions
    # -------------------------------------------------------------------------
    pre.b = pre.o / 'bids'                                                      # response-level output directory
    pre.d = pre.o / 'derivatives'                                               # trial-level output directory
    pre.c = {                                                                   # source condition name and analysis task pair
        'UniR': 'PTD-RMC',                                                      # spatial target detection and spatial classification
        'VkRg': 'PTD-DPC',                                                      # spatial target detection and verbal classification
        'VgRk': 'LTD-RMC',                                                      # verbal target detection and spatial classification
        'UniV': 'LTD-DPC'}                                                      # verbal target detection and verbal classification

    # preparation stages
    # -------------------------------------------------------------------------
    print('Step 1/3: converting OSF responses to BIDS.', flush = True)          # progress
    pre = abm_bids(pre)                                                         # source responses and BIDS metadata
    print('Step 2/3: reconstructing participant trial states.', flush = True)   # progress
    pre = abm_trials(pre)                                                       # trial-level participant tables
    print('Step 3/3: preparing stimuli and block designs.', flush = True)       # progress
    pre = abm_design(pre)                                                       # generic stimuli and participant-specific targets
    print(f'Prepared {len(pre.p)} participants and {pre.n:,} responses.')       # summary
    print(f'BIDS dataset: {pre.b}')                                             # BIDS output location
    print(f'Derivative dataset: {pre.d}')                                       # trial-level output location
    return pre                                                                  # complete preparation state


if __name__ == '__main__':                                                      # command-line execution
    rdir  = Path(__file__).resolve().parent.parent                              # project root directory
    pre   = SimpleNamespace()                                                   # shared preparation state
    pre.s = rdir / 'data' / 'osf'                                               # original experimental source data
    pre.o = rdir / 'data' / 'experiment'                                        # generated experimental datasets
    pre   = abm_prepare(pre)                                                    # execute all three preparation stages
