# Optimal dual-task performance

This repository contains a starting implementation of the rational analysis
approach developed within the DFG project
[Individual determinants of response-strategy formation in voluntary task
switching](https://gepris.dfg.de/gepris/projekt/569073723?language=en)
(project 569073723).

The project investigates why individuals adopt different response strategies
when coordinating two concurrent tasks. The present implementation focuses on
the theoretical and computational part of that programme. It represents the
free concurrent dual-tasking paradigm as a finite-horizon Markov decision
process, derives optimal response policies by backward induction, and embeds
these policies in an agentic behavioral model. This provides a basis for
relating observed response sequences to individual differences in subjective
task-switching and task-imbalance costs.

The implementation is an initial model of response choices. It does not yet
cover the full project scope, including response timing, EEG, reward
manipulations, personality, or cognitive-ability measures.

## Repository contents

- `code/abm/` contains the task, agent, dynamic-programming, simulation, and
  estimation components.
- `code/abm_preparation.py` converts the original experimental data and
  stimuli into BIDS and analysis-ready derivatives.
- `code/abm_validation.py` validates task-state statistics using simulations.
- `code/abm_evaluation.py` fits and compares the agent models for individual
  participants.
- `code/abm_description.py` computes descriptive statistics.
- `code/viz/` contains the scripts used to generate the report figures.
- `data/osf/` contains the original behavioral data and stimulus material.
- `data/experiment/` contains the prepared BIDS dataset and derivatives.
- `data/validation/` contains the simulation-based validation summary.
- `figures/` contains the generated figures.
- `report/` contains the Quarto source, bibliography, and rendered report.
- `presentation/` contains project presentation material.

## Installation

The analysis uses Python 3.11. Install the Python dependencies from the
repository root:

```shell
python -m pip install -r requirements.txt
```

Rendering the report additionally requires Quarto and a LaTeX installation.

## Running the analysis

The main processing order is:

1. `python code/abm_preparation.py`
2. `python code/abm_validation.py`
3. `python code/abm_evaluation.py`
4. `python code/abm_description.py`
5. Run the scripts in `code/viz/` to regenerate individual figures.
6. Render `report/Ostwald-et-al.qmd` with Quarto to build the report.

## License

The repository is distributed under the GNU General Public License, version 3.
