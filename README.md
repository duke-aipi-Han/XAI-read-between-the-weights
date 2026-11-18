# Read Between the Weights

A proposed XAI experiment comparing predictions of neuron ablation effects from weights alone, fixed probes, and reviewer-chosen probes. The initial target is a 121-parameter synthetic classifier; a pretrained MNIST classifier is a stretch goal. Final deliverables will include a presentation slide deck, a print-ready handout, and a project poster.

This repository currently contains a project skeleton and research plans only. The experiment, CLI, training, evaluation, and reviewer collection are not implemented or run.

## Setup

With Python 3.10 or newer installed, create an isolated environment from the project root and install the planned dependencies and package:

```sh
python3.10 -m venv .venv  # or another installed Python 3.10+ executable
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Dependency versions and runtime have not yet been tested. The local `.env` file is ignored by Git; API integration is optional and has not been implemented. Collect paid responses only through an explicitly user-authorized run.

## Project files

- [Reviewer prompt](prompts/REVIEWER_PROMPT.md): template for future reviewer sessions.
- [experiment.json](experiment.json): proposed defaults, not validated hyperparameters.
- [PROTOCOL.md](PROTOCOL.md): study design, pilot changes, and protocol freeze.
- `src/weight_reader/` and `tests/`: placeholders for implementation and checks.
- `checkpoints/`, `packets/`, `responses/`, and `results/`: placeholders for experiment artifacts; `private/` holds private data and is ignored by Git.

Keep the answer key, private evaluation data, and planning files out of reviewer sessions. Do not modify any parent project's `sources/` directory.
