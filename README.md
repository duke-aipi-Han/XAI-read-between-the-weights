# Read Between the Weights

A proposed XAI experiment comparing predictions of neuron ablation effects from weights alone, fixed probes, and reviewer-chosen probes. The initial target is a 121-parameter synthetic classifier; a pretrained MNIST classifier is a stretch goal. Final deliverables will include a presentation slide deck, a print-ready handout, and a project poster.

The synthetic classifier, deterministic datasets, calibration circuit, and training pilot are implemented. Budgeted fixed and adaptive probing and copy-on-write weight edits are available; reviewer packet export, response collection, and held-out evaluation are later phases. No reviewer study has been run.

## Setup

With Python 3.10 or newer installed, create an isolated environment from the project root and install the planned dependencies and package:

```sh
python3.10 -m venv .venv  # or another installed Python 3.10+ executable
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

See [PROTOCOL.md](PROTOCOL.md) for the tested versions and training pilot. The local `.env` file is ignored by Git; API integration is optional and has not been implemented. Collect paid responses only through an explicitly user-authorized run.

## Project files

- [Reviewer prompt](prompts/REVIEWER_PROMPT.md): template for future reviewer sessions.
- [experiment.json](experiment.json): proposed defaults, not validated hyperparameters.
- [PROTOCOL.md](PROTOCOL.md): study design, pilot changes, and protocol freeze.
- `src/weight_reader/` and `tests/`: synthetic model, data, training, and foundation checks; later modules remain placeholders.
- `checkpoints/`, `packets/`, `responses/`, and `results/`: placeholders for experiment artifacts; `private/` holds private data and is ignored by Git.

Keep the answer key, private evaluation data, and planning files out of reviewer sessions. Do not modify any parent project's `sources/` directory.

## Synthetic model and pilot

The CPU classifier is a 4→8→8→1 ReLU network with 121 parameters. Its target is whether the first two of four independently sampled coordinates have the same sign: an XNOR of their sign bits away from zero. If either coordinate is exactly zero, the label is 0. The hand-set calibration network implements the rule exactly away from the axes. The model exposes hidden activations and first-layer zero-ablation; the tests show that zeroing a second-layer weight column has the equivalent effect for the corresponding first-layer unit. This provides a basis for later probing and weight edits, without claiming any improvement from an edit.

To prepare and train from a fresh output directory, after installing the package:

```sh
python -m weight_reader.cli prepare --config experiment.json --output-dir ./pilot
python -m weight_reader.cli train --config experiment.json --output-dir ./pilot
python -m unittest discover -s tests -v
```

Omit `--output-dir` to use the project root. Existing datasets and checkpoints are not overwritten. Keep the generated `private/` data out of reviewer sessions, including when using a separate output directory. Training uses only training and validation splits; the final evaluation split is not used for model selection.

## Exploratory weight-editing task

`checkpoints/editing_start_seed_11.pt` is an intentionally degraded copy of the trained seed 11 checkpoint. It reaches 90.72% validation accuracy, with a concentrated failure in Q++. The unchanged trained checkpoint is a reference; the starting point is a controlled fault, not a naturally occurring training failure. The fault and reference must be withheld from a reviewer in a blind experiment. See [PROTOCOL.md](PROTOCOL.md) for construction details and limits.

Use a JSON array of four-coordinate inputs to inspect logits, then propose an edit as JSON. Inputs must be finite and lie in `[-1, 1]`. For example, save `[[0.5, 0.5, 0, 0], [0.5, -0.5, 0, 0]]` as `inputs.json`, then run:

```sh
python -m weight_reader.cli inspect --checkpoint checkpoints/editing_start_seed_11.pt --inputs inputs.json
```

An edit specifies a parameter name, an index per dimension (`null` selects an entire dimension), and a finite multiplier. For example, `{"parameter": "fc2.weight", "index": [null, 0], "scale": 1.1}` scales one downstream weight column. Save the proposal as `edit.json`, then run:

```sh
python -m weight_reader.cli edit --checkpoint checkpoints/editing_start_seed_11.pt --edit edit.json --output checkpoints/candidate.pt
python -m weight_reader.cli validate --checkpoint checkpoints/candidate.pt
```

The example is a format illustration, not a recommended edit. Edits create new checkpoints and cannot overwrite existing ones. Validation reports overall and quadrant accuracy; repeated validation queries can overfit this development set. Freeze candidate edits and the evaluation procedure before using the private final evaluation set. The exploratory `inspect` command is unbudgeted; use the session workflow below when comparing probe conditions.

## Exploration notebook

Open [the model exploration notebook](notebooks/model_exploration.ipynb) in VS Code with the project environment selected, or install the requirements and run `python -m jupyter lab notebooks/model_exploration.ipynb`. The saved notebook includes executed outputs. The [probe walkthrough](notebooks/probe_walkthrough.ipynb) illustrates fixed and adaptive probing with temporary demonstration sessions. Running it again uses the saved checkpoints and pilot metrics and exports presentation figures as PNG and SVG files in `results/figures/`. It uses the validation split, not the final evaluation set.

## Budgeted manual probing

Create a separate session for each reviewer, condition, and checkpoint. Each session snapshots the selected checkpoint and its SHA-256 hash. The default budget in `experiment.json` is four rounds of eight inputs. Fixed rounds use a reproducible input set with two examples per quadrant per round; adaptive rounds accept reviewer-chosen inputs. Both return only input/logit pairs.

```sh
# Fixed probes: four calls, changing --round from 1 to 4.
python -m weight_reader.cli start-probe-session --session-id fixed-example --condition fixed_probes --checkpoint checkpoints/editing_start_seed_11.pt
python -m weight_reader.cli probe --session-id fixed-example --round 1

# Adaptive probes: provide one JSON file per round.
python -m weight_reader.cli start-probe-session --session-id adaptive-example --condition adaptive_probes --checkpoint checkpoints/editing_start_seed_11.pt
python -m weight_reader.cli probe --session-id adaptive-example --round 1 --inputs batch1.json
```

An adaptive batch file contains `{"inputs": [[...], ...], "hypothesis": "optional explanation"}`: exactly eight arrays of four finite coordinates in `[-1, 1]`. Repeat inputs count toward the budget. Rounds must be completed in order. Replaying a logged round with the same inputs returns its saved logits; a different input for an existing round is rejected. Fixed sessions do not accept input files.

The private session metadata and frozen model are under `private/probe_sessions/`; transcripts are under `responses/probes/`. Only relay the JSON printed by `probe` to the reviewer. The project files, fault construction, reference checkpoint, labels, and private data must remain outside a blind session. The bounded relay is a foundation for manual collection; packet export, reviewer response import, formal experiment freeze, and edit evaluation are not yet implemented.
