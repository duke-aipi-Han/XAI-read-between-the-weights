# Read Between the Weights

A proposed XAI experiment comparing predictions of neuron ablation effects from weights alone, fixed probes, and reviewer-chosen probes. The initial target is a 121-parameter synthetic classifier; a pretrained MNIST classifier is a stretch goal. Final deliverables will include a presentation slide deck, a print-ready handout, and a project poster.

The synthetic classifier, deterministic datasets, calibration circuit, and training pilot are implemented. Budgeted fixed and adaptive probing and copy-on-write weight edits are available; ablation-review packet export is available; repair proposal import, study freeze, and held-out scoring are available. No reviewer study has been run.

## Setup

With Python 3.10 or newer installed, create an isolated environment from the project root and install the planned dependencies and package:

```sh
python3.10 -m venv .venv  # or another installed Python 3.10+ executable
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

See [PROTOCOL.md](PROTOCOL.md) for the tested versions and training pilot. The local `.env` file is ignored by Git; The optional calibration API relay is described below. Collect paid responses only through an explicitly user-authorized run.

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

## Presentation notebooks

Review the executed notebooks in order:

1. [Model exploration](notebooks/01_model_exploration.ipynb): the task, architecture, training, and controlled weakness.
2. [Probe walkthrough](notebooks/02_probe_walkthrough.ipynb): fixed and adaptive probing.
3. [Export walkthrough](notebooks/03_export_walkthrough.ipynb): the ablation-review packet and prediction task.
4. [Repair walkthrough](notebooks/04_repair_walkthrough.ipynb): a scripted repair proposal and calibration comparison.
5. [Calibration and study](notebooks/05_calibration_and_study.ipynb): the prepared reviewer sessions and the path to held-out scoring.

Open them in VS Code with the project environment selected, or run `python -m jupyter lab notebooks/`. The probe, export, and repair walkthroughs use temporary demonstration sessions. The calibration notebook reads local prepared session records. The notebooks save presentation figures as PNG and SVG files in `results/figures/`. They use validation data, not the final evaluation set. No actual reviewer response is represented.

The notebooks read local checkpoints. If they are missing, run `prepare` and `train` in the project root as described above, then recreate the controlled weakened checkpoint:

```sh
printf '%s\n' '{"parameter":"fc2.weight","index":[null,5],"scale":0.5}' > private/demo_fault.json
python -m weight_reader.cli edit --checkpoint checkpoints/trained_seed_11.pt --edit private/demo_fault.json --output checkpoints/editing_start_seed_11.pt
printf '%s\n' '{"parameter":"fc2.weight","index":[null,0],"scale":0.5}' > private/calibration_fault.json
python -m weight_reader.cli edit --checkpoint checkpoints/calibration.pt --edit private/calibration_fault.json --output checkpoints/repair_calibration.pt
```

This construction is operator-only information and must be withheld from a blind reviewer. Checkpoints and exported figures are ignored by Git under the current repository configuration.

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

The private session metadata and frozen model are under `private/probe_sessions/`; transcripts are under `responses/probes/`. Only relay the JSON printed by `probe` to the reviewer. The project files, fault construction, reference checkpoint, labels, and private data must remain outside a blind session. The bounded relay is a foundation for manual collection; ablation-response import and an ablation-study freeze are not yet implemented.

## Export an ablation-review session

The exporter renders the existing neuron-ablation prediction prompt with the complete weights. For example, to prepare a calibration session without making an API call:

```sh
python -m weight_reader.cli export-review \
  --session-id calibration-reviewer-a --reviewer reviewer-a --phase calibration \
  --condition adaptive_probes --checkpoint checkpoints/calibration.pt
```

The command prints paths to an opaque JSON weight packet and a rendered Markdown prompt in `packets/`. The prompt already embeds the packet: send the rendered prompt in a fresh reviewer conversation and relay only the permitted probe results. For fixed or adaptive conditions, export automatically creates the corresponding probe session; run the `probe` command above with the same session ID. Weights-only sessions have no probe budget. Use a fresh session ID for each reviewer, condition, and checkpoint.

A private record in `private/review_sessions/` ties the session and reviewer to the checkpoint and stores checkpoint, packet, prompt, and template hashes. Keep that record, the reference model, fault construction, task labels, and evaluation data away from the reviewer. The exporter currently supports the original ablation-prediction study; use `--task repair` for the separate weight-editing task described below. Do not collect paid responses without explicit authorization.

## Request and freeze a repair proposal

Two adaptive calibration sessions are prepared locally; their operator instructions are in `private/repair_calibration_checklist.md`. The repair task tells the reviewer the intended XNOR behavior and asks it to propose one multiplicative edit to a single parameter, row, or column, with a scale from 0 to 2. It is separate from the ablation-prediction task, which keeps the target rule private. Create an opaque repair packet and prompt with:

```sh
python -m weight_reader.cli export-review --task repair \
  --session-id repair-calibration-a --reviewer reviewer-a --phase calibration \
  --condition adaptive_probes --checkpoint checkpoints/calibration.pt
```

Relay the four probe rounds for fixed or adaptive sessions, as described above. After the reviewer submits its final JSON, save the exact reply to a file and freeze the proposal:

```sh
python -m weight_reader.cli import-repair --session-id repair-calibration-a --file reply.json
python -m weight_reader.cli validate-repair --session-id repair-calibration-a
```

Import archives the original response even if it is malformed. If necessary, one explicit `--format-repair` retry is permitted; preserve both replies and verify manually that the repair changes formatting only. A valid proposal is applied to a copy of the frozen session checkpoint and recorded with hashes; the reference is not changed. All four probe rounds must be complete before a probe-session proposal can be frozen.

`validate-repair` is restricted to **calibration** sessions and prints before/after accuracy on the development validation split, including quadrant breakdowns. It does not access the final evaluation split. Main-study candidates and the evaluation protocol must be frozen together before final evaluation, as described below. Repeated calibration on validation data can overfit it. No real reviewer response has been collected.

## Plan and evaluate a main repair study

Before creating **main** repair sessions, prepare a JSON array listing every intended session. This illustration declares one reviewer across all three conditions; add the other reviewers and checkpoints required by your study before planning. Session IDs must be unique, and checkpoint paths are read when the plan is created.

```json
[
  {"session_id":"r1-weights", "reviewer":"reviewer-1", "condition":"weights_only", "checkpoint":"checkpoints/editing_start_seed_11.pt"},
  {"session_id":"r1-fixed", "reviewer":"reviewer-1", "condition":"fixed_probes", "checkpoint":"checkpoints/editing_start_seed_11.pt"},
  {"session_id":"r1-adaptive", "reviewer":"reviewer-1", "condition":"adaptive_probes", "checkpoint":"checkpoints/editing_start_seed_11.pt"}
]
```

Save the array as `private/repair_plan_input.json`, then run:

```sh
python -m weight_reader.cli plan-repair-study --plan private/repair_plan_input.json
```

The plan records the config, prompt, private split, and checkpoint hashes. For each planned entry, export with `export-review --task repair --phase main` and the matching session ID, reviewer, condition, and checkpoint. Relay probes and import the proposal as above. Main repair exports that are not in the plan are rejected.

After collection, freeze the entire cohort. If every planned response is frozen, use `python -m weight_reader.cli freeze-repair-study`. Otherwise create a JSON object mapping each missing session ID to a reason and pass `--missing private/missing.json`. Missing and partial sessions remain visible in the results; an already-frozen proposal cannot be marked missing. The freeze verifies the evidence and records existing partial transcripts. No new main repair sessions, proposals, or probe rounds can be added afterward.

Only after the study is frozen, run:

```sh
python -m weight_reader.cli evaluate-repair-study
```

The command checks frozen hashes and the saved final split, then saves overall and quadrant before/after accuracy and accuracy change for every completed session to `results/repair_study.json`. Missing sessions appear with reasons and completed probe counts. Re-running returns the saved result after integrity checks. This output is a descriptive comparison, not a significance test. Use a fresh, predeclared study for further collection; do not tune on held-out results.

## Optional API relay for calibration

The API relay currently supports the two prepared **repair calibration** sessions only. It uses `gpt-6.2-astra` directly through OpenAI and `qwen/qwen3.8-27b` through OpenRouter, pinned to the Novita backend with provider fallback disabled. Both use medium reasoning and a 4,096-token maximum per turn. Novita availability, support for the requested parameters, account access, and the credentials have not been verified through a provider request.

Run this local-only check to see whether the configuration, session artifacts, and `OPENAI_API_KEY` / `OPENROUTER_API_KEY` are present:

```sh
python -m weight_reader.cli remote-preflight
```

After a paid run is explicitly authorized, each invocation below sends **one billable API request** and returns only a journal path and status. Start with one session and invoke its command once per turn: the first call sends the prompt; each of the following four calls relays the prior probe batch's input/logit results. A completed fifth turn writes `final.txt`, whose path is printed, for use with `import-repair`. Do not use the manual probe relay simultaneously for the same session.

```sh
python -m weight_reader.cli remote-step --session-id repair-calibration-astra --execute
python -m weight_reader.cli remote-step --session-id repair-calibration-qwen --execute
```

Requests, full provider responses and any returned usage are stored in `private/remote_calls/`, without API keys. OpenAI uses stored Responses API conversation state; OpenRouter requests replay the prior messages and any returned reasoning details. Settings are pinned across sessions on the first call. Failures, incomplete output, missing token usage or missing OpenRouter cost stop the session and are never automatically retried; inspect the private journal before taking further action.

`remote_config.json` specifies a local USD 10 **reservation ceiling**. Each request reserves a conservative estimated maximum before dispatch; reservations are not released, including after failures. This is not a provider-side spending limit or a guarantee of the final bill: pricing can change, account activity outside this workspace is not counted, and actual billing may differ. Set separate provider-side spending limits and verify current rates before calling. OpenRouter's request also rejects providers above its configured per-token price thresholds. Never put credentials in the tracked configuration.

Main-study API collection is intentionally not implemented by this relay; it still needs integration with the predeclared cohort and freeze controls.
