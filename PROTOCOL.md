# Experiment protocol

## Synthetic classifier and pilot

The study compares weights-only, fixed-probe, and adaptive-probe reviewer predictions against held-out neuron ablations. Proposed defaults are in [experiment.json](experiment.json). The 4→8→8→1 network takes four independent Uniform(-1, 1) inputs and predicts whether the first two have the same sign (XNOR of their sign bits away from zero; any exact-zero coordinate yields label 0). Weights use output-by-input orientation; zero logits predict class 0. The hand-set calibration circuit implements the rule off the axes, with four unused first-layer units.

The November 18, 2026 local CPU pilot used the proposed shared train and validation splits and all three predeclared initialization seeds. Checkpoints were selected by minimum validation loss; the final evaluation split was not used for training or selection. Training used one CPU thread. Times below are measured training loop times, not end-to-end runtime.

| Seed | Best epoch | Epochs run | Validation loss | Validation accuracy | Training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 338 | 388 | 0.004984 | 99.80% | 2.06 s |
| 22 | 270 | 320 | 0.004306 | 99.90% | 1.70 s |
| 33 | 195 | 245 | 0.006739 | 99.90% | 1.29 s |

The 95% validation target was met in this pilot; no hyperparameter adjustment was made. These are validation results, not reviewer or held-out intervention results. Training details are recorded in `results/training_pilot.json`. Tested environment: Python 3.13.14, PyTorch 2.15.0, NumPy 2.5.5, pandas 3.1.1, matplotlib 3.11.2, and SciPy 1.18.1. No paid API calls or reviewer sessions have been made.

Fixed-probe samples cycle quadrants so each batch of eight contains two examples per quadrant. Record further pilot observations, protocol changes, and freeze details here as the work proceeds. Final deliverables include a presentation slide deck, a print-ready handout, and a project poster.

## Controlled editing pilot

To make an interpretable repair possible, the starting editing checkpoint is a deliberately damaged copy of the trained seed 11 model: multiply column 5 of `fc2.weight` by 0.5. On the validation split, its accuracy is 90.72% overall and 63.67% in Q++; the original model achieves 99.80% overall and 100% in Q++. Multiplying that column by 2 restores the original checkpoint and its validation accuracy. This is a controlled sanity check, not evidence that a reviewer identified the issue or that arbitrary edits generalize. The starting checkpoint can be regenerated using the `edit` command and the stated multiplier; the original remains unchanged.

This adds an exploratory model-repair task to the original ablation-prediction study. Future reviewer tests should keep the construction, original checkpoint, and validation labels private, predefine what probes and feedback are permitted, record edit proposals, and assess the frozen candidate once on untouched final evaluation data. Do not present the sanity-check repair as a blinded reviewer outcome or use the final split for iterative tuning.

## Manual probe relay

The probe relay snapshots each session's checkpoint and hash, enforces the configured four rounds of eight inputs, records inputs and original-model logits, and replays saved rounds idempotently. Fixed batches contain two inputs per quadrant and are shared across checkpoints; adaptive sessions accept reviewer-selected batches. Duplicate inputs count toward the budget. These facilities have been tested with synthetic examples only; no reviewer responses have been collected. For editing experiments, predefine the information and feedback given to the reviewer and keep fault construction and the reference model private.

## Reviewer packet export

The manual exporter renders the ablation-prediction prompt for weights-only, fixed-probe, and adaptive-probe conditions. It assigns an opaque packet ID, includes full output-by-input weights, and records checkpoint, packet, prompt, and template hashes in a private session mapping. Probe sessions use the same frozen checkpoint as the packet. This is preparation only: the editing task requires its own reviewer instructions and evaluation procedure before collection. No real reviewer responses or API calls have been made.

## Separate weight-repair task

The repair prompt gives the reviewer the intended XNOR task and same evidence-condition budgets, then requests a single scalar, matrix row, or matrix column multiplied by a finite factor in [0, 2]. The intended target and permitted edits are explicitly specified; the reference checkpoint and constructed fault are withheld. The original ablation-prediction task remains separate and does not reveal the target.

The operator can archive a reply, freeze one validated proposal and its copied checkpoint, and compare a calibration proposal with its starting model on validation accuracy and quadrant accuracy. The original checkpoint is unchanged. One explicitly designated format-only retry is allowed after an invalid response, with both forms preserved; the operator must check that scientific content was not altered. Final evaluation has not been implemented or run. Before main evaluation, predeclare and freeze the full set of sessions, responses, split, and comparison procedure to prevent repeated held-out tuning.
