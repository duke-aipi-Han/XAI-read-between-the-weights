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

The operator can archive a reply, freeze one validated proposal and its copied checkpoint, and compare a calibration proposal with its starting model on validation accuracy and quadrant accuracy. The original checkpoint is unchanged. One explicitly designated format-only retry is allowed after an invalid response, with both forms preserved; the operator must check that scientific content was not altered. Final repair evaluation has been implemented but not run. Predeclare the full set of sessions and freeze responses, split, and comparison procedure before main evaluation to prevent repeated held-out tuning.

## Repair cohort freeze and held-out scoring

Before main collection, predeclare the reviewer, evidence condition, and checkpoint for every session. The plan records hashes of the checkpoint, prompt, full configuration, and prepared private data splits. Freeze records each completed proposal and its packet, prompt, response, checkpoint, and probe transcript hashes. Missing or partial sessions must be explicitly identified with reasons and remain in the result table; already-frozen proposals cannot be excluded as missing. Probe transcripts of completed sessions are checked against the checkpoint and planned input budget.

Once frozen, no new main repair sessions, proposals, or probe rounds can be added through the workflow. Held-out scoring verifies those hashes and computes paired before/after overall and quadrant accuracy for the frozen cohort, retaining missing rows. The report is descriptive and does not establish statistical significance. No final repair evaluation has been run. This workflow is separate from the original ablation-prediction evaluation, which remains unimplemented.

## Prepared repair calibration

Two adaptive repair calibration sessions have been prepared locally, one for each planned reviewer deployment. Both use the same deliberately weakened copy of the handcrafted circuit, four rounds of eight probes, and the same repair prompt. This is distinct from the learned seed 11 checkpoint intended for the initial main comparison.

The controlled calibration fault halves column 0 of the handcrafted network's `fc2.weight`. On the validation split, the weakened circuit has 91.50% accuracy overall and 66.02% in Q++; the original handcrafted circuit has 100% accuracy on that split. These are operator-side validation diagnostics, not reviewer observations or held-out results. The fault, original checkpoint, and validation outcomes should not be sent to a reviewer. The local private checklist identifies the two prompt files and relay commands. No reviewer responses or paid calls have been made; calibrate each deployment and record any prompt changes before creating the main repair study plan.

## Provider preflight — November 18, 2026

The intended API deployments are OpenAI `gpt-6.2-astra` and OpenRouter `qwen/qwen3.8-27b`, subject to account access and a calibration pilot. The prepared adaptive repair calibration prompts use identical content apart from opaque packet IDs and refer to the same checkpoint; no real probe rounds or responses have been collected. An offline rehearsal using temporary synthetic responses completed four probe rounds, response import, and validation comparison for each session; all 17 local tests passed. The private final evaluation split was not scored.

The repository currently provides a **manual** relay, not an API client. It does not automatically load `.env`, preserve complete API conversations and returned provider metadata, pin an OpenRouter backend, enforce a spending cap, or resume provider requests. Those gaps must be addressed before automated API collection. The proposed USD 10 cap in `experiment.json` is an intention, not an enforced limit. OpenRouter may route to different backends unless provider routing is restricted; choose and record one backend and its supported reasoning configuration, and disable cross-provider fallback for comparable conditions. Preserve full conversation continuity, including provider-required reasoning metadata, and record requested and returned model, serving provider, settings, output cap, timestamps, per-call usage and cost, errors, and raw responses. Do not log credentials.

Credential names were checked locally without transmitting them: `OPENROUTER_API_KEY` was present and nonempty; `OPENAI_API_KEY` was not found. Presence does not establish validity, model access, account funding, or available credit. No authenticated provider check or paid request was made.

## Calibration API relay — local implementation

A one-turn-at-a-time API relay is now implemented for the prepared repair calibration sessions. OpenAI uses `gpt-6.2-astra` through the Responses API; OpenRouter uses `qwen/qwen3.8-27b` with `novita` as the sole permitted backend, no fallback, parameter support required, and a per-token price ceiling. Both request medium reasoning and at most 4,096 output tokens per turn. Actual backend availability, key validity, account balance, model access, and parameter support remain untested until an authorized live calibration.

The relay stores private request and full response journals, preserves OpenAI response continuity and OpenRouter reasoning details, and reserves a conservative local cost estimate before each request. It never retries a failed or ambiguous request automatically. The USD 10 local reservation ceiling is not a provider-side billing limit and does not account for other account use or future price changes. Configure independent provider spending controls. The relay is limited to calibration; automated main-study collection has not been implemented. Both credential names and session artifacts passed local preflight, and no model calls were made.
