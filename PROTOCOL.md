# Experiment protocol

## Synthetic classifier and pilot

The study compares weights-only, fixed-probe, and adaptive-probe reviewer predictions against held-out neuron ablations. Proposed defaults are in [experiment.json](experiment.json). The 4→8→8→1 network takes four independent Uniform(-1, 1) inputs and predicts whether the first two have the same sign (XNOR of their sign bits away from zero; any exact-zero coordinate yields label 0). Weights use output-by-input orientation; zero logits predict class 0. The hand-set calibration circuit implements the rule off the axes, with four unused first-layer units.

The November 18, 2026 local CPU pilot used the proposed shared train and validation splits and all three predeclared initialization seeds. Checkpoints were selected by minimum validation loss; the final evaluation split was not used for training or selection. Training used one CPU thread. Times below are measured training loop times, not end-to-end runtime.

| Seed | Best epoch | Epochs run | Validation loss | Validation accuracy | Training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 338 | 388 | 0.004984 | 99.80% | 2.06 s |
| 22 | 270 | 320 | 0.004306 | 99.90% | 1.70 s |
| 33 | 195 | 245 | 0.006739 | 99.90% | 1.29 s |

The 95% validation target was met in this pilot; no hyperparameter adjustment was made. These are validation results, not reviewer or held-out intervention results. Training details are recorded in `results/training_pilot.json`. Tested environment: Python 3.13.14, PyTorch 2.15.0, NumPy 2.5.5, pandas 3.1.1, matplotlib 3.11.2, and SciPy 1.18.1. At the time of the training pilot, no reviewer calls had been made.

Fixed-probe samples cycle quadrants so each batch of eight contains two examples per quadrant. Record further pilot observations, protocol changes, and freeze details here as the work proceeds. Final deliverables include a presentation slide deck, a print-ready handout, and a project poster.

## Controlled editing pilot

To make an interpretable repair possible, the starting editing checkpoint is a deliberately damaged copy of the trained seed 11 model: multiply column 5 of `fc2.weight` by 0.5. On the validation split, its accuracy is 90.72% overall and 63.67% in Q++; the original model achieves 99.80% overall and 100% in Q++. Multiplying that column by 2 restores the original checkpoint and its validation accuracy. This is a controlled sanity check, not evidence that a reviewer identified the issue or that arbitrary edits generalize. The starting checkpoint can be regenerated using the `edit` command and the stated multiplier; the original remains unchanged.

This adds an exploratory model-repair task to the original ablation-prediction study. Future reviewer tests should keep the construction, original checkpoint, and validation labels private, predefine what probes and feedback are permitted, record edit proposals, and assess the frozen candidate once on untouched final evaluation data. Do not present the sanity-check repair as a blinded reviewer outcome or use the final split for iterative tuning.

## Manual probe relay

The probe relay snapshots each session's checkpoint and hash, enforces the configured four rounds of eight inputs, records inputs and original-model logits, and replays saved rounds idempotently. Fixed batches contain two inputs per quadrant and are shared across checkpoints; adaptive sessions accept reviewer-selected batches. Duplicate inputs count toward the budget. The initial offline checks used synthetic examples; the first live calibration is documented below. For editing experiments, predefine the information and feedback given to the reviewer and keep fault construction and the reference model private.

## Reviewer packet export

The manual exporter renders the ablation-prediction prompt for weights-only, fixed-probe, and adaptive-probe conditions. It assigns an opaque packet ID, includes full output-by-input weights, and records checkpoint, packet, prompt, and template hashes in a private session mapping. Probe sessions use the same frozen checkpoint as the packet. This exporter addresses the original ablation task; the separate repair workflow and its first calibration are documented below.

## Separate weight-repair task

The repair prompt gives the reviewer the intended XNOR task and same evidence-condition budgets, then requests a single scalar, matrix row, or matrix column multiplied by a finite factor in [0, 2]. The intended target and permitted edits are explicitly specified; the reference checkpoint and constructed fault are withheld. The original ablation-prediction task remains separate and does not reveal the target.

The operator can archive a reply, freeze one validated proposal and its copied checkpoint, and compare a calibration proposal with its starting model on validation accuracy and quadrant accuracy. The original checkpoint is unchanged. One explicitly designated format-only retry is allowed after an invalid response, with both forms preserved; the operator must check that scientific content was not altered. Final repair evaluation has been implemented but not run. Predeclare the full set of sessions and freeze responses, split, and comparison procedure before main evaluation to prevent repeated held-out tuning.

## Repair cohort freeze and held-out scoring

Before main collection, predeclare the reviewer, evidence condition, and checkpoint for every session. The plan records hashes of the checkpoint, prompt, full configuration, and prepared private data splits. Freeze records each completed proposal and its packet, prompt, response, checkpoint, and probe transcript hashes. Missing or partial sessions must be explicitly identified with reasons and remain in the result table; already-frozen proposals cannot be excluded as missing. Probe transcripts of completed sessions are checked against the checkpoint and planned input budget.

Once frozen, no new main repair sessions, proposals, or probe rounds can be added through the workflow. Held-out scoring verifies those hashes and computes paired before/after overall and quadrant accuracy for the frozen cohort, retaining missing rows. The report is descriptive and does not establish statistical significance. No final repair evaluation has been run. This workflow is separate from the original ablation-prediction evaluation, which remains unimplemented.

## Prepared repair calibration

Two adaptive repair calibration sessions have been prepared locally, one for each planned reviewer deployment. Both use the same deliberately weakened copy of the handcrafted circuit, four rounds of eight probes, and the same repair prompt. This is distinct from the learned seed 11 checkpoint intended for the initial main comparison.

The controlled calibration fault halves column 0 of the handcrafted network's `fc2.weight`. On the validation split, the weakened circuit has 91.50% accuracy overall and 66.02% in Q++; the original handcrafted circuit has 100% accuracy on that split. These are operator-side validation diagnostics, not reviewer observations or held-out results. The fault, original checkpoint, and validation outcomes should not be sent to a reviewer. The local private checklist identifies the two prompt files and relay commands. This section described the pre-run state; the first live calibration is documented below. Record prompt changes before creating the main repair study plan.

## Provider preflight — November 18, 2026

The intended API deployments are OpenAI `gpt-6.2-astra` and OpenRouter `qwen/qwen3.8-27b`, subject to account access and a calibration pilot. The prepared adaptive repair calibration prompts use identical content apart from opaque packet IDs and refer to the same checkpoint; no real probe rounds or responses had been collected at preflight. An offline rehearsal using temporary synthetic responses completed four probe rounds, response import, and validation comparison for each session; all 17 local tests passed. The private final evaluation split was not scored.

The repository currently provides a **manual** relay, not an API client. It does not automatically load `.env`, preserve complete API conversations and returned provider metadata, pin an OpenRouter backend, enforce a spending cap, or resume provider requests. Those gaps must be addressed before automated API collection. The proposed USD 10 cap in `experiment.json` is an intention, not an enforced limit. OpenRouter may route to different backends unless provider routing is restricted; choose and record one backend and its supported reasoning configuration, and disable cross-provider fallback for comparable conditions. Preserve full conversation continuity, including provider-required reasoning metadata, and record requested and returned model, serving provider, settings, output cap, timestamps, per-call usage and cost, errors, and raw responses. Do not log credentials.

Credential names were checked locally without transmitting them: `OPENROUTER_API_KEY` was present and nonempty; `OPENAI_API_KEY` was not found. Presence does not establish validity, model access, account funding, or available credit. No authenticated provider check or paid request was made.

## Calibration API relay — local implementation

A one-turn-at-a-time API relay is now implemented for the prepared repair calibration sessions. OpenAI uses `gpt-6.2-astra` through the Responses API; OpenRouter uses `qwen/qwen3.8-27b` with `novita` as the sole permitted backend, no fallback, parameter support required, and a per-token price ceiling. Both request medium reasoning and at most 4,096 output tokens per turn. Actual backend availability, key validity, account balance, model access, and parameter support remain untested until an authorized live calibration.

The relay stores private request and full response journals, preserves OpenAI response continuity and OpenRouter reasoning details, and reserves a conservative local cost estimate before each request. It never retries a failed or ambiguous request automatically. The USD 10 local reservation ceiling is not a provider-side billing limit and does not account for other account use or future price changes. Configure independent provider spending controls. The relay is limited to calibration; automated main-study collection has not been implemented. Both credential names and session artifacts passed local preflight; the live run is documented below.

## First live repair calibration — November 18, 2026

OpenAI `gpt-6.2-astra` completed four adaptive rounds and returned a valid final edit. Its frozen proposal multiplies `fc2.weight[0,0]` by 2, reversing the known handcrafted calibration defect. On 1,024 development-validation examples, accuracy rose from 91.50% to 100.00% (+8.50 percentage points); Q++ rose from 66.02% to 100.00%. Other quadrants were already 100% on this validation sample. This is one controlled calibration case, not a held-out or main-study finding, and does not establish that probes caused the successful proposal.

OpenRouter served `qwen/qwen3.8-27b` through Novita. Its first response proposed four probe batches and a final edit before receiving any probe results, so zero probe rounds were delivered and the session is recorded as incomplete rather than a valid adaptive result. The raw response is preserved. OpenRouter reported 7,628 completion tokens for a request specifying `max_tokens=4096`; the reason for the discrepancy is unverified. The remote relay was updated to stop on any future reported output count above the requested cap; the Qwen session must not be automatically retried. A separately planned calibration revision is needed before comparing the reviewers.

The local reservation ledger totals $8.265983 of the $10 ceiling; this is not actual spend or a provider-side billing guarantee. OpenRouter reported $0.02356356 for Qwen. OpenAI's five Astra calls cost an estimated $0.076895 using returned usage/cache counts and published standard rates; actual billing should be checked with the provider. No main repair cohort has been planned or scored, and the final evaluation split remains untouched. Derived results are in `results/repair_calibration_results.json`; raw responses and API journals are preserved separately.

## Qwen calibration investigation

The first Qwen/Novita reply ended with `finish_reason=stop`, not a length-truncation reason. The API reported 7,628 completion tokens, including 5,997 reasoning tokens, despite the request specifying `max_tokens=4096`. The output therefore also contained 1,631 non-reasoning completion tokens according to the returned counters. OpenRouter documents that this limit generally covers reasoning and visible output together, but notes provider variation; the cause of the larger reported number is not established. The response was complete but violated the adaptive protocol by proposing all four batches and an edit before any feedback. More tokens would not by itself correct that turn-taking failure.

Prepared `prompts/REPAIR_PROMPT_V2.md` as an untested calibration revision that explicitly requires one JSON batch per turn, stopping for feedback, and submitting the final edit only after the fourth result. The existing calibration transcripts and prompts remain unchanged. The existing output-token cap and the guard against reported overruns remain unchanged. A future separately recorded calibration can test the revision, preserving provider settings and budget accounting; prompt changes must be applied consistently to the main comparison. No additional model request was made during this investigation.

## Revised staged Qwen calibration — November 18, 2026

The user authorized a fresh Qwen/Novita calibration with `prompts/REPAIR_PROMPT_V2.md`, which explicitly requires one probe batch per turn and waiting for each result. The new session completed four rounds (32 inputs), then returned a valid edit multiplying `fc2.weight[0,0]` by 2. This restored the controlled calibration reference: on the same 1,024 development-validation examples, overall accuracy rose from 91.50% to 100.00%, and Q++ accuracy rose from 66.02% to 100.00%. The original premature session remains incomplete and is not overwritten or scored as adaptive.

On the revised session's first turn, OpenRouter reported 4,832 completion tokens for `max_tokens=4096`, at a reported $0.01524108 against a $0.08145 local reservation. The user had explicitly accepted token overruns. The already-returned response contained one valid batch and was accepted without retry through a separately hashed resolution record; no original journal was modified. All later turns completed normally. Total reported cost for the five revised Qwen calls was $0.0449448. The cumulative local reservation ledger was $8.70395 of $10; this remains an estimate rather than a provider-side billing guarantee.

Qwen's edit is correct on this controlled case, but its explanation is not entirely correct: it stated zero-coordinate cases were unchanged. For x=(0.5,0,0,0), the logit changes from -1.25 to 0.0, although both remain class 0. This illustrates why the edit's numerical and qualitative claims should be checked separately. The staged Qwen prompt differed from Astra's original calibration prompt; these are calibration observations, not a matched main-study comparison. No final evaluation set was used.


## Main repair collection — November 18, 2026

A six-session repair cohort was predeclared on the learned seed 11 checkpoint with the staged prompt, crossing Astra and Qwen with weights-only, fixed probes, and adaptive probes. This is a repair experiment, separate from the original ablation-prediction task. Astra's weights-only proposal was collected and frozen. Qwen's weights-only response requested probes instead of returning the required edit and is preserved as incomplete; it was not converted into a proposal. The four probe-condition sessions have not started. No cohort freeze or held-out evaluation has been performed. Notebook 8 reports collection status and will show outcomes once the frozen cohort is evaluated.

Qwen's response reported 16,584 completion tokens despite a 4,096-token request, with reported cost $0.05124636 under its $0.08228 reservation. In accordance with the user's acceptance of overruns, the already-returned response was reconciled without a retry; its scientific protocol violation remains recorded. The retained-reservation ledger totals $9.55039 of the $10 local ceiling.

A disabled-by-default alternative budget policy has been prepared for review. It would settle completed calls using conservative returned-usage bounds, including all reported overrun tokens, and retain the full reservation for ambiguous or pending calls. The original ledger and $10 ceiling remain unchanged. The current preview is a proposal, not an activation; explicit user approval is required because the prior policy retains reservations after completion. Neither accounting approach guarantees provider-side billing.


## Budget policy update — November 18, 2026

The user authorized settlement of completed calls using actual provider-reported dollar costs where available. When a completed response lacks a dollar cost, as with these OpenAI responses, the tracker uses reported token counts at conservative uncached reservation rates; this remains an estimate. Unresolved or pending calls retain their full reservations. The original reservation ledger, audit journals, and $10 local ceiling remain in place. Before further collection the adjusted committed amount was $0.497539, including estimates; this is not a provider invoice or a provider-side spending limit.


## First learned-checkpoint repair results — November 18, 2026

The six planned sessions ended with four valid frozen proposals and two incomplete sessions. Qwen weights-only requested probes instead of an edit; Qwen fixed-probe first acknowledged READY but, after receiving its first fixed batch, requested its own adaptive batch instead of acknowledging READY. Neither response was converted into an edit. Astra completed all three conditions and Qwen completed all four adaptive probe batches and returned a valid edit. A fixed-condition overrun validator was corrected to accept an already-returned READY acknowledgement; that response was reconciled without a repeated API call. Subsequent Qwen turns also reported output tokens over the requested cap; accepted returned responses were recorded individually and were not retried.

After freezing the cohort, the same 4,096 held-out examples were evaluated once. The starting checkpoint achieved 91.85% accuracy. Astra independently chose the same edit in all three conditions: multiply column 5 of `fc2.weight` by 2. Each resulted in 99.76% accuracy (+7.91 percentage points). In Q++, accuracy increased from 68.16% to 99.80%; the other quadrants were unchanged. Qwen's adaptive proposal zeroed column 3 of `fc1.weight` and left overall accuracy at 91.85% on this sample; small quadrant changes offset one another. Missing Qwen conditions cannot be scored as zero. This is one checkpoint with one conversation per condition, and the same Astra edit across conditions does not establish that probes caused the improvement.

At collection end, local committed cost was $1.659285 of $10: $0.452822 in rounded-up provider-reported costs, $1.112282 in conservative OpenAI usage-based estimates, and $0.094181 retained for an unresolved response. This includes earlier calibration calls and is not an invoice; OpenRouter's reported total and the OpenAI estimate have different bases. The original gross reservation ledger is preserved for audit, and provider billing can differ. Results are stored in `results/repair_study.json` and explained in notebook 8.


## Second repair case and staged protocol — November 18, 2026

A second isolated six-session repair study has been prepared using a different trained checkpoint (seed 22), two controlled parameter changes, and fresh independently seeded validation, fixed-probe and final-evaluation splits. The operator explored pairs of changes on development data and selected a case with substantial residual weakness after one permitted edit. On 2,048 development-validation inputs, the starting model achieved 92.04% accuracy and one operator-selected single edit achieved 94.63%. These numbers were used to select and illustrate the case; they are optimistic development diagnostics, not reviewer outcomes or held-out results. No claim that the case is harder for either reviewer is established by this selection.

The v3 repair prompt supplies separate turn rules for weights-only, fixed, and adaptive evidence. The v3 relay sends batch numbers and condition-specific next-action reminders with probe results; both reviewers use the same prompt and rules per condition. Mocked conversations completed all three conditions with four batches of eight probes where applicable. The six main sessions are predeclared and exported in a separate workspace. No new live requests or held-out evaluation were made. If a pilot motivates another prompt revision, create a new study plan rather than altering this one after collection begins.

The separate workspace uses a required link to the original remote-call ledger and lock so the existing $10 ceiling covers both studies; remote calls fail closed if the link or ledger is missing. Prompt changes and the new checkpoint mean the second study is a new case, not an identical replication of the first.


## Qwen v3 protocol pilot — November 18, 2026

The user authorized a three-condition pilot of the v3 instructions on the existing weakened handcrafted calibration circuit. Three fresh Qwen/Novita conversations used the same settings, prompt template, and checkpoint. The predeclared sessions completed: weights-only returned a valid edit in one call, while fixed and adaptive each completed four eight-input batches and then returned valid edits in five calls. The earlier fixed-probe protocol failure did not recur in this pilot. All three selected the same scalar edit and increased accuracy on the 2,048-example development split from 91.99% to 100.00%; these are calibration observations, not results for the new learned checkpoint or evidence that probing caused the edit.

OpenRouter reported $0.10205618 for the 11 pilot calls. Two responses exceeded the requested 4,096 output-token cap (4,118 in weights-only and 5,262 in adaptive); both reported costs within their reservations and were accepted under the previously authorized overrun policy without retries. The shared local ledger totals $1.761347 committed of $10, including estimated and retained amounts from earlier work; it is not a provider invoice. Responses and probe transcripts are archived, and notebook 10 summarizes the test. No harder-checkpoint model calls or held-out scoring occurred.


## Second learned-checkpoint repair results — November 18, 2026

Astra completed the revised v3 adaptive calibration on the handcrafted circuit, including all four batches and a valid final edit, raising development accuracy from 91.99% to 100%. The Qwen v3 pilot was completed earlier. The six predeclared main sessions were then collected using the same seed 22 starting checkpoint and revised condition-specific prompt. All three sessions for each reviewer returned valid edits. Qwen weights-only and adaptive final outputs were wrapped in Markdown fences; the original bytes were archived, and removal of only the outer fences was recorded as one format-only repair per session. There were no content edits or automatic retries.

The entire cohort was frozen before evaluating 4,096 fresh balanced held-out inputs. Starting accuracy was 92.33%. Astra's weights-only, fixed, and adaptive edits produced 96.26%, 95.92%, and 96.29% accuracy; Qwen's produced 92.38%, 93.77%, and 93.07%, respectively. These are one-checkpoint, one-session-per-condition descriptive outcomes. In particular, Astra's weights-only and adaptive results differ by one example, and the Qwen fixed-probe result exceeds its weights-only and adaptive results in this case; neither observation establishes causality or a general ranking.

Paired held-out counts show the trade-offs: Astra fixed corrected 180 errors but introduced 33, compared with 165 corrections and four regressions for Astra weights-only. Qwen fixed corrected 60 and introduced one, compared with 28 corrections and 26 regressions for Qwen weights-only. The complete paired counts and quadrant results are archived in `results/repair_v2_error_transitions.json` and `results/repair_v2_study.json`; notebook 11 and `results/repair_v2_report.md` explain the results.

OpenRouter reported $0.36818246 for the 11 Qwen main calls. Seven responses reported completion-token counts above the requested 4,096 cap; all were accepted within their original local reservations without retry. The overall shared local ledger was $3.245196 committed of $10, including prior calls, conservative OpenAI token-cost estimates and an unresolved earlier reservation; this is not a provider invoice.

## Astra from-scratch construction follow-up — November 19, 2026

In one separate, user-authorized session, GPT-6.2 Astra received the task definition and the required 4→8→8→1 ReLU architecture, and was asked to return all 121 weights and biases. It was not given an existing target checkpoint, probe results, or held-out examples, and no tools were configured. This was a new construction task, not an edit or a repetition of the prior repair conditions. The complete prompt is in `prompts/FROM_SCRATCH.md`; the request, raw response and frozen numerical network are preserved in ignored local files. The response was validated and frozen before this follow-up evaluation.

The constructed network scored **4,096/4,096 (100.00%)** on the same balanced held-out examples used for the prior repair study, with 1,024/1,024 in each quadrant. On that sample, the original trained model scored 99.68%, the best earlier repair (Astra adaptive) 96.29%, and the weakened model 92.33%. Reuse of an already analyzed held-out set makes this an exploratory comparison, not a fresh confirmatory test. It is one attempt on a simple, explicitly described task; construction and restricted repair are different problems, so the score does not establish a general advantage for either method.

The returned network implements a version of `abs(x0 + x1) - abs(x0 - x1)` and ignores the nuisance inputs; the previously hand-designed calibration circuit implements the same type of sign-matching construction. Manual spot checks at the axes returned a zero logit, matching the specified class-0 rule. This limited check is not an exhaustive numerical robustness test. Evaluation and reproducible reconstruction are in `scripts/from_scratch.py`; the local generated comparison is `results/astra_from_scratch.json`, which is ignored by Git.

The single API request reported 340 input and 441 output tokens. At the existing conservative reservation-policy rates the ledger settles this call at $0.03326; this is an estimate, not a provider invoice. The shared ledger afterward reported $3.278456 committed of the $10 local ceiling, including prior calls and an unresolved earlier reservation. No automatic retries were made.

## Qwen from-scratch construction follow-up

Qwen3.8-27B, served through OpenRouter/Novita, received the identical construction prompt in one independent conversation with medium reasoning requested, no supplied target weights, probes, or held-out examples, and no configured tools. The already-returned response reported 8,697 output tokens against a requested 4,096 cap and $0.02624514 in provider cost, within its $0.079462 reservation. Under the previously accepted overrun policy it was reconciled with a separate hashed resolution and no retry. The single output bias was returned as a scalar; it was wrapped in a one-element array as a documented format-only normalization, with no value changed. The original response and frozen network remain in ignored local files.

After freezing, the resulting network scored **4,096/4,096 (100.00%)** on the same previously analyzed balanced held-out set, including 1,024/1,024 in each quadrant. It uses a different logit calculation from Astra: `ReLU(min(x0, x1)) + ReLU(min(-x0, -x1))`; opposite-sign examples have logit zero, which predicts class 0 under the stated threshold. Five axis spot checks also returned the expected class; this is not an exhaustive numerical robustness test. The comparison remains exploratory, with one attempt per model on a simple known task and a reused test set. Notebook 12 displays both networks and the local generated results are ignored by Git.

The shared ledger afterward reported $3.304702 committed of the $10 local ceiling, including prior calls, conservative OpenAI estimates, the reported Qwen cost, and an unresolved earlier reservation; it is not a provider invoice. `experiments/from_scratch_models.json` registers reviewer models for the reusable runner while preserving provider constraints and reservation ceilings.
