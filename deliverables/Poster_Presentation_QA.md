# Read Between the Weights: Poster Q&A

A preparation guide for conversations with fellow Duke AIPI students and faculty. The answers describe the **harder-checkpoint repair study on the poster**, unless otherwise stated. They are deliberately short enough to say aloud; the follow-up details help if someone asks for more.

## A 30-second opening

> We tested whether a large language model can inspect and repair a much smaller neural network by editing its weights directly. The small network solves a synthetic sign-matching task, and we deliberately weakened two sets of connections. We compared two LLM reviewers under three evidence conditions: weights only, fixed probes, and adaptive probes. All six proposed edits improved accuracy on the same held-out examples, but some also introduced new errors. This is an exploratory case, not evidence that one model or probing strategy will generally win.

## The problem and the small network

### 1. What is your research question?

Can an LLM use a small network's weights, optionally supplemented with input/output probes, to propose a useful direct weight edit? We also want to understand whether the evidence available to the reviewer is associated with different outcomes.

### 2. What task does the small network solve? Is it XNOR?

Yes, away from zero. The first two inputs determine the class: same sign gives 1 and different signs give 0. Equivalently, it is XNOR of the two sign bits. If either coordinate is exactly zero, the defined label is 0. The other two inputs are nuisance variables.

For example, `(−0.4, −0.7)` has label 1, while `(−0.4, +0.7)` has label 0. The poster's matrix shows the **target labels**, not the network's predictions or accuracies.

### 3. Why use such a simple synthetic task?

We know the correct answer for every input and can inspect the complete network. That makes it possible to identify failures, apply controlled changes, and measure side effects. The trade-off is that success on this toy task does not establish success on a large or naturally trained model.

### 4. What is the network's architecture and how do you get 121 parameters?

It is a fully connected **4 → 8 → 8 → 1** network with ReLU hidden layers and a single output logit. Including biases: `(4 × 8 + 8) + (8 × 8 + 8) + (8 × 1 + 1) = 121` parameters. A positive logit predicts class 1; zero or negative predicts class 0.

### 5. How was the original model trained?

It was trained on synthetic examples with Adam and binary cross-entropy on logits; the training code selects a checkpoint using validation loss. The displayed case starts from the trained seed-22 checkpoint. Training used gradients, but the subsequent reviewer-proposed repair did not retrain the target model.

### 6. Why weaken a model that was already good? How was it weakened?

An intentionally controlled fault creates a measurable repair opportunity while preserving a strong starting model. We multiplied all outgoing weights from first-hidden-layer unit 0 by **0.35** and those from unit 3 by **0.50**. In the second-layer weight matrix these are columns 0 and 3. The orange lines on the poster represent those changed connections; the neurons themselves were not deleted.

The case was chosen after development-set exploration. It is a constructed stress test, not a naturally occurring failure or an unbiased sample of all model weaknesses.

### 7. How good were the original and weakened networks?

On the same **4,096 balanced held-out examples**, the original scored **99.68%** and the weakened network scored **92.33%**. The original score is a reference, not the performance of an LLM-proposed edit. The weakened model scored **78.22%** in the both-negative quadrant, so the overall average hides a concentrated weakness.

## Reviewers and evidence

### 8. What or who is a reviewer?

A reviewer is a large language model given the small model's weights, architecture and task description, and asked to propose one edit. The reviewers were **GPT-6.2 Astra**, accessed via OpenAI's API, and **Qwen3.8-27B**, served through OpenRouter with Novita selected. They are separate from the 121-parameter target network.

### 9. Why compare these two models? Is this a fair comparison?

They give us one proprietary frontier API model and one open-weight model under the same basic repair task and evidence conditions. It is a useful case study, but not a controlled comparison of size alone: their training, architectures, serving providers, and reasoning implementations differ. Astra's parameter count is not publicly disclosed. A medium reasoning setting was requested for both, but that label does not guarantee equal compute.

### 10. What information did the reviewers receive?

Both received the target task, architecture, complete weights and biases, the allowed edit format, and the instruction to improve accuracy while avoiding regressions. Depending on condition, they also received probe inputs and original-network logits. They were not given the operator's fault construction, the untouched reference checkpoint, or the final evaluation examples and labels.

The task rule was **not hidden**: reviewers could infer a probe's correct class from its input. Describing the task as unknown would be inaccurate.

### 11. What is a probe, and what is a logit?

A probe is a test input sent through the unchanged target network; the reviewer sees the input and the resulting **logit**, the model's raw output before a probability transformation. The sign of the logit determines the predicted class. Its magnitude provides information about the model's output, although it should not automatically be treated as a calibrated probability.

### 12. What are the three conditions?

- **Weights only:** no probe queries.
- **Fixed probes:** 32 predetermined inputs, in four batches of eight.
- **Adaptive probes:** the reviewer selects 32 inputs over four rounds and sees earlier responses before selecting later batches.

Each reviewer-condition pair was a fresh session. Probe conditions use the same maximum number of inputs; repeated adaptive inputs count toward that budget. The underlying target checkpoint stays unchanged during probing.

### 13. Were the fixed probes chosen to help or reveal specific errors?

They were designed to provide **balanced coverage**, not recorded as hand-picked examples of errors. A seeded generator produced 32 inputs and assigned signs so each batch had two examples per quadrant; the remaining coordinates varied randomly. Both reviewers received the same fixed set. One fixed probe happened to expose a negative/positive error Qwen discussed.

The checkpoint itself was deliberately selected and weakened after development exploration. The available records do not show that fixed probes were selected by their outcomes, although they cannot rule out every possible informal inspection.

### 14. Did the adaptive reviewer get more information?

It had the same number of probe inputs as the fixed condition but could choose later inputs based on earlier outputs. The actual inputs could differ by reviewer. Fixed batches guaranteed quadrant coverage; adaptive batches did not have that constraint. We did not separately equalize the reviewers' reasoning compute, and we ran only one session per condition.

## Edits and evaluation

### 15. What was the reviewer allowed to edit?

Exactly one multiplicative edit to a single parameter, a row, or a column of a weight matrix, or a single bias. The multiplier had to be between **0 and 2**. A row or column can change multiple numerical weights, so “one edit” does not necessarily mean one scalar weight. Each edit was applied to a copy of the starting checkpoint.

### 16. Is editing just setting weights to zero? What is ablation?

No. An **ablation** temporarily removes or suppresses a component to study its effect; zeroing is a common ablation. Here a reviewer could choose zero, but could also scale a parameter up or down. The poster reports a **repair** study, not results from the project's separate ablation-prediction task.

### 17. Was the model fine-tuned after the proposed edit?

No. We multiplied the selected weight or weights by the proposed factor, without gradient updates or retraining, and evaluated the edited copy. The initial model had been trained normally; the repair step was the direct edit.

### 18. What was Astra's fixed-probe edit?

It multiplied **all outgoing weights from first-layer hidden unit 3 by 1.6**, meaning column 3 of `fc2.weight`. On the held-out set, it corrected 180 previously wrong predictions but introduced 33 new errors. Accuracy in Q-- rose from **78.22% to 95.80%**, while Q-+ fell from **91.21% to 87.99%**.

A higher overall score does not imply that every region improved, nor does the edit fully restore the original network. Because unit 3 had been weakened to 0.5 of its original outgoing weights, multiplying the weakened column by 1.6 makes it 0.8 of that original column, with the other introduced weakness still present.

### 18a. What did each reviewer edit, and did it restore the damaged weights?

The original damage scaled the **outgoing connections** of first-hidden-layer units 0 and 3. Before any reviewer edit, all eight outgoing weights from unit 0 were at **35%** of their original values, and all eight from unit 3 were at **50%**. Each session started from that same weakened model and made its own independent edit.

| Reviewer / test | What was edited | First-hidden-layer units touched | Damaged units touched | Result relative to original weights |
|---|---|---:|---:|---|
| Astra / weights only | One **incoming** weight to unit 3: `fc1.weight[3,1] × 2.00` | 1: unit 3 | 1: unit 3, but its incoming weight was **not** one of the weakened weights | Edited incoming weight becomes **200%** of its original. Unit 3's eight damaged outgoing weights stay at **50%**. |
| Astra / fixed probes | All eight **outgoing** weights from unit 3: `fc2.weight[:,3] × 1.60` | 1: unit 3 | 1: unit 3; all 8 of its damaged outgoing weights | Those eight weights become **80%** of their original values: 50% × 1.60. |
| Astra / adaptive probes | One **incoming** weight to unit 3: `fc1.weight[3,1] × 1.99` | 1: unit 3 | 1: unit 3, but its incoming weight was **not** one of the weakened weights | Edited incoming weight becomes **199%** of its original. Unit 3's eight damaged outgoing weights stay at **50%**. |
| Qwen / weights only | All eight **outgoing** weights from unit 0: `fc2.weight[:,0] × 0.50` | 1: unit 0 | 1: unit 0; all 8 of its damaged outgoing weights | Those eight weights become **17.5%** of their original values: 35% × 0.50. |
| Qwen / fixed probes | One **incoming** weight to unit 2: `fc1.weight[2,1] × 0.80` | 1: unit 2 | **0**: unit 2 was not one of the two units with weakened outgoing weights | Edited incoming weight becomes **80%** of its original. The two damaged outgoing columns stay at **35%** and **50%**. |
| Qwen / adaptive probes | One **outgoing** weight from unit 3 to second-layer unit 5: `fc2.weight[5,3] × 2.00` | 1: unit 3 | 1: unit 3; **1 of its 8** damaged outgoing weights | That one weight returns to **100%** of its original value: 50% × 2.00. Unit 3's other seven outgoing weights stay at **50%**. |

**How to count neurons:** The table counts distinct *first-hidden-layer units associated with the edited connections*. A single incoming-weight edit changes one connection into one first-layer unit. An outgoing-column edit changes eight connections from one first-layer unit to **eight different second-layer units**; the Qwen adaptive scalar affects one second-layer unit. “Damaged units touched” includes an incoming edit to unit 3, but that does **not** mean it restored the damaged outgoing weights. Unit 0's damaged column stays at 35% unless the session edits it; unit 3's stays at 50% unless the session edits it. Percentages are relative to the corresponding signed original weight, not percentages of accuracy.

**Why did Qwen's weights-only edit make an already weakened column smaller?** It hypothesized that reducing unit 0's influence could improve the decision boundary. That further reduced the selected column relative to the original network, yet yielded a small net accuracy gain: 28 errors corrected and 26 introduced. An edit can compensate for behavior without reversing the original fault.

### 19. What do Q++, Q+−, Q−+ and Q−− mean?

The first sign is input `x0`; the second is `x1`. Q-- means both are negative, so the target is 1. Q-+ means `x0` is negative and `x1` is positive, so the target is 0. Each quadrant contributes **1,024** of the 4,096 held-out examples.

### 20. What is the difference between errors corrected, new errors, and overall accuracy change?

We compare predictions on the **same examples** before and after each edit. Corrected errors were wrong before and right after; new errors were right before and wrong after. Net accuracy change in percentage points is `(corrected − new errors) / 4,096 × 100`.

For Astra fixed: `(180 − 33) / 4,096 × 100 = +3.59 percentage points`. The final accuracy is **95.92%**, up from **92.33%**. This is a percentage-point change, not a 3.59% relative improvement.

### 21. Which edit had the highest overall accuracy?

Astra adaptive achieved **96.29%**, a gain of **3.96 percentage points**, on this sample. Astra weights-only achieved **96.26%**, only one example lower. These are descriptive single-run results, not a statistically established general ranking.

### 22. Did fixed probes beat adaptive probes for Qwen? Why might that happen?

In this case Qwen fixed reached **93.77%** versus **93.07%** for adaptive. Fixed corrected 60 errors and introduced one; adaptive corrected 30 and introduced none. The fixed session focused on a negative/positive failure and selected a first-layer scalar edit; the adaptive session focused much of its probing on asymmetric both-negative inputs and selected a second-layer scalar edit. The latter helped that region less on the held-out set.

That is a plausible explanation of the observed difference, not proof of its cause. Fixed probes provided guaranteed quadrant coverage; the adaptive session chose its own distribution and repeated two inputs. One conversation each cannot tell us whether the probe strategy, stochastic variation, or another factor produced the difference.

### 23. Did the LLM really understand the network?

We can say its proposed edit **improved the measured behavior**. Its natural-language explanation is a hypothesis. A good edit alone does not prove that the explanation or the model's internal causal account is correct; that would require additional targeted interventions and comparisons.

### 24. How were development data and the final evaluation separated?

The case and an illustrative operator edit were explored using development validation data. The six reviewer proposals and study configuration were then frozen before the separate final evaluation. The final set was generated from a different seed and balanced across quadrants. The original-network reference score was calculated later on that same final set for the poster; it was not used to choose the six reviewer edits.

### 25. Did the reviewers get to test their edits on the final set?

No. They saw only the evidence permitted by their condition and returned one proposal. The operator froze the cohort, applied each proposal to a separate copy, and then measured all edits on the same held-out examples. They were not given feedback from the final evaluation to revise their edits.

## Interpretation, controls and next steps

### 26. Does this show that probes cause better repairs?

No. In this single case, Qwen's fixed-probe edit did better than its weights-only edit, while Astra's weights-only edit was already strong. Different sessions can produce different outputs, and there is one run per condition. Repeated, independently sampled sessions across multiple checkpoints would be needed to estimate variation and assess whether a probe condition helps.

### 27. Does this prove Astra is better than Qwen?

No. Astra's edits improved accuracy more **on this selected checkpoint, in these sessions**. That is a useful observation, not a general model ranking. Model size, provider, training, prompts, and actual reasoning compute are not controlled to isolate a single cause.

### 28. How would you make the experiment stronger?

Predeclare several independently trained checkpoints and several types of faults; repeat each reviewer-condition pairing with fresh sessions; report the distribution of changes and uncertainty across sessions and checkpoints. Include paired subgroup metrics and regressions, not just overall accuracy. Keep selection and evaluation separate and record failures as well as successes.

### 29. What baselines would you add?

Useful comparisons include no edit, random valid edits, simple weight-magnitude or gradient-informed rules, and a small validation-budget search over the same allowed edits. A human or scripted operator using matched evidence and query budgets could also help. Define the resources available to each baseline so the comparison is interpretable. These additional baselines have **not** been established by the poster results.

### 30. Why not simply use the original checkpoint, retrain, or fine-tune?

If the original checkpoint is available, restoring it is a natural practical baseline. Training or fine-tuning may also be a more effective route when data and compute are available. The research question here is narrower: whether a targeted edit proposed from model evidence can change behavior without a new training run. We did not show that this approach is cheaper, safer, or better than fine-tuning.

### 31. Can the finding generalize to real neural networks or LLMs?

We do not know. This network is tiny, the rule is known, and the weaknesses were introduced deliberately. Larger models have many interacting parameters and less transparent objectives. A next step would be multiple learned checkpoints with different faults, followed by a carefully defined larger task such as the project's MNIST stretch goal.

### 32. Could this become recursive self-improvement (RSI)?

Direct weight editing could be part of a future loop in which a model helps identify a weakness, propose a change to its own weights, and test whether to keep it. This project did **not** demonstrate RSI: the reviewers edited a separate small network once each. Iterated improvement is not guaranteed; errors and regressions can accumulate.

A recent research preprint reports a model proposing edits to its own weights in a limited setting: [Fan et al., *Language Models Can Rewrite Their Own Weights*](https://arxiv.org/abs/2611.21885). Treat it as early research, not proof of open-ended improvement.

### 33. How much did the experiment cost?

For the harder-checkpoint main study, OpenRouter reported approximately **$0.37 for 11 Qwen calls**. The shared local ledger showed about **$3.25 committed out of a $10 ceiling** after the run, including earlier calls, conservative OpenAI usage-based estimates, and an unresolved reservation. Those numbers have different scopes; the ledger is not a provider invoice, and the Qwen figure is not the combined study cost.

### 34. Can someone reproduce or audit the results?

The code, configuration, prompt, protocol, notebooks, and final presentation materials are in the project. To reduce repository contents, raw responses, probe transcripts, generated results, and checkpoints are ignored by Git; a fresh clone therefore does **not** contain everything needed to rerun or audit the exact historical sessions. The local workspace retains those artifacts. A new run would incur API costs and need its own records.

### 35. What is the honest take-home message?

A language model can propose a direct edit that improves this deliberately weakened small network, and probe access may change what it discovers. The same edit can also introduce new mistakes. The evidence is one controlled case with one session per condition, so it motivates a broader experiment rather than settling whether probes or a particular reviewer are generally better.

## Quick numbers to remember

| Item | Value |
|---|---:|
| Small network | 4 → 8 → 8 → 1; 121 parameters |
| Original / weakened accuracy | 99.68% / 92.33% |
| Held-out examples | 4,096; 1,024 per quadrant |
| Reviewers / conditions | 2 / 3; one session per combination |
| Probe budget | 32 inputs, in four batches of eight |
| Best overall result in this case | Astra adaptive: 96.29%, +3.96 percentage points |
| Astra fixed: corrected / new | 180 / 33 |
| Qwen fixed: corrected / new | 60 / 1 |
| Qwen adaptive: corrected / new | 30 / 0 |

Project references: [Poster PDF](Read_Between_the_Weights_Poster_36x24.pdf), [harder-case notebook](../notebooks/11_harder_repair_results.ipynb), [protocol](../PROTOCOL.md), and [repair prompt](../prompts/REPAIR_PROMPT_V3.md). The local `results/repair_v2_report.md` and raw session records are ignored by Git and may be absent from a fresh clone.
