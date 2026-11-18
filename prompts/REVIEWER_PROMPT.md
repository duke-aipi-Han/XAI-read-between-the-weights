# Reviewer prompt template

**For the experiment operator.** Render the text after the delimiter with the exported weight packet and one of the following instructions. Send it in a fresh reviewer conversation. Supply probe results only as specified by the chosen condition.

| Condition | Text for `{{CONDITION_INSTRUCTIONS}}` |
| --- | --- |
| Weights only | Submit your final prediction now using the supplied weights and architecture. |
| Fixed probes | You will receive four batches of eight fixed input/logit pairs. Reply `READY` to receive each batch. After the fourth batch, submit your final prediction. |
| Adaptive probes | Choose four batches of eight inputs. For each batch, reply with a JSON object with an `inputs` array. Each input contains four finite values in [-1, 1]. You may add a short `hypothesis` string describing what the batch tests. Use the logits you receive to choose your next batch. After the fourth batch, submit your final prediction. |

---

Interpret the small neural network in the evidence packet. Use the supplied weights, the evidence provided under your condition, and your reasoning. Probe feedback, where available, consists of the unmodified network's logits. Follow your condition instructions and submit the final response as one JSON object.

{{CONDITION_INSTRUCTIONS}}

The network outputs a real-valued logit z, with positive z predicting class 1 and nonpositive z predicting class 0. Its computation is:

```text
h1 = ReLU(W1 @ x + b1)
h2 = ReLU(W2 @ h1 + b2)
z  = (W3 @ h2 + b3)[0]
```

Weights use output-by-input orientation. Interpret all eight neurons in h1; IDs are integers 0 through 7. Inputs x0 through x3 are independent Uniform(-1,1). Four evaluation quadrants refer to signs of x0 and x1, in this exact order:

```text
Q++: x0 > 0, x1 > 0
Q+-: x0 > 0, x1 < 0
Q-+: x0 < 0, x1 > 0
Q--: x0 < 0, x1 < 0
```

Within each quadrant, x0 and x1 remain uniform over their respective half-intervals, and x2 and x3 remain independent Uniform(-1,1). Predict averages over these distributions.

For each first-layer neuron, explain its activation pattern and hypothesize its downstream role. A neuron may have a mixed role or no material effect. Predict its activation probability in each quadrant. Then predict the mean logit change in each quadrant when its activation is set to zero after the first ReLU, leaving other first-layer activations unchanged and recomputing the entire remainder of the network. Define every change as **ablated logit minus original logit**. Use signed real numbers in logit units.

Rank all eight neurons from greatest to least expected absolute per-example logit change under this ablation, over the full input distribution. Also select exactly two neurons that you hypothesize form an informative functional group, describe that hypothesis, and predict their simultaneous zero-ablation effect in each quadrant. Do not assume joint effects add linearly.

Keep explanations brief and distinguish inference from certainty. Do not assume the model fits its training task perfectly. Provide your best numerical estimates even when uncertain, and explain uncertainty in the text fields. Include a short overall-function hypothesis, but do not invent a task name as fact.

Output schema:

- `packet_id`: copy from the evidence packet.
- `function_hypothesis`: a short string.
- `neurons`: exactly eight objects, one per neuron, with:
  - `neuron_id`: integer 0–7.
  - `activation_description`: short string explaining input sensitivity.
  - `downstream_role`: short string explaining how it contributes to z.
  - `activation_probability_by_quadrant`: four numbers in [0,1], in the stated order.
  - `ablation_delta_logit_by_quadrant`: four finite signed numbers in the stated order.
  - `uncertainty`: short string.
- `importance_ranking`: all integers 0–7 exactly once, greatest effect first.
- `group`: object with:
  - `neuron_ids`: exactly two distinct integers 0–7.
  - `hypothesis`: short string identifying the expected role and conditions where it applies.
  - `joint_ablation_delta_logit_by_quadrant`: four finite signed numbers in the stated order.
  - `uncertainty`: short string.

Evidence packet:

```json
{{EVIDENCE_PACKET_JSON}}
```
