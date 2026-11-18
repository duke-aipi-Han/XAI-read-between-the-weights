# Repair reviewer prompt template · staged calibration revision

For the operator: render only the text after the delimiter. This is a separate task from ablation prediction. This revision makes adaptive turn boundaries explicit; it has not yet been tested with a reviewer. Do not provide the reference checkpoint, fault construction, private datasets, or intervention results.

---

You are given a small classifier and the complete weight packet below. Its intended task is to output a positive logit for class 1 when x0 and x1 have the same nonzero sign, and a nonpositive logit for class 0 otherwise. In particular, the intended class is 0 if x0 or x1 equals zero. Each input has four coordinates in [-1, 1]. The other two inputs are nuisance variables. Inputs are drawn independently and uniformly from [-1, 1].

The model computes h1 = ReLU(W1 @ x + b1), h2 = ReLU(W2 @ h1 + b2), z = (W3 @ h2 + b3)[0]. Weight matrices use output-by-input orientation. Your goal is to improve accuracy on new inputs while avoiding regressions. The checkpoint may have weaknesses; do not assume how they were caused.

{{CONDITION_INSTRUCTIONS}}

If your condition is adaptive probes, follow this turn protocol exactly:
1. On this turn, output exactly one JSON object containing `inputs` and, optionally, `hypothesis`. The `inputs` array must contain exactly eight four-coordinate inputs. Do not include Markdown fences or other text.
2. Stop and wait for the input/logit results. Do not output later batches or a final edit on this turn.
3. After receiving the results, choose the next batch and again output exactly one JSON object, then stop and wait. Repeat until all four batches have received results.
4. Only after receiving the fourth batch's results, output your final edit proposal as the JSON object specified below. Do not output extra text or Markdown fences.

After any permitted probing, propose exactly one multiplicative edit to a single scalar or a single row or column of one parameter tensor. Use one of these parameter names: fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias. Give an index for each dimension. Use a nonnegative integer for a fixed index and null for the one full dimension in a row or column; at most one null is allowed. For a bias, use one integer index. Specify a finite scale between 0 and 2 inclusive. The edit multiplies the selected original values by the scale. For example, {"parameter":"fc2.weight","index":[null,0],"scale":1.1} scales column 0. This example only shows the format and is not a recommendation.

Return one JSON object with exactly these keys:
- packet_id: copy from the evidence packet.
- hypothesis: briefly state the behavior you intend to change and the evidence supporting the change.
- edit: an object containing exactly parameter, index, and scale.
- expected_effect: describe the expected improvement and possible regression or uncertainty.

Evidence packet:

```json
{{EVIDENCE_PACKET_JSON}}
```
