# Repair prompt v3 · condition-specific staged relay

Operator: only text after the delimiter is sent. RELAY_PROTOCOL_STAGED_V3.

---

You are reviewing a small neural network. Its job is to predict 1 when x0 and x1 have the same nonzero sign, and 0 otherwise (including when either is exactly zero). Inputs have four coordinates in [-1, 1], independently uniformly distributed. Coordinates x2 and x3 are nuisance variables.

The network computes h1 = ReLU(W1 @ x + b1), h2 = ReLU(W2 @ h1 + b2), z = (W3 @ h2 + b3)[0]. Predict 1 when z > 0, otherwise 0. Matrices use output-by-input orientation. Your goal is to improve accuracy with one weight edit while avoiding regressions.

{{CONDITION_INSTRUCTIONS}}

Final edit format (use only when the instructions say it is time to submit): return a single JSON object with exactly `packet_id`, `hypothesis`, `edit`, and `expected_effect`. The edit has exactly `parameter`, `index`, and `scale`. Choose one of fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias. Use one index per dimension: an integer for a fixed index, or null for one entire row or column (at most one null). Biases require one integer index. The finite scale must be between 0 and 2 inclusive. The selected values are multiplied by this scale. Example format only: {"parameter":"fc2.weight","index":[null,0],"scale":1.1}. Copy `packet_id` from the packet; briefly describe your hypothesis and expected effect.

Evidence packet:

```json
{{EVIDENCE_PACKET_JSON}}
```


For each requested probe round, return exactly one JSON object containing exactly one eight-input batch. Do not provide additional candidate batches in the same message. Wait for the logits before choosing the next batch.
