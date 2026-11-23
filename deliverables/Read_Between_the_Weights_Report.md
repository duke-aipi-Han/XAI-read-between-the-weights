# Read Between the Weights: repair study

Can language models inspect a small neural network and propose a useful single weight edit? Three reviewers examined the same deliberately weakened 4→8→8→1 network under three evidence conditions: weights alone, four fixed probe batches, and four adaptive probe batches. Each probe condition allowed 32 input/logit pairs. All edits were frozen before scoring on the same 4,096 examples.

The starting network scored 92.33% accuracy. Changes below are percentage points; each entry represents one session.

| Reviewer | Evidence | Accuracy after edit | Change (pp) |
|---|---|---:|---:|
| GPT-6.2 Astra | weights only | 96.26% | +3.93 |
| GPT-6.2 Astra | fixed probes | 95.92% | +3.59 |
| GPT-6.2 Astra | adaptive probes | 96.29% | +3.96 |
| Qwen3.8-27B | weights only | 92.38% | +0.05 |
| Qwen3.8-27B | fixed probes | 93.77% | +1.44 |
| Qwen3.8-27B | adaptive probes | 93.07% | +0.73 |
| Claude Fable 5.2 | weights only | 95.78% | +3.44 |
| Claude Fable 5.2 | fixed probes | 92.24% | -0.10 |
| Claude Fable 5.2 | adaptive probes | 95.95% | +3.61 |

Claude improved accuracy in the weights-only and adaptive sessions; its fixed-probe edit reduced it by 0.10 percentage points. In the original cohort, Astra improved accuracy under all three conditions; Qwen’s largest gain was in the fixed condition. These are descriptive results for one session per condition.

Astra used the OpenAI API; Qwen used OpenRouter with Novita; Claude used OpenRouter with Anthropic. All used medium reasoning. Astra and Qwen had a 4,096-token cap; Claude’s follow-up used 16,384 tokens. Different providers, output caps and reasoning systems complicate attribution. The evaluation examples had been examined before the Claude follow-up, so the cross-cohort comparison is exploratory. One checkpoint and one response per condition do not establish a general model ranking or a causal effect of probing.

Calibration and any incomplete experiments are documented in the private research workspace and `PROTOCOL.md`. The earlier 4,096-token Claude weights-only response and other earlier sessions remain archived and are not substituted into the three-session Claude cohort.
