"""Build public-facing three-reviewer tables from frozen repair-study cohorts."""
import csv, hashlib, json
from pathlib import Path
from weight_reader.repair_study import evaluate_study
root=Path('.'); earlier=root/'private/studies/repair_v2'; latest=root/'private/studies/repair_retry'
old=evaluate_study(earlier); new=evaluate_study(latest)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(earlier/'private/splits.pt')==sha(latest/'private/splits.pt')
assert sha(earlier/'checkpoints/editing_start_seed_22_v2.pt')==sha(latest/'checkpoints/starting.pt')
assert old['sample_count']==new['sample_count']==4096
names={'astra':'GPT-6.2 Astra','qwen':'Qwen3.8-27B','fable':'Claude Fable 5.2'}
rows=[]
for s in old['sessions']+new['sessions']:
 key=s['session_id'].split('-')[1]
 if key not in names: continue
 if s['status']!='frozen': raise ValueError('A planned three-reviewer result is not frozen')
 rows.append({'reviewer':names[key],'condition':s['condition'],'before_accuracy':s['before']['accuracy'],'after_accuracy':s['after']['accuracy'],'change_pp':s['accuracy_change']*100,'session_id':s['session_id']})
assert len(rows)==9 and len({(r['reviewer'],r['condition']) for r in rows})==9
out=root/'deliverables';out.mkdir(exist_ok=True)
with (out/'Read_Between_the_Weights_Results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(out/'Read_Between_the_Weights_Results.json').write_text(json.dumps({'sample_count':4096,'status':'exploratory','sessions':rows},indent=2)+'\n')
lines=['# Read Between the Weights: repair study','','Can language models inspect a small neural network and propose a useful single weight edit? Three reviewers examined the same deliberately weakened 4→8→8→1 network under three evidence conditions: weights alone, four fixed probe batches, and four adaptive probe batches. Each probe condition allowed 32 input/logit pairs. All edits were frozen before scoring on the same 4,096 examples.','','The starting network scored 92.33% accuracy. Changes below are percentage points; each entry represents one session.','','| Reviewer | Evidence | Accuracy after edit | Change (pp) |','|---|---|---:|---:|']
for s in rows: lines.append(f"| {s['reviewer']} | {s['condition'].replace('_',' ')} | {100*s['after_accuracy']:.2f}% | {s['change_pp']:+.2f} |")
lines+=['','Claude improved accuracy in the weights-only and adaptive sessions; its fixed-probe edit reduced it by 0.10 percentage points. In the original cohort, Astra improved accuracy under all three conditions; Qwen’s largest gain was in the fixed condition. These are descriptive results for one session per condition.','','Astra used the OpenAI API; Qwen used OpenRouter with Novita; Claude used OpenRouter with Anthropic. All used medium reasoning. Astra and Qwen had a 4,096-token cap; Claude’s follow-up used 16,384 tokens. Different providers, output caps and reasoning systems complicate attribution. The evaluation examples had been examined before the Claude follow-up, so the cross-cohort comparison is exploratory. One checkpoint and one response per condition do not establish a general model ranking or a causal effect of probing.','','Calibration and any incomplete experiments are documented in the private research workspace and `PROTOCOL.md`. The earlier 4,096-token Claude weights-only response and other earlier sessions remain archived and are not substituted into the three-session Claude cohort.']
(out/'Read_Between_the_Weights_Report.md').write_text('\n'.join(lines)+'\n')
print('Built 9-row three-reviewer report and tables.')
