"""Report the frozen paid retry alongside the original v2 cohort."""
import csv, json
from pathlib import Path
root=Path('.'); retry=root/'private/studies/repair_retry'
old=json.loads((root/'results/repair_v2_study.json').read_text())
new=json.loads((retry/'results/repair_study.json').read_text())
assert old['sample_count']==new['sample_count']==4096
names={'astra':'GPT-6.2 Astra','qwen':'Qwen3.8-27B','fable':'Claude Fable 5.2 (retry)','neon':'Gemini 4 Neon (retry)'}
rows=[]
single=root/'private/studies/repair_neon_single'
single_file=single/'results/repair_study.json'
extra=json.loads(single_file.read_text()) if single_file.exists() else None
unfrozen_single=False
if extra is None and (single/'private/repair_study/plan.json').exists():
    plan=json.loads((single/'private/repair_study/plan.json').read_text())
    statuses=[]
    for entry in plan['sessions']:
        issue=single/'private/review_sessions'/entry['session_id']/'session_issue.json'
        if (single/'private/review_sessions'/entry['session_id']/'repair_frozen.json').exists():
            raise ValueError('Do not report individual scores until the cohort is frozen')
        reason=json.loads(issue.read_text())['reason'] if issue.exists() else 'Not collected'
        statuses.append(dict(entry,status='incomplete',reason=reason))
    extra={'sessions':statuses}
    unfrozen_single=True
records=old['sessions']+([r for r in new['sessions'] if 'fable' in r['session_id']]+extra['sessions'] if extra else new['sessions'])
for record in records:
    key='neon' if record['session_id'].startswith('neon-single-') else record['session_id'].split('-')[1]
    cohort=single if record['session_id'].startswith('neon-single-') else (retry if key in ('fable','neon') else root/'private/studies/repair_v2')
    probe=cohort/'responses/probes'/record['session_id']
    rounds=len(list(probe.glob('round-*.json'))) if probe.exists() else 0
    rows.append({'reviewer':names[key],'condition':record['condition'],'status':record['status'],'probe_rounds':rounds,'accuracy_before':record.get('before',{}).get('accuracy'),'accuracy_after':record.get('after',{}).get('accuracy'),'change_pp':100*record['accuracy_change'] if 'accuracy_change' in record else None,'reason':record.get('reason',''),'session_id':record['session_id']})
out=root/'deliverables'; out.mkdir(exist_ok=True)
with (out/'Read_Between_the_Weights_Retry_Results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'Read_Between_the_Weights_Retry_Results.json').write_text(json.dumps({'sample_count':4096,'status':'exploratory','sessions':rows},indent=2)+'\n')
lines=['# Read Between the Weights: additional reviewer retry','','An exploratory comparison on the previously examined 4,096-example evaluation set and the same weakened seed-22 checkpoint. Changes are percentage points of accuracy from the 92.33% starting accuracy. Each cell is one independent session; missing sessions are not zero.','','| Reviewer | Condition | Status | Probe rounds | Accuracy after | Change (pp) |','|---|---|---|---:|---:|---:|']
for r in rows:
    a=f"{100*r['accuracy_after']:.2f}%" if r['accuracy_after'] is not None else '—'; d=f"{r['change_pp']:+.2f}" if r['change_pp'] is not None else '—'
    lines.append(f"| {r['reviewer']} | {r['condition'].replace('_',' ')} | {r['status']} | {r['probe_rounds']} | {a} | {d} |")
lines += ['', ('Fable retry: Anthropic via OpenRouter, medium reasoning, 16,384 output tokens. Gemini retry: paid Google Vertex Priority via OpenRouter, medium reasoning, 16,384 output tokens, with a revised single-batch instruction.' if extra else 'Fable retry: Anthropic via OpenRouter, medium reasoning, 16,384 output tokens. Gemini retry: Google Vertex Priority via OpenRouter, medium reasoning, 4,096 output tokens. A subsequent higher-cap calibration did not yield a usable main cohort.'), 'Earlier Astra and Qwen sessions had a 4,096-token cap and different providers; results can reflect deployment, prompt and reasoning differences as well as model differences.', '', 'An earlier Fable weights-only session under the 4,096-token cap achieved 96.00% accuracy (+3.66 pp); its probe sessions were incomplete. Earlier Gemini calibration attempts failed with HTTP 429. Those records remain in the previous frozen cohort and are not substituted into this one.', '', '## Missing and incomplete sessions', '']
missing=[r for r in rows if r['status']!='frozen']
if not missing:lines.append('None.')
for r in missing:lines.append(f"- {r['session_id']}: {r['reason']}")
if unfrozen_single:
    lines += ['', 'The revised-prompt Gemini cohort has not been frozen or evaluated because none of its main sessions returned a valid edit. Its weights-only and fixed sessions exhausted the 16,384-token cap; the adaptive session could not be started under the shared reservation ceiling. The calibration completed successfully.']
lines+=['','Different caps and providers, a single checkpoint, one response per condition, and reuse of an examined held-out set prevent broad model rankings or causal claims about probing. Original raw responses and any format-only repairs remain in the private workspace.']
(out/'Read_Between_the_Weights_Retry_Report.md').write_text('\n'.join(lines)+'\n')
print('Wrote exploratory comparison report and data.')
