"""Publish an exploratory, four-reviewer repair comparison after cohort freeze."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('.'); additional=root/'private/studies/repair_additional'
old=json.loads((root/'results/repair_v2_study.json').read_text())
new=json.loads((additional/'results/repair_study.json').read_text())
assert old['sample_count']==new['sample_count']==4096
mapping={'astra':'GPT-6.2 Astra','qwen':'Qwen3.8-27B','fable':'Claude Fable 5.2','neon':'Gemini 4 Neon'}
rows=[]
for record in old['sessions']+new['sessions']:
    tokens=record['session_id'].split('-'); model=tokens[1]
    cohort = additional if model in ('fable','neon') else root/'private/studies/repair_v2'
    probe_path = cohort/'responses/probes'/record['session_id']
    rounds = len(list(probe_path.glob('round-*.json'))) if probe_path.exists() else 0
    rows.append({'completed_probe_rounds':rounds, 'reviewer':mapping[model], 'condition':record['condition'], 'status':record['status'], 'before':record.get('before',{}).get('accuracy'), 'after':record.get('after',{}).get('accuracy'), 'change_pp':100*record['accuracy_change'] if 'accuracy_change' in record else None, 'reason':record.get('reason',''), 'session_id':record['session_id']})
out=root/'results';out.mkdir(exist_ok=True)
with (out/'four_reviewer_repair.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'four_reviewer_repair.json').write_text(json.dumps({'sample_count':4096,'status':'exploratory','sessions':rows},indent=2)+'\n')
lines=['# Read Between the Weights: four-reviewer repair comparison','','Exploratory replication of the completed repair v2 study on the identical weakened seed-22 network and 4,096 previously examined held-out examples. Each session proposes one constrained weight edit; reported changes are percentage points of accuracy relative to the same 92.33% starting accuracy. A missing response is not scored as zero.','','| Reviewer | Evidence | Status | Probe rounds | Accuracy after edit | Change (percentage points) |','|---|---|---|---:|---:|---:|']
for r in rows:
    acc=f"{100*r['after']:.2f}%" if r['after'] is not None else '—'
    ch=f"{r['change_pp']:+.2f}" if r['change_pp'] is not None else '—'
    lines.append(f"| {r['reviewer']} | {r['condition'].replace('_',' ')} | {r['status']} | {r['completed_probe_rounds']} | {acc} | {ch} |")
lines += ['', 'The new cohort reuses the V3 repair prompt, probe inputs, checkpoint and split from the earlier study. Each probe condition allows four rounds of eight logits. New reviewers used medium reasoning and a 4,096-token cap. Providers, reasoning systems, and sampling can differ; one session per condition cannot establish model rankings or causal effects of probes. The held-out set was previously inspected, so this comparison is exploratory.', '', '## Incomplete sessions', '']
for r in rows:
    if r['status']!='frozen': lines.append(f"- {r['session_id']}: {r['reason']}")
lines+=['','Claude fixed completed four probe rounds; Claude adaptive completed three. Gemini calibration attempts completed four and two probe rounds respectively, but neither returned a valid final edit.', '', 'Reported cost across new provider responses, including returned incomplete responses, was $1.319292; the shared ledger retains larger conservative reservations for unresolved calls. Two HTTP 429 calls returned no cost metadata. Provider-reported charges do not guarantee the final invoice.', '', 'The raw responses, provider journals, budget ledger, hashes and freeze are stored in the local private workspace. Original raw responses and any format-only repairs are preserved.']
(out/'four_reviewer_repair_report.md').write_text('\n'.join(lines)+'\n')
fig,ax=plt.subplots(figsize=(10,5.8)); conditions=['weights_only','fixed_probes','adaptive_probes']; labels=['Weights only','Fixed probes','Adaptive probes'];colors=['#287b8c','#cb653f','#8a77ad','#d49a37']; y=0
for reviewer in mapping.values():
    for condition in conditions:
        r=next(x for x in rows if x['reviewer']==reviewer and x['condition']==condition)
        if r['change_pp'] is not None:
            v=r['change_pp']; ax.barh(y,v,color=colors[list(mapping.values()).index(reviewer)],height=.7); ax.text(v+(0.06 if v>=0 else -0.06),y,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
        else: ax.text(.05,y,'incomplete',va='center',color='#667780',fontsize=9,style='italic')
        y+=1
    y+=.4
ax.set_yticks([i*3.4+j for i in range(4) for j in range(3)],[f'{reviewer} / {label}' for reviewer in mapping.values() for label in labels],fontsize=9)
ax.invert_yaxis();ax.axvline(0,color='#71848a',linewidth=.7);ax.set_xlabel('Change in held-out accuracy (percentage points)');ax.set_title('One proposed edit per session',loc='left',weight='bold');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
for spine in ax.spines.values():spine.set_visible(False)
fig.tight_layout();(out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/four_reviewer_repair.png',dpi=180);plt.close(fig)
print(out/'four_reviewer_repair_report.md')
