"""Prepare a separate blinded repair study without model calls or held-out scoring.

Run from the project root with PYTHONPATH=src. Reviewers receive only rendered
packets, never this operator script, source checkpoint, or private workspace.
"""
import json
import os
from pathlib import Path

import torch

from weight_reader.data import make_splits
from weight_reader.editing import edit_weights, validation_metrics
from weight_reader.evidence import export_review
from weight_reader.model import WeightReaderMLP
from weight_reader.repair_study import plan_study
from weight_reader.remote_budget import active_commitment

root=Path('.').resolve()
study=root/'private/studies/repair_v2'
source=root/'checkpoints/trained_seed_22.pt'
template=root/'prompts/REPAIR_PROMPT_V3.md'
config=json.loads((root/'experiments/repair_v2.json').read_text())
remote_config=json.loads((root/'remote_config.json').read_text())
existing_budget=root/'private/remote_calls'
if study.exists():
    raise SystemExit(f'Refusing to overwrite existing study: {study}')
if not source.is_file() or not (existing_budget/'ledger.json').is_file():
    raise SystemExit('The trained seed 22 checkpoint and shared remote ledger are required.')
# Verify the original budget policy and config before creating any new study.
active_commitment(root,remote_config)
torch.set_num_threads(1)
model=WeightReaderMLP(); model.load_state_dict(torch.load(source,map_location='cpu',weights_only=True)); model.eval()
weakened=edit_weights(edit_weights(model,'fc2.weight',[None,3],.5),'fc2.weight',[None,0],.35).eval()
# A development-only reference edit demonstrates that the restricted action space
# contains a useful partial repair. It is not included in reviewer packets.
partial=edit_weights(weakened,'fc2.weight',[None,3],1.5).eval()
splits=make_splits(config)
before=validation_metrics(weakened,splits['validation'])
after=validation_metrics(partial,splits['validation'])
reference=validation_metrics(model,splits['validation'])
if not (.85 < before['accuracy'] < .96 and after['accuracy']-before['accuracy'] > .02):
    raise SystemExit('Development checks failed; no study has been created.')
(study/'private').mkdir(parents=True)
(study/'checkpoints').mkdir()
# Both studies use the very same ledger, lock, and active $10 policy.
remote_link=study/'private/remote_calls'
relative_target=os.path.relpath(existing_budget,remote_link.parent)
(study/'private/shared_remote_budget.json').write_text(json.dumps({'relative_target':relative_target},indent=2)+'\n')
remote_link.symlink_to(relative_target,target_is_directory=True)
if remote_link.resolve()!=existing_budget.resolve():
    raise SystemExit('Shared budget link does not resolve correctly.')
torch.save(splits,study/'private/splits.pt')
checkpoint=study/'checkpoints/editing_start_seed_22_v2.pt'
torch.save(weakened.state_dict(),checkpoint)
entries=[]
for key,reviewer in [('astra','gpt-6.2-astra'),('qwen','Qwen/Qwen3.8-27B')]:
    for name,condition in [('weights','weights_only'),('fixed','fixed_probes'),('adaptive','adaptive_probes')]:
        entries.append({'session_id':f'v2-{key}-{name}','reviewer':reviewer,'condition':condition,'checkpoint':str(checkpoint)})
(study/'private/plan_input.json').write_text(json.dumps(entries,indent=2)+'\n')
plan_study(study,entries,config,template)
for entry in entries:
    export_review(study,entry['session_id'],checkpoint,entry['condition'],entry['reviewer'],'main',config,template,task='repair')
summary={'study':'repair_v2','phase':'prepared_not_run','split':'development_validation',
         'validation_count':before['count'],'starting_accuracy':before['accuracy'],
         'demonstrated_single_edit_accuracy':after['accuracy'],'reference_accuracy':reference['accuracy'],
         'starting_quadrants':before['quadrants'], 'demonstrated_single_edit_quadrants':after['quadrants'],
         'note':'An operator-selected development check, not a reviewer result or held-out score.'}
(root/'results').mkdir(exist_ok=True)
output=root/'results/repair_v2_setup.json'
with output.open('x') as f: json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps({'study_dir':str(study),'sessions':len(entries),'shared_budget':str(remote_link.resolve()),
                  'development_before':before['accuracy'],'development_after':after['accuracy'],
                  'network_calls':0},indent=2))
