"""Predeclare a replication of repair v2 for two additional reviewers; no API calls."""
import json, os, shutil
from pathlib import Path
from weight_reader.evidence import export_review
from weight_reader.repair_study import plan_study
from weight_reader.remote_budget import active_commitment

root=Path('.').resolve()
study=root/'private/studies/repair_neon_high'
prior=root/'private/studies/repair_v2'
config=json.loads((root/'experiments/repair_v2.json').read_text())
remote=json.loads((root/'remote_config.json').read_text())
template=root/'prompts/REPAIR_PROMPT_V3.md'
checkpoint=prior/'checkpoints/editing_start_seed_22_v2.pt'
if study.exists(): raise SystemExit('Study already exists; refusing overwrite')
active_commitment(root,remote)
(study/'private').mkdir(parents=True)
(study/'checkpoints').mkdir()
shutil.copyfile(prior/'private/splits.pt',study/'private/splits.pt')
shutil.copyfile(checkpoint,study/'checkpoints/starting.pt')
link=study/'private/remote_calls'; target=os.path.relpath(root/'private/remote_calls',link.parent)
(study/'private/shared_remote_budget.json').write_text(json.dumps({'relative_target':target},indent=2)+'\n')
link.symlink_to(target,target_is_directory=True)
models=[('neon','google/gemini-4-neon')]
for short,reviewer in models:
    export_review(study,'neon-high-'+short+'-cal',root/'checkpoints/repair_calibration.pt','adaptive_probes',reviewer,'calibration',config,template,task='repair')
entries=[{'session_id':f'neon-high-{short}-{name}','reviewer':reviewer,'condition':condition,'checkpoint':str(study/'checkpoints/starting.pt')} for short,reviewer in models for name,condition in [('weights','weights_only'),('fixed','fixed_probes'),('adaptive','adaptive_probes')]]
(study/'private/plan_input.json').write_text(json.dumps(entries,indent=2)+'\n')
plan_study(study,entries,config,template)
for e in entries: export_review(study,e['session_id'],e['checkpoint'],e['condition'],e['reviewer'],'main',config,template,task='repair')
print('Prepared 1 adaptive calibration session and 3 main sessions; no calls made.')
