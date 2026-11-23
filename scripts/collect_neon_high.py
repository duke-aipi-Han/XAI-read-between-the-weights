"""Run the predeclared repair replication; stop each model if its pilot fails."""
import json, sys
from pathlib import Path
from weight_reader.remote_neon_high import remote_step
from weight_reader.repair import import_repair, validate_repair
root=Path('private/studies/repair_neon_high'); cfg=json.loads(Path('remote_config.json').read_text()); exp=json.loads(Path('experiments/repair_v2.json').read_text())
plan=json.loads((root/'private/repair_study/plan.json').read_text())
def collect(sid,condition):
    folder=root/'private/remote_calls'/sid
    wanted=1 if condition=='weights_only' else 5
    for turn in range(1,wanted+1):
        path=folder/f'turn-{turn:02d}.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row.get('status')!='completed': raise ValueError('existing unresolved turn; no retry')
            continue
        remote_step(root,sid,'.env',cfg)
        print(f'{sid}: turn {turn}/{wanted} complete',flush=True)
    if (root/'private/review_sessions'/sid/'repair_frozen.json').exists():
        return
    try:
        import_repair(root,sid,folder/'final.txt')
    except ValueError:
        import re
        text=(folder/'final.txt').read_text()
        match=re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```\s*', text, re.IGNORECASE)
        if not match: raise
        cleaned=root/'private/review_sessions'/sid/'format_only.txt'
        cleaned.write_text(match.group(1)+'\n')
        import_repair(root,sid,cleaned,format_repair=True)
    print(f'{sid}: proposal validated and frozen',flush=True)
def issue(sid,exc):
    path=root/'private/review_sessions'/sid/'session_issue.json'
    if not path.exists(): path.write_text(json.dumps({'status':'incomplete','reason':str(exc),'exception_type':type(exc).__name__},indent=2)+'\n')
    print(f'{sid}: incomplete: {exc}',flush=True)
for short in (sys.argv[1:] or ['neon']):
    pilot='neon-high-'+short+'-cal'
    try:
        collect(pilot,'adaptive_probes')
        result=validate_repair(root,pilot,exp)
        (root/'private'/f'{short}_calibration_result.json').write_text(json.dumps(result,indent=2)+'\n')
    except Exception as exc:
        issue(pilot,exc)
        for e in plan['sessions']:
            if e['session_id'].startswith('neon-high-'+short+'-'): issue(e['session_id'],ValueError('calibration incomplete'))
        continue
    for e in plan['sessions']:
        if e['session_id'].startswith('neon-high-'+short+'-'):
            try: collect(e['session_id'],e['condition'])
            except Exception as exc: issue(e['session_id'],exc)
print('Collection complete; cohort remains unfrozen.',flush=True)
