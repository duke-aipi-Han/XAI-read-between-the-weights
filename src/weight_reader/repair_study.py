"""Predeclare, freeze, and evaluate a repair cohort on the held-out split."""
import hashlib
import io
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch

from .data import make_split
from .editing import validation_metrics
from .model import WeightReaderMLP
from .probe import _inputs, _session_lock, _session_paths


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _config_bytes(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _study(root):
    return Path(root) / 'private' / 'repair_study'


def _write_new(path, data):
    # Publish complete JSON atomically and refuse to replace an existing artifact.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
            temporary = f.name
            json.dump(data, f, indent=2, allow_nan=False)
            f.write('\n')
        os.link(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def plan_study(root, entries, config, template):
    """Predeclare the exact reviewer × condition × checkpoint sessions before collection."""
    root = Path(root)
    if not isinstance(entries, list) or not entries:
        raise ValueError('study plan must be a nonempty array of sessions')
    splits = config['splits']
    seeds = [item['seed'] for item in splits.values()]
    if len(set(seeds)) != len(seeds):
        raise ValueError('dataset split seeds must be distinct')
    if not (root / 'private' / 'splits.pt').is_file():
        raise ValueError('prepare the private data splits before planning')
    planned, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'session_id', 'reviewer', 'condition', 'checkpoint'}:
            raise ValueError('each planned session needs session_id, reviewer, condition, checkpoint')
        sid, reviewer, condition = entry['session_id'], entry['reviewer'], entry['condition']
        _session_paths(root, sid)
        if sid in seen:
            raise ValueError('planned session IDs must be unique')
        seen.add(sid)
        if not isinstance(reviewer, str) or not reviewer.strip() or condition not in ('weights_only', 'fixed_probes', 'adaptive_probes'):
            raise ValueError('each session needs a reviewer and valid condition')
        if (root / 'private' / 'review_sessions' / sid).exists():
            raise ValueError('plan before creating any listed session')
        checkpoint_hash = _hash(Path(entry['checkpoint']).read_bytes())
        planned.append({'session_id': sid, 'reviewer': reviewer, 'condition': condition,
                        'checkpoint_sha256': checkpoint_hash})
    if len({(e['reviewer'], e['condition'], e['checkpoint_sha256']) for e in planned}) != len(planned):
        raise ValueError('duplicate reviewer, condition, and checkpoint combination')
    study = _study(root)
    study.mkdir(parents=True, exist_ok=True)
    with _session_lock(study):
        if (study / 'plan.json').exists():
            raise ValueError('repair study already planned in this output directory')
        existing = root / 'private' / 'review_sessions'
        for path in existing.glob('*/session.json'):
            m = json.loads(path.read_text())
            if m.get('task') == 'repair' and m.get('phase') == 'main':
                raise ValueError('cannot plan after main repair sessions already exist')
        plan = {'version': 1, 'sessions': planned, 'config_sha256': _hash(_config_bytes(config)),
                'template_sha256': _hash(Path(template).read_bytes()),
                'splits_sha256': _hash((root / 'private' / 'splits.pt').read_bytes()),
                'evaluation_split': config['splits']['final_evaluation'],
                'fixed_probe_split': config['splits']['fixed_probes'],
                'probe_rounds': config['reviewer']['probe_rounds'],
                'inputs_per_round': config['reviewer']['inputs_per_round']}
        _write_new(study / 'plan.json', plan)
    return study / 'plan.json'


def verify_planned_export(root, sid, checkpoint, condition, reviewer, config, template):
    study = _study(root)
    if (study / 'freeze.json').exists():
        raise ValueError('repair study is frozen; new main sessions are blocked')
    plan = json.loads((study / 'plan.json').read_text())
    if _hash(_config_bytes(config)) != plan['config_sha256'] or _hash(Path(template).read_bytes()) != plan['template_sha256']:
        raise ValueError('config or repair prompt differs from the predeclared plan')
    candidate = {'session_id': sid, 'reviewer': reviewer, 'condition': condition,
                 'checkpoint_sha256': _hash(Path(checkpoint).read_bytes())}
    if candidate not in plan['sessions']:
        raise ValueError('session does not match the predeclared plan')


def _session_artifacts(root, entry, plan):
    """Check consistency and return paths of all evidence to lock into the freeze."""
    sid = entry['session_id']
    private = root / 'private' / 'review_sessions' / sid
    meta = json.loads((private / 'session.json').read_text())
    frozen = json.loads((private / 'repair_frozen.json').read_text())
    if any(meta.get(k) != entry[k] for k in ('session_id', 'reviewer', 'condition', 'checkpoint_sha256')) or meta.get('task') != 'repair' or meta.get('phase') != 'main':
        raise ValueError(f'{sid}: session does not match plan')
    if frozen.get('status') != 'frozen_not_evaluated' or frozen.get('packet_id') != meta['packet_id'] or frozen.get('checkpoint_sha256') != meta['checkpoint_sha256']:
        raise ValueError(f'{sid}: invalid frozen proposal')
    raw_dir = root / 'responses' / 'raw' / sid
    accepted = raw_dir / ('format_repair.txt' if frozen['format_repair_used'] else 'original.txt')
    checks = [(private/'checkpoint.pt', meta['checkpoint_sha256']),
              (private/'repair_checkpoint.pt', frozen['edited_checkpoint_sha256']),
              (root/'packets'/f"{meta['packet_id']}.json", meta['packet_sha256']),
              (root/'packets'/f"{meta['packet_id']}.md", meta['prompt_sha256']),
              (raw_dir/'original.txt', frozen['original_response_sha256']),
              (accepted, frozen['accepted_response_sha256'])]
    paths = [private/'session.json', private/'repair_frozen.json']
    for path, expected in checks:
        if _hash(path.read_bytes()) != expected:
            raise ValueError(f'{sid}: artifact hash mismatch: {path.name}')
        if path not in paths:
            paths.append(path)
    if entry['condition'] != 'weights_only':
        probe_dir, transcript_dir = _session_paths(root, sid)
        probe_meta = json.loads((probe_dir / 'session.json').read_text())
        if probe_meta['checkpoint_sha256'] != meta['checkpoint_sha256'] or probe_meta['condition'] != entry['condition'] or _hash((probe_dir / 'checkpoint.pt').read_bytes()) != meta['checkpoint_sha256']:
            raise ValueError(f'{sid}: probe session does not match review')
        if probe_meta['rounds'] != plan['probe_rounds'] or probe_meta['batch_size'] != plan['inputs_per_round']:
            raise ValueError(f'{sid}: probe budget differs from plan')
        if entry['condition'] == 'fixed_probes':
            expected_inputs = make_split(**plan['fixed_probe_split'])[0].tolist()
            if probe_meta['fixed_inputs'] != expected_inputs:
                raise ValueError(f'{sid}: fixed probe inputs differ from plan')
        model = WeightReaderMLP()
        model.load_state_dict(torch.load(probe_dir/'checkpoint.pt', map_location='cpu', weights_only=True))
        model.eval()
        paths += [probe_dir/'session.json', probe_dir/'checkpoint.pt']
        for i in range(1, probe_meta['rounds']+1):
            path = transcript_dir/f'round-{i}.json'
            record = json.loads(path.read_text())
            if record.get('round') != i:
                raise ValueError(f'{sid}: invalid probe round number')
            inputs = _inputs(record['inputs'], probe_meta['batch_size'])
            if entry['condition'] == 'fixed_probes':
                start = (i-1) * probe_meta['batch_size']
                if inputs.tolist() != expected_inputs[start:start+probe_meta['batch_size']]:
                    raise ValueError(f'{sid}: fixed transcript inputs differ from plan')
            with torch.no_grad():
                expected_logits = model(inputs).squeeze(1)
            actual = record['logits']
            if not isinstance(actual, list) or len(actual) != len(inputs):
                raise ValueError(f'{sid}: invalid probe logits')
            if not torch.isfinite(torch.tensor(actual)).all() or not torch.allclose(torch.tensor(actual), expected_logits, rtol=1e-6, atol=1e-6):
                raise ValueError(f'{sid}: probe logits do not match checkpoint')
            paths.append(path)
    remote = root/'private'/'remote_calls'/sid
    if remote.is_dir():
        from .remote import _responses_from_journal
        journals = _responses_from_journal(remote)
        required = 1 if entry['condition'] == 'weights_only' else 5
        if len(journals) != required:
            raise ValueError(f'{sid}: remote conversation is incomplete')
        # The remote transcript must match the original response. A separately
        # archived, explicitly designated format-only repair may differ in bytes.
        if journals[-1]['text'].encode('utf-8') != (raw_dir/'original.txt').read_bytes():
            raise ValueError(f'{sid}: original response differs from remote conversation')
        pinned = (root/'private'/'remote_calls'/'config.sha256')
        if any(row['config_sha256'] != pinned.read_text().strip() for row in journals):
            raise ValueError(f'{sid}: remote settings do not match pinned settings')
        paths += sorted(remote.glob('turn-*.json'))
        paths += sorted(remote.glob('resolution-*.json'))
        paths += [remote/'final.txt', pinned]
    return {str(path.relative_to(root)): _hash(path.read_bytes()) for path in paths}


def freeze_study(root, missing):
    # Keep any provider request from racing with a study freeze. Remote calls
    # acquire the same locks in remote-then-study order.
    remote = Path(root) / 'private' / 'remote_calls'
    if remote.is_dir():
        with _session_lock(remote):
            return _freeze_study(root, missing)
    return _freeze_study(root, missing)


def _freeze_study(root, missing):
    """Freeze all planned proposals or explicitly record missing sessions with reasons."""
    root = Path(root)
    study = _study(root)
    if not (study / 'plan.json').is_file():
        raise ValueError('repair study has not been planned')
    if not isinstance(missing, dict) or any(not isinstance(v, str) or not v.strip() for v in missing.values()):
        raise ValueError('missing must map session IDs to nonempty reasons')
    with _session_lock(study):
        if (study / 'freeze.json').exists():
            raise ValueError('repair study is already frozen')
        plan_bytes = (study / 'plan.json').read_bytes()
        plan = json.loads(plan_bytes)
        if set(missing) - {entry['session_id'] for entry in plan['sessions']}:
            raise ValueError('missing list contains an unplanned session')
        if _hash((root/'private'/'splits.pt').read_bytes()) != plan['splits_sha256']:
            raise ValueError('private data splits changed since planning')
        entries = []
        with ExitStack() as stack:
            for entry in plan['sessions']:
                sid = entry['session_id']
                session_dir = root/'private'/'review_sessions'/sid
                if session_dir.is_dir():
                    stack.enter_context(_session_lock(session_dir))
            for entry in plan['sessions']:
                sid = entry['session_id']
                frozen_path = root/'private'/'review_sessions'/sid/'repair_frozen.json'
                if sid in missing:
                    if frozen_path.exists():
                        raise ValueError(f'{sid}: cannot mark a frozen proposal as missing')
                    partial = []
                    session_dir = root/'private'/'review_sessions'/sid
                    raw_dir = root/'responses'/'raw'/sid
                    probe_dir, transcripts = _session_paths(root, sid)
                    for name in ('session.json', 'checkpoint.pt', 'session_issue.json'):
                        path = session_dir/name
                        if path.is_file(): partial.append(path)
                    for name in ('original.txt', 'format_repair.txt'):
                        path = raw_dir/name
                        if path.is_file(): partial.append(path)
                    for name in ('session.json', 'checkpoint.pt'):
                        path = probe_dir/name
                        if path.is_file(): partial.append(path)
                    rounds = sorted(transcripts.glob('round-*.json')) if transcripts.is_dir() else []
                    partial += rounds
                    remote_dir = root/'private'/'remote_calls'/sid
                    if remote_dir.is_dir():
                        partial += sorted(remote_dir.glob('turn-*.json'))
                        partial += sorted(remote_dir.glob('resolution-*.json'))
                        for name in ('final.txt', 'config.sha256'):
                            path = remote_dir/name
                            if path.is_file(): partial.append(path)
                    artifacts = {str(path.relative_to(root)): _hash(path.read_bytes()) for path in partial}
                    entries.append({'session_id':sid, 'status':'missing', 'reason':missing[sid],
                                    'completed_probe_rounds':len(rounds), 'artifacts':artifacts})
                else:
                    if not frozen_path.is_file():
                        raise ValueError(f'{sid}: proposal is not frozen; explicitly record it missing')
                    entries.append({'session_id':sid, 'status':'frozen', 'artifacts':_session_artifacts(root,entry,plan)})
            if not any(item['status'] == 'frozen' for item in entries):
                raise ValueError('at least one frozen proposal is required')
            freeze = {'version':1, 'plan_sha256':_hash(plan_bytes), 'sessions':entries,
                      'evaluation_split':plan['evaluation_split'], 'splits_sha256':plan['splits_sha256']}
            _write_new(study/'freeze.json', freeze)
    return study/'freeze.json'


def evaluate_study(root):
    """Score the frozen cohort once; return aggregate accuracy and explicit missing rows."""
    root = Path(root)
    study = _study(root)
    if not (study/'freeze.json').is_file():
        raise ValueError('freeze the repair study before held-out evaluation')
    with _session_lock(study):
        freeze_bytes = (study/'freeze.json').read_bytes()
        freeze = json.loads(freeze_bytes)
        plan_bytes = (study/'plan.json').read_bytes()
        if _hash(plan_bytes) != freeze['plan_sha256']:
            raise ValueError('study plan changed after freeze')
        plan = json.loads(plan_bytes)
        if _hash((root/'private'/'splits.pt').read_bytes()) != freeze['splits_sha256']:
            raise ValueError('private data splits changed after freeze')
        for item in freeze['sessions']:
            for name, expected in item['artifacts'].items():
                if _hash((root/name).read_bytes()) != expected:
                    raise ValueError(f"{item['session_id']}: frozen artifact changed")
        output = root/'results'/'repair_study.json'
        if output.exists():
            result = json.loads(output.read_text())
            if result.get('freeze_sha256') != _hash(freeze_bytes):
                raise ValueError('existing result belongs to a different freeze')
            return result
        splits = torch.load(root/'private'/'splits.pt', map_location='cpu', weights_only=True)
        final = splits['final_evaluation']
        expected = make_split(**freeze['evaluation_split'])
        if not all(torch.equal(a,b) for a,b in zip(final, expected)):
            raise ValueError('saved final split does not match the planned configuration')
        planned_by_id = {item['session_id']:item for item in plan['sessions']}
        rows = []
        for item in freeze['sessions']:
            entry = planned_by_id[item['session_id']]
            row = dict(entry, status=item['status'])
            if item['status'] == 'missing':
                row['reason'] = item['reason']
                row['completed_probe_rounds'] = item['completed_probe_rounds']
            else:
                session = root/'private'/'review_sessions'/item['session_id']
                models = []
                for name in ('checkpoint.pt','repair_checkpoint.pt'):
                    model = WeightReaderMLP()
                    model.load_state_dict(torch.load(session/name,map_location='cpu',weights_only=True))
                    models.append(model.eval())
                before, after = (validation_metrics(model, final) for model in models)
                row.update(before=before, after=after, accuracy_change=after['accuracy']-before['accuracy'])
            rows.append(row)
        result = {'study':'weight_repair', 'split':'final_evaluation', 'freeze_sha256':_hash(freeze_bytes),
                  'sample_count':len(final[0]), 'sessions':rows}
        output.parent.mkdir(parents=True,exist_ok=True)
        _write_new(output,result)
        return result
