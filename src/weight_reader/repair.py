"""Archive and freeze a constrained reviewer repair without using evaluation data."""
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path

import torch

from .editing import edit_weights
from .model import WeightReaderMLP
from .probe import _session_lock, _session_paths


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _parse_response(text, packet_id, model):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    try:
        response = json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON: {exc.msg}') from exc
    if not isinstance(response, dict) or set(response) != {'packet_id', 'hypothesis', 'edit', 'expected_effect'}:
        raise ValueError('response must contain exactly packet_id, hypothesis, edit, and expected_effect')
    if response['packet_id'] != packet_id:
        raise ValueError('packet ID does not match this session')
    for field in ('hypothesis', 'expected_effect'):
        if not isinstance(response[field], str) or not response[field].strip():
            raise ValueError(f'{field} must be a nonempty string')
    edit = response['edit']
    if not isinstance(edit, dict) or set(edit) != {'parameter', 'index', 'scale'}:
        raise ValueError('edit must contain exactly parameter, index, and scale')
    param, index, scale = edit['parameter'], edit['index'], edit['scale']
    parameters = dict(model.named_parameters())
    if not isinstance(param, str) or param not in parameters:
        raise ValueError('unknown parameter')
    if not isinstance(index, list) or len(index) != parameters[param].ndim or sum(i is None for i in index) > (1 if parameters[param].ndim == 2 else 0):
        raise ValueError('index must select one scalar or one matrix row or column')
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 <= scale <= 2:
        raise ValueError('scale must be finite and in [0, 2]')
    return response, edit_weights(model, param, index, scale)


def import_repair(root, session_id, response_path, format_repair=False):
    """Preserve original bytes, validate one proposal, and snapshot its edited model.

    A second response can only be recorded explicitly as one format-only repair;
    the operator remains responsible for verifying it did not change scientific content.
    """
    root = Path(root)
    probe_dir, transcript_dir = _session_paths(root, session_id)
    session_dir = root / 'private' / 'review_sessions' / session_id
    if not (session_dir / 'session.json').is_file():
        raise ValueError('unknown or incomplete review session')
    with _session_lock(session_dir):
        meta = json.loads((session_dir / 'session.json').read_text())
        if meta.get('task') != 'repair':
            raise ValueError('session is not a repair task')
        if (session_dir / 'repair_frozen.json').exists():
            raise ValueError('repair proposal already frozen')
        source = (session_dir / 'checkpoint.pt').read_bytes()
        if _sha(source) != meta['checkpoint_sha256']:
            raise ValueError('review checkpoint has changed')
        packet_path = root / 'packets' / (meta['packet_id'] + '.json')
        prompt_path = root / 'packets' / (meta['packet_id'] + '.md')
        if _sha(packet_path.read_bytes()) != meta['packet_sha256'] or _sha(prompt_path.read_bytes()) != meta['prompt_sha256']:
            raise ValueError('packet or prompt has changed')
        if meta['condition'] != 'weights_only':
            probe_meta = json.loads((probe_dir / 'session.json').read_text())
            if _sha((probe_dir / 'checkpoint.pt').read_bytes()) != meta['checkpoint_sha256']:
                raise ValueError('probe checkpoint does not match review checkpoint')
            if any(not (transcript_dir / f'round-{i}.json').is_file() for i in range(1, probe_meta['rounds'] + 1)):
                raise ValueError('complete all probe rounds before submitting a repair')
        raw = Path(response_path).read_bytes()
        archive = root / 'responses' / 'raw' / session_id
        archive.mkdir(parents=True, exist_ok=True)
        original = archive / 'original.txt'
        if format_repair:
            if not original.exists():
                raise ValueError('cannot submit a format repair without an original response')
            destination = archive / 'format_repair.txt'
        else:
            destination = original
        if destination.exists():
            raise ValueError('response already archived; original cannot be overwritten')
        with destination.open('xb') as file:
            file.write(raw)
        model = WeightReaderMLP()
        model.load_state_dict(torch.load(io.BytesIO(source), map_location='cpu', weights_only=True))
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('response is not UTF-8; original has been archived') from exc
        try:
            response, edited = _parse_response(text, meta['packet_id'], model)
        except ValueError as exc:
            raise ValueError(f'{exc}; response has been archived') from exc
        buffer = io.BytesIO()
        torch.save(edited.state_dict(), buffer)
        edited_bytes = buffer.getvalue()
        with (session_dir / 'repair_checkpoint.pt').open('xb') as file:
            file.write(edited_bytes)
        record = {'session_id': session_id, 'packet_id': meta['packet_id'],
                  'original_response_sha256': _sha(original.read_bytes()),
                  'accepted_response_sha256': _sha(raw), 'format_repair_used': format_repair,
                  'checkpoint_sha256': meta['checkpoint_sha256'],
                  'edited_checkpoint_sha256': _sha(edited_bytes), 'edit': response['edit'],
                  'status': 'frozen_not_evaluated'}
        target = session_dir / 'repair_frozen.json'
        with tempfile.NamedTemporaryFile(mode='w', dir=session_dir, delete=False) as file:
            json.dump(record, file, indent=2, allow_nan=False)
            file.write('\n')
            temp_name = file.name
        os.replace(temp_name, target)
        return target


def validate_repair(root, session_id, config):
    """Compare one frozen calibration edit on validation data, never the final split."""
    from .data import make_split
    from .editing import validation_metrics

    root = Path(root)
    _session_paths(root, session_id)  # Validate before constructing a private path.
    session_dir = root / 'private' / 'review_sessions' / session_id
    meta = json.loads((session_dir / 'session.json').read_text())
    if meta.get('task') != 'repair' or meta['phase'] != 'calibration':
        raise ValueError('development comparisons are only available for repair calibration sessions')
    record = json.loads((session_dir / 'repair_frozen.json').read_text())
    models = []
    for filename, key in [('checkpoint.pt', 'checkpoint_sha256'),
                          ('repair_checkpoint.pt', 'edited_checkpoint_sha256')]:
        contents = (session_dir / filename).read_bytes()
        if _sha(contents) != record[key]:
            raise ValueError('frozen checkpoint hash mismatch')
        model = WeightReaderMLP()
        model.load_state_dict(torch.load(io.BytesIO(contents), map_location='cpu', weights_only=True))
        model.eval()
        models.append(model)
    split = make_split(**config['splits']['validation'])
    before, after = (validation_metrics(model, split) for model in models)
    return {'session_id': session_id, 'phase': 'calibration', 'split': 'validation',
            'before': before, 'after': after,
            'accuracy_change': after['accuracy'] - before['accuracy']}
