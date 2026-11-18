"""Budgeted, resumable manual probing against an immutable checkpoint snapshot."""
import fcntl
import hashlib
import io
import json
import math
import re
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

from .data import make_split
from .model import WeightReaderMLP


def _session_paths(root, session_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', session_id):
        raise ValueError('session ID must contain only letters, digits, hyphens, and underscores')
    root = Path(root)
    return root / 'private' / 'probe_sessions' / session_id, root / 'responses' / 'probes' / session_id


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _inputs(rows, batch_size):
    if not isinstance(rows, list) or len(rows) != batch_size:
        raise ValueError(f'inputs must contain exactly {batch_size} four-coordinate inputs')
    for row in rows:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError('each input must contain exactly four numbers')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v) > 1 for v in row):
            raise ValueError('input coordinates must be finite numbers in [-1, 1]')
    return torch.tensor(rows, dtype=torch.float32)


@contextmanager
def _session_lock(directory):
    with (directory / '.lock').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def start_session(root, session_id, checkpoint, condition, config):
    """Create a uniquely named session and snapshot its model and fixed inputs."""
    if condition not in ('fixed_probes', 'adaptive_probes'):
        raise ValueError('condition must be fixed_probes or adaptive_probes')
    settings = config['reviewer']
    rounds, batch_size = settings['probe_rounds'], settings['inputs_per_round']
    if (isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0 or
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0):
        raise ValueError('probe rounds and batch size must be positive integers')
    if condition == 'fixed_probes' and batch_size % 4:
        raise ValueError('fixed probe batch size must be divisible by four')
    checkpoint_bytes = Path(checkpoint).read_bytes()
    # Validate that the snapshot is a compatible model before creating a session.
    model = WeightReaderMLP()
    model.load_state_dict(torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True))
    private_dir, transcript_dir = _session_paths(root, session_id)
    if private_dir.exists() or transcript_dir.exists():
        raise ValueError('session ID already exists')
    fixed = None
    if condition == 'fixed_probes':
        split = config['splits']['fixed_probes']
        if not split['quadrant_balanced'] or split['size'] != rounds * batch_size:
            raise ValueError('fixed split must be balanced and match the session budget')
        fixed = make_split(**split)[0].tolist()
    private_dir.mkdir(parents=True, exist_ok=False)
    transcript_dir.mkdir(parents=True, exist_ok=False)
    (private_dir / 'checkpoint.pt').write_bytes(checkpoint_bytes)
    metadata = {'session_id': session_id, 'condition': condition, 'rounds': rounds,
                'batch_size': batch_size, 'checkpoint_sha256': _sha256(checkpoint_bytes),
                'fixed_inputs': fixed}
    (private_dir / 'session.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')


def probe_round(root, session_id, round_number, payload=None):
    """Return original logits and record one immutable transcript; replay is idempotent."""
    private_dir, transcript_dir = _session_paths(root, session_id)
    if not (private_dir / 'session.json').is_file() or not transcript_dir.is_dir():
        raise ValueError('unknown or incomplete session')
    with _session_lock(private_dir):
        metadata = json.loads((private_dir / 'session.json').read_text())
        if not isinstance(round_number, int) or isinstance(round_number, bool) or not 1 <= round_number <= metadata['rounds']:
            raise ValueError('round is outside the session budget')
        hypothesis = None
        if metadata['condition'] == 'fixed_probes':
            if payload is not None:
                raise ValueError('fixed sessions do not accept input files')
            start = (round_number - 1) * metadata['batch_size']
            inputs = _inputs(metadata['fixed_inputs'][start:start + metadata['batch_size']], metadata['batch_size'])
        else:
            if not isinstance(payload, dict) or set(payload) - {'inputs', 'hypothesis'} or 'inputs' not in payload:
                raise ValueError('adaptive input must be an object with inputs and optional hypothesis')
            hypothesis = payload.get('hypothesis')
            if hypothesis is not None and not isinstance(hypothesis, str):
                raise ValueError('hypothesis must be a string')
            inputs = _inputs(payload['inputs'], metadata['batch_size'])
        rows = inputs.tolist()
        checkpoint_path = private_dir / 'checkpoint.pt'
        if _sha256(checkpoint_path.read_bytes()) != metadata['checkpoint_sha256']:
            raise ValueError('session checkpoint has changed')
        transcript = transcript_dir / f'round-{round_number}.json'
        if transcript.exists():
            saved = json.loads(transcript.read_text())
            if saved['inputs'] != rows:
                raise ValueError('round already recorded with different inputs')
            return {'inputs': saved['inputs'], 'logits': saved['logits']}
        if any(not (transcript_dir / f'round-{i}.json').is_file() for i in range(1, round_number)):
            raise ValueError('complete earlier rounds first')
        model = WeightReaderMLP()
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        model.eval()
        with torch.no_grad():
            logits = model(inputs).squeeze(1).tolist()
        if not all(math.isfinite(v) for v in logits):
            raise ValueError('checkpoint produced non-finite logits')
        record = {'round': round_number, 'inputs': rows, 'logits': logits}
        if hypothesis is not None:
            record['hypothesis'] = hypothesis
        # Publish a complete transcript atomically while holding the session lock.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=transcript_dir, delete=False) as file:
                temporary = file.name
                json.dump(record, file, indent=2, allow_nan=False)
                file.write('\n')
            os.replace(temporary, transcript)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return {'inputs': rows, 'logits': logits}
