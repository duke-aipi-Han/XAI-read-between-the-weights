"""Create opaque reviewer packets for the ablation-prediction study."""
import hashlib
import io
import json
import secrets
from pathlib import Path

import torch

from .model import WeightReaderMLP
from .probe import _session_paths, _session_lock, start_session


CONDITION_INSTRUCTIONS = {
    'weights_only': 'Submit your final prediction now using the supplied weights and architecture.',
    'fixed_probes': 'You will receive four batches of eight fixed input/logit pairs. Reply `READY` to receive each batch. After the fourth batch, submit your final prediction.',
    'adaptive_probes': 'Choose four batches of eight inputs. For each batch, reply with a JSON object with an `inputs` array. Each input contains four finite values in [-1, 1]. You may add a short `hypothesis` string describing what the batch tests. Use the logits you receive to choose your next batch. After the fourth batch, submit your final prediction.',
}


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def export_review(root, session_id, checkpoint, condition, reviewer, phase, config, template_path, task='ablation'):
    if task == 'repair' and phase == 'main':
        from .repair_study import verify_planned_export
        study = Path(root) / 'private' / 'repair_study'
        if not (study / 'plan.json').is_file():
            raise ValueError('plan the repair study before exporting main sessions')
        with _session_lock(study):
            verify_planned_export(root, session_id, checkpoint, condition, reviewer, config, template_path)
            return _export_review(root, session_id, checkpoint, condition, reviewer, phase, config, template_path, task)
    return _export_review(root, session_id, checkpoint, condition, reviewer, phase, config, template_path, task)


def _export_review(root, session_id, checkpoint, condition, reviewer, phase, config, template_path, task='ablation'):
    """Export reviewer-facing files and private mapping; never send the private mapping."""
    if task not in ('ablation', 'repair'):
        raise ValueError('invalid review task')
    if condition not in CONDITION_INSTRUCTIONS or phase not in ('calibration', 'main'):
        raise ValueError('invalid condition or phase')
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('reviewer identity is required')
    # This prompt explicitly promises four rounds of eight; do not silently drift.
    if (config['reviewer']['probe_rounds'], config['reviewer']['inputs_per_round']) != (4, 8):
        raise ValueError('reviewer template requires four rounds of eight inputs')
    root = Path(root)
    probe_private, transcript_dir = _session_paths(root, session_id)
    private_dir = root / 'private' / 'review_sessions' / session_id
    if private_dir.exists() or probe_private.exists() or transcript_dir.exists():
        raise ValueError('session ID already exists')
    source_bytes = Path(checkpoint).read_bytes()
    model = WeightReaderMLP()
    model.load_state_dict(torch.load(io.BytesIO(source_bytes), map_location='cpu', weights_only=True))
    state = model.state_dict()
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError('checkpoint contains non-finite weights')
    template = Path(template_path).read_text()
    delimiter = '\n---\n'
    if template.count(delimiter) != 1:
        raise ValueError('reviewer template must have one operator delimiter')
    body = template.split(delimiter, 1)[1].strip()
    if body.count('{{CONDITION_INSTRUCTIONS}}') != 1 or body.count('{{EVIDENCE_PACKET_JSON}}') != 1:
        raise ValueError('reviewer template placeholders are missing or duplicated')
    packet_id = 'p_' + secrets.token_hex(12)
    packet = {'packet_id': packet_id, 'architecture': [4, 8, 8, 1],
              'hidden_activation': 'ReLU', 'output': 'logit', 'weight_orientation': 'output_by_input',
              'parameters': {name: state[key].tolist() for name, key in (
                  ('W1', 'fc1.weight'), ('b1', 'fc1.bias'), ('W2', 'fc2.weight'),
                  ('b2', 'fc2.bias'), ('W3', 'fc3.weight'), ('b3', 'fc3.bias'))}}
    packet_text = json.dumps(packet, indent=2, allow_nan=False) + '\n'
    instruction = CONDITION_INSTRUCTIONS[condition]
    if task == 'repair':
        instruction = instruction.replace('final prediction', 'final edit proposal')
    prompt = body.replace('{{CONDITION_INSTRUCTIONS}}', instruction)
    prompt = prompt.replace('{{EVIDENCE_PACKET_JSON}}', packet_text.rstrip()) + '\n'
    packet_dir = root / 'packets'
    packet_path, prompt_path = packet_dir / f'{packet_id}.json', packet_dir / f'{packet_id}.md'
    if packet_path.exists() or prompt_path.exists():
        raise ValueError('packet ID collision; retry')
    # Reserve the session before writing public artifacts, so duplicate IDs cannot overwrite it.
    private_dir.mkdir(parents=True, exist_ok=False)
    snapshot = private_dir / 'checkpoint.pt'
    snapshot.write_bytes(source_bytes)
    if condition != 'weights_only':
        start_session(root, session_id, snapshot, condition, config)
    packet_dir.mkdir(parents=True, exist_ok=True)
    with packet_path.open('x') as f:
        f.write(packet_text)
    with prompt_path.open('x') as f:
        f.write(prompt)
    mapping = {'session_id': session_id, 'reviewer': reviewer, 'phase': phase, 'task': task,
               'condition': condition, 'packet_id': packet_id,
               'checkpoint_sha256': _hash(source_bytes),
               'packet_sha256': _hash(packet_text.encode('utf-8')),
               'prompt_sha256': _hash(prompt.encode('utf-8')),
               'template_sha256': _hash(template.encode('utf-8'))}
    (private_dir / 'session.json').write_text(json.dumps(mapping, indent=2) + '\n')
    return packet_path, prompt_path
