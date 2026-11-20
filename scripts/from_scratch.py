"""A separately journaled model construction and frozen held-out evaluation.

No network request occurs without ``collect --execute``. The existing budget
ledger and settlement policy apply; prior events are never retried or replaced.
"""
import argparse
import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path

import torch
from weight_reader.data import make_split
from weight_reader.model import WeightReaderMLP
from weight_reader.editing import validation_metrics
from weight_reader.probe import _session_lock
from weight_reader.remote import _extract, _make_body, _reserve_estimate, _responses_from_journal, _utc, read_key, validate_config
from weight_reader.remote_budget import active_commitment, commitment_report

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'experiments/from_scratch_models.json'
REMOTE = ROOT / 'private/remote_calls'
PROMPT = ROOT / 'prompts/FROM_SCRATCH.md'
SHAPES = {'W1': (8, 4), 'b1': (8,), 'W2': (8, 8), 'b2': (8,), 'W3': (1, 8), 'b3': (1,)}


def config():
    result = json.loads((ROOT / 'remote_config.json').read_text())
    validate_config(result)
    digest = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (REMOTE / 'config.sha256').read_text().strip() != digest:
        raise ValueError('Pinned remote configuration does not match')
    return result, digest


def parse_weights(text):
    value = text.strip()
    if value.startswith('```json\n') and value.endswith('```'):
        value = value[len('```json\n'):-3].strip()
    data = json.loads(value)
    if not isinstance(data, dict) or set(data) != set(SHAPES):
        raise ValueError('Response must contain exactly W1, b1, W2, b2, W3, b3')
    def check(obj, shape):
        if not shape:
            if isinstance(obj, bool) or not isinstance(obj, (float, int)) or not math.isfinite(obj):
                raise ValueError('Every weight must be a finite number')
            return
        if not isinstance(obj, list) or len(obj) != shape[0]:
            raise ValueError('Weight shape mismatch')
        for item in obj:
            check(item, shape[1:])
    for name, shape in SHAPES.items():
        check(data[name], shape)
    return data


def network(weights):
    model = WeightReaderMLP()
    state = {key: torch.tensor(weights[name], dtype=torch.float32) for name, key in (
        ('W1', 'fc1.weight'), ('b1', 'fc1.bias'), ('W2', 'fc2.weight'),
        ('b2', 'fc2.bias'), ('W3', 'fc3.weight'), ('b3', 'fc3.bias'))}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Weights are not finite in float32')
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_response(text):
    """Document format-only normalization; never alter a numerical value."""
    value = text.strip()
    repairs = []
    if value.startswith('```json\n') and value.endswith('```'):
        value = value[len('```json\n'):-3].strip()
        repairs.append('Removed outer JSON code fence')
    data = json.loads(value)
    if isinstance(data, dict) and isinstance(data.get('b3'), (int, float)) and not isinstance(data['b3'], bool):
        data['b3'] = [data['b3']]
        repairs.append('Wrapped scalar output bias b3 in a one-element array')
    return parse_weights(json.dumps(data, allow_nan=False)), repairs


def reviewer_config(alias, cfg):
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,24}', alias):
        raise ValueError('Invalid reviewer alias')
    models = json.loads(REGISTRY.read_text())
    if alias not in models:
        raise ValueError('Reviewer is not configured in experiments/from_scratch_models.json')
    item = models[alias]
    provider = item.get('provider')
    if provider not in ('openai', 'openrouter') or not isinstance(item.get('model'), str) or not item['model']:
        raise ValueError('Reviewer must specify a supported provider and model ID')
    section = cfg[provider]
    price_keys = ('input_reservation_usd_per_million', 'output_reservation_usd_per_million') if provider == 'openai' else ('max_prompt_usd_per_million', 'max_completion_usd_per_million')
    for user_key, limit_key in zip(('input_usd_per_million','output_usd_per_million'),price_keys):
        price=item.get(user_key)
        if isinstance(price,bool) or not isinstance(price,(int,float)) or not math.isfinite(price) or not 0 <= price <= section[limit_key]:
            raise ValueError('Registered prices must be nonnegative and within the existing provider reservation limits')
    return item


def paths(alias):
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,24}', alias):
        raise ValueError('Invalid reviewer alias')
    if alias == 'astra':
        return 'v3-astra-from-scratch', ROOT / 'private/from_scratch_astra/frozen.json', ROOT / 'results/astra_from_scratch.json'
    return f'v3-{alias}-from-scratch', ROOT / f'private/from_scratch_{alias}/frozen.json', ROOT / f'results/{alias}_from_scratch.json'


def collect(execute, alias):
    cfg, digest = config()
    model_cfg = reviewer_config(alias, cfg)
    provider = model_cfg['provider']
    session, _, _ = paths(alias)
    key_name = 'OPENAI_API_KEY' if provider == 'openai' else 'OPENROUTER_API_KEY'
    prompt = PROMPT.read_text()
    body = _make_body(provider, cfg, [], prompt)
    body['model'] = model_cfg['model']
    reservation = _reserve_estimate(provider, cfg, body, [])
    if not execute:
        print(json.dumps({'model': body['model'], 'reservation_usd': reservation,
                          'committed_usd': active_commitment(ROOT, cfg),
                          'ceiling_usd': cfg['budget_usd'],
                          'key_present': bool(read_key(key_name, ROOT / '.env')),
                          'prior_attempt_exists': (REMOTE / session / 'turn-01.json').exists()}))
        return
    key = read_key(key_name, ROOT / '.env')
    if not key:
        raise ValueError(f'{provider} API key missing')
    with _session_lock(REMOTE):
        used = active_commitment(ROOT, cfg)
        if used + reservation > cfg['budget_usd']:
            raise ValueError('Local reservation ceiling would be exceeded')
        folder = REMOTE / session
        folder.mkdir(exist_ok=True)
        event_path = folder / 'turn-01.json'
        if event_path.exists():
            raise ValueError('A prior attempt exists; inspect it instead of repeating the request')
        ledger_path = REMOTE / 'ledger.json'
        ledger = json.loads(ledger_path.read_text())
        event = {'session_id': session, 'turn': 1, 'provider': provider, 'status': 'pending',
                 'created_at': _utc(), 'config_sha256': digest, 'user_text': prompt,
                 'request': body, 'reserved_usd': reservation}
        with event_path.open('x') as f:
            json.dump(event, f, indent=2, allow_nan=False)
        ledger['reservations'].append({'session_id': session, 'turn': 1, 'reserved_usd': reservation})
        temp = REMOTE / 'ledger.next.json'
        temp.write_text(json.dumps(ledger, indent=2) + '\n')
        os.replace(temp, ledger_path)
        req = urllib.request.Request(cfg[provider]['endpoint'], data=json.dumps(body).encode(),
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=180) as stream:
                response = json.loads(stream.read())
            event['response'] = response
            event['text'] = _extract(provider, response, cfg['max_output_tokens'])
            event['status'] = 'completed'
        except Exception as exc:
            event['status'] = 'unresolved'
            event['error_type'] = type(exc).__name__
            event['http_status'] = getattr(exc, 'code', None)
            raise ValueError('Request failed or was incomplete; see journal; no automatic retry') from None
        finally:
            event['completed_at'] = _utc()
            event_path.write_text(json.dumps(event, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'completed', 'reservation_usd': reservation,
                      'usage': response.get('usage'), 'reported_model': response.get('model')}))


def freeze(alias):
    session, frozen_path, _ = paths(alias)
    if frozen_path.exists():
        raise ValueError('Already frozen')
    path = REMOTE / session / 'turn-01.json'
    raw = path.read_bytes()
    event = _responses_from_journal(path.parent, through_turn=1)[0]
    if event.get('user_text') != PROMPT.read_text():
        raise ValueError('The construction prompt differs from the one in the request')
    weights, repairs = normalize_response(event['text'])
    network(weights)  # Validate before accessing evaluation inputs.
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    data = {'session_id': session, 'frozen_at': _utc(),
            'response_sha256': hashlib.sha256(raw).hexdigest(),
            'prompt_sha256': hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
            'format_repairs': repairs, 'weights': weights}
    with frozen_path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
    print(json.dumps({'status':'frozen', 'format_repairs':repairs, 'held_out_loaded':False}))


def reconcile(alias):
    """Accept a valid already-returned output overrun within its reservation."""
    session, _, _ = paths(alias)
    with _session_lock(REMOTE):
        folder = REMOTE / session
        path = folder / 'turn-01.json'
        raw = path.read_bytes()
        event = json.loads(raw)
        target = folder / 'resolution-01.json'
        if target.exists() or event.get('status') != 'unresolved' or event.get('provider') != 'openrouter':
            raise ValueError('No unresolved OpenRouter overrun to reconcile')
        response = event['response']; usage = response['usage']
        count = usage.get('completion_tokens'); cost = usage.get('cost'); reserved = event['reserved_usd']
        if (isinstance(count, bool) or not isinstance(count, int) or count <= event['request']['max_tokens'] or
                isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or not 0 <= cost <= reserved):
            raise ValueError('Reported overrun or cost is not eligible for reconciliation')
        ledger = json.loads((REMOTE / 'ledger.json').read_text())
        if not any(x['session_id'] == session and x['turn'] == 1 and x['reserved_usd'] == reserved for x in ledger['reservations']):
            raise ValueError('Matching reservation is missing')
        text = _extract('openrouter', response, count)
        normalize_response(text)  # Reject incomplete or malformed network content.
        record = {'type':'accepted_reported_output_overrun', 'journal_sha256':hashlib.sha256(raw).hexdigest(),
                  'reviewed_at':_utc(), 'requested_output_tokens':event['request']['max_tokens'],
                  'reported_output_tokens':count, 'reported_cost_usd':cost, 'reserved_usd':reserved,
                  'text':text, 'reason':'Accepted previously authorized reported-token overrun within reservation; no retry.'}
        with target.open('x') as f:
            json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps({'status':'reconciled', 'reported_output_tokens':count, 'reported_cost_usd':cost, 'reservation_usd':reserved}))


def evaluate(alias):
    session, frozen_path, result_path = paths(alias)
    if result_path.exists():
        raise ValueError('Result exists; do not silently rescore or replace it')
    frozen = json.loads(frozen_path.read_text())
    raw = (REMOTE / session / 'turn-01.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != frozen['response_sha256']:
        raise ValueError('Frozen response hash mismatch')
    model = network(frozen['weights'])
    options = json.loads((ROOT / 'experiments/repair_v2.json').read_text())['splits']['final_evaluation']
    split = make_split(**options)
    with torch.no_grad():
        if not torch.isfinite(model(split[0])).all():
            raise ValueError('Constructed network produced a non-finite held-out logit')
    reference = WeightReaderMLP()
    reference.load_state_dict(torch.load(ROOT / 'checkpoints/trained_seed_22.pt', map_location='cpu', weights_only=True))
    reference.eval()
    rows = json.loads((ROOT / 'results/repair_v2_study.json').read_text())['sessions']
    baseline = rows[0]['before']
    best = max(rows, key=lambda row: row['after']['accuracy'])
    comparison = {'model': session, 'evaluation_split': options, 'accuracy': validation_metrics(model, split),
                  'original_trained': validation_metrics(reference, split), 'weakened': baseline,
                  'best_prior_repair': {'session_id': best['session_id'], 'accuracy': best['after']['accuracy']},
                  'freeze_sha256': hashlib.sha256(frozen_path.read_bytes()).hexdigest()}
    result_path.parent.mkdir(exist_ok=True)
    with result_path.open('x') as f:
        json.dump(comparison, f, indent=2, allow_nan=False)
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['collect', 'reconcile', 'freeze', 'evaluate', 'budget'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--reviewer', default='astra', help='Alias from experiments/from_scratch_models.json')
    args = parser.parse_args()
    if args.command == 'collect': collect(args.execute, args.reviewer)
    elif args.command == 'reconcile': reconcile(args.reviewer)
    elif args.command == 'freeze': freeze(args.reviewer)
    elif args.command == 'evaluate': evaluate(args.reviewer)
    else:
        c, _ = config()
        report = commitment_report(ROOT, c)
        print(json.dumps({k: report[k] for k in ('ceiling_usd', 'committed_usd', 'remaining_usd')}))
