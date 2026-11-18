"""Optional, explicitly activated settlement of completed remote reservations."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .probe import _session_lock


def _config_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _nonnegative_integer(value):
    return isinstance(value,int) and not isinstance(value,bool) and value >= 0


def commitment_report(root, config):
    """Use provider-reported cost where available; otherwise retain safe estimates."""
    remote=Path(root)/'private'/'remote_calls'
    ledger_path=remote/'ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'reservations':[]}
    rows=[]
    for entry in ledger['reservations']:
        sid, turn, reserved = entry['session_id'], entry['turn'], entry['reserved_usd']
        # Session IDs come from the local ledger, but still reject path traversal.
        from .probe import _session_paths
        _session_paths(root,sid)
        path=remote/sid/f'turn-{turn:02d}.json'
        amount=reserved; basis='reservation_retained'; reason='no verified completed usage'
        if path.is_file():
            event=json.loads(path.read_text())
            if event.get('session_id')!=sid or event.get('turn')!=turn or event.get('reserved_usd')!=reserved:
                raise ValueError('ledger and remote journal do not match')
            response=event.get('response',{})
            usage=response.get('usage',{})
            completed=event.get('status')=='completed'
            if not completed and event.get('status')=='unresolved' and (path.parent/f'resolution-{turn:02d}.json').is_file():
                from .remote import _responses_from_journal
                # Revalidates the immutable original, the resolution, and response contents.
                reconciled=_responses_from_journal(path.parent, through_turn=turn)
                completed=any(x['turn']==turn and x['status']=='completed' for x in reconciled)
            cost=usage.get('cost')
            if completed and isinstance(cost,(int,float)) and not isinstance(cost,bool) and math.isfinite(cost) and cost >= 0:
                amount=cost; basis='provider_reported_cost'; reason=None
            elif completed and event.get('provider')=='openai':
                a,b=usage.get('input_tokens'),usage.get('output_tokens')
                if _nonnegative_integer(a) and _nonnegative_integer(b):
                    amount=(a*config['openai']['input_reservation_usd_per_million']+
                            b*config['openai']['output_reservation_usd_per_million'])/1_000_000
                    basis='conservative_uncached_token_rates'; reason=None
        # Round upward to avoid releasing a fraction of a microdollar.
        amount=math.ceil(amount*1_000_000)/1_000_000
        rows.append({'session_id':sid,'turn':turn,'original_reservation_usd':reserved,
                     'committed_usd':amount,'basis':basis,'reason':reason})
    total=round(sum(row['committed_usd'] for row in rows),6)
    return {'policy':'actual_cost_or_conservative_usage','ceiling_usd':config['budget_usd'],
            'gross_reservations_usd':round(sum(x['reserved_usd'] for x in ledger['reservations']),6),
            'committed_usd':total,'remaining_usd':round(config['budget_usd']-total,6),'turns':rows}


def activate_settlement(root, config):
    """Activate only after explicit operator approval; does not call a model."""
    from .remote import validate_config
    validate_config(config)
    remote=Path(root)/'private'/'remote_calls'
    if not remote.is_dir(): raise ValueError('no remote ledger exists')
    with _session_lock(remote):
        pinned=remote/'config.sha256'
        if not pinned.is_file() or pinned.read_text().strip()!=_config_hash(config):
            raise ValueError('remote configuration differs from pinned settings')
        policy=remote/'budget_policy.json'
        if policy.exists(): raise ValueError('budget policy already activated')
        report=commitment_report(root,config)
        if report['committed_usd'] > config['budget_usd']:
            raise ValueError('conservative committed total exceeds the ceiling')
        record={'policy':'actual_cost_or_conservative_usage','config_sha256':_config_hash(config),
                'activated_at':datetime.now(timezone.utc).isoformat(),
                'ceiling_usd':config['budget_usd']}
        with policy.open('x') as file:
            json.dump(record,file,indent=2);file.write('\n')
        return report


def active_commitment(root,config):
    """Return committed amount under the active policy; default retains all reservations."""
    remote=Path(root)/'private'/'remote_calls'
    ledger=remote/'ledger.json'
    reservations=json.loads(ledger.read_text())['reservations'] if ledger.exists() else []
    policy=remote/'budget_policy.json'
    if not policy.exists():
        return sum(x['reserved_usd'] for x in reservations)
    record=json.loads(policy.read_text())
    if record.get('policy')!='actual_cost_or_conservative_usage' or record.get('config_sha256')!=_config_hash(config):
        raise ValueError('active budget policy does not match pinned settings')
    return commitment_report(root,config)['committed_usd']
