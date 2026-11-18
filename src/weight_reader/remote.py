"""One-turn-at-a-time, explicitly invoked API relay with private journaling.

No request is sent by preflight or import. The local reservation ledger is a
conservative estimate, not a provider-side billing cap or a pricing guarantee.
"""
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .probe import _session_lock, _session_paths, probe_round


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def read_key(name, env_path):
    value = os.environ.get(name, '').strip()
    if value:
        return value
    path = Path(env_path)
    if not path.is_file():
        return ''
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        if key.strip().removeprefix('export ').strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            return value.strip()
    return ''


def validate_config(config):
    if config.get('version') != 1:
        raise ValueError('unsupported remote configuration')
    if not isinstance(config.get('budget_usd'), (int, float)) or isinstance(config['budget_usd'], bool) or not math.isfinite(config['budget_usd']) or not 0 < config['budget_usd'] <= 10:
        raise ValueError('remote budget must be positive and at most USD 10')
    cap = config.get('max_output_tokens')
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= 4096:
        raise ValueError('output token cap must be in 1..4096')
    if config.get('reasoning_effort') != 'medium':
        raise ValueError('the comparison requires medium reasoning effort')
    ai, router = config['openai'], config['openrouter']
    if ai['endpoint'] != 'https://api.openai.com/v1/responses' or ai['model'] != 'gpt-6.2-astra':
        raise ValueError('unexpected OpenAI endpoint or model')
    if router['endpoint'] != 'https://openrouter.ai/api/v1/chat/completions' or router['model'] != 'qwen/qwen3.8-27b':
        raise ValueError('unexpected OpenRouter endpoint or model')
    if router['provider'] != 'novita':
        raise ValueError('the selected OpenRouter provider must remain novita')
    for section, key, minimum in ((ai, 'input_reservation_usd_per_million', 20),
                                  (ai, 'output_reservation_usd_per_million', 60),
                                  (router, 'max_prompt_usd_per_million', 0.5),
                                  (router, 'max_completion_usd_per_million', 5)):
        value = section[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
            raise ValueError(f'invalid price reservation: {key}')


def _session(root, sid):
    _session_paths(root, sid)
    session = Path(root) / 'private' / 'review_sessions' / sid
    meta = json.loads((session / 'session.json').read_text())
    if meta.get('task') != 'repair' or meta.get('phase') not in ('calibration', 'main'):
        raise ValueError('remote relay supports repair sessions only')
    if meta.get('phase') == 'main':
        study=Path(root)/'private'/'repair_study'
        if not (study/'plan.json').is_file() or (study/'freeze.json').exists():
            raise ValueError('main repair study is unplanned or frozen')
        plan=json.loads((study/'plan.json').read_text())
        expected={key:meta[key] for key in ('session_id','reviewer','condition','checkpoint_sha256')}
        if expected not in plan['sessions'] or meta['template_sha256'] != plan['template_sha256']:
            raise ValueError('main session does not match the predeclared study plan')
    if meta.get('reviewer') == 'gpt-6.2-astra':
        provider = 'openai'
    elif meta.get('reviewer') == 'Qwen/Qwen3.8-27B':
        provider = 'openrouter'
    else:
        raise ValueError('session reviewer does not match a configured deployment')
    packet_id = meta['packet_id']
    if not re.fullmatch(r'p_[0-9a-f]{24}', packet_id):
        raise ValueError('invalid packet ID')
    prompt_path = Path(root) / 'packets' / (packet_id + '.md')
    prompt = prompt_path.read_bytes()
    if _digest(prompt) != meta['prompt_sha256']:
        raise ValueError('rendered prompt changed')
    if _digest((session/'checkpoint.pt').read_bytes()) != meta['checkpoint_sha256']:
        raise ValueError('review checkpoint changed')
    packet_path = Path(root) / 'packets' / (packet_id + '.json')
    if _digest(packet_path.read_bytes()) != meta['packet_sha256']:
        raise ValueError('review packet changed')
    if meta['condition'] != 'weights_only':
        probe_dir, transcript_dir = _session_paths(root, sid)
        probe = json.loads((probe_dir/'session.json').read_text())
        if (probe['checkpoint_sha256'] != meta['checkpoint_sha256'] or
                _digest((probe_dir/'checkpoint.pt').read_bytes()) != meta['checkpoint_sha256'] or
                probe['condition'] != meta['condition'] or probe['rounds'] != 4 or probe['batch_size'] != 8):
            raise ValueError('probe session differs from the review session or budget')
    if (session/'repair_frozen.json').exists():
        raise ValueError('session response is already frozen')
    if meta['phase'] == 'main' and (Path(root)/'private'/'repair_study'/'freeze.json').exists():
        raise ValueError('main repair study is frozen')
    return meta, provider, prompt.decode('utf-8')


def _remote_budget_root(root):
    remote_root=Path(root)/'private'/'remote_calls'
    shared_manifest=Path(root)/'private'/'shared_remote_budget.json'
    if shared_manifest.exists():
        target=json.loads(shared_manifest.read_text())['relative_target']
        expected=(remote_root.parent/target).resolve()
        if (not remote_root.is_symlink() or remote_root.resolve()!=expected or
                not (expected/'ledger.json').is_file()):
            raise ValueError('required shared remote budget ledger is unavailable')
    return remote_root


def preflight(root, env_path, config, session_ids=None):
    """Check local configuration and key presence without network access."""
    validate_config(config)
    _remote_budget_root(root)
    status = {'network_calls_made': False, 'credentials_validated_with_provider': False, 'credentials': {}, 'sessions': {},
              'openrouter_provider':config['openrouter']['provider'], 'fallback_enabled':False,
              'budget_usd':config['budget_usd'], 'max_output_tokens':config['max_output_tokens']}
    for provider, name in [('openai','OPENAI_API_KEY'), ('openrouter','OPENROUTER_API_KEY')]:
        status['credentials'][provider] = bool(read_key(name, env_path))
    for sid in (session_ids if session_ids is not None else ('repair-calibration-astra','repair-calibration-qwen')):
        try:
            meta, provider, _ = _session(root,sid)
            folder = Path(root)/'private'/'remote_calls'/sid
            rows = _responses_from_journal(folder) if folder.is_dir() else []
            if not rows and meta['condition'] != 'weights_only':
                _, transcripts = _session_paths(root,sid)
                if any(transcripts.glob('round-*.json')):
                    raise ValueError('session already has manual probe rounds')
            if len(rows) >= (1 if meta['condition']=='weights_only' else 5):
                raise ValueError('remote session has already completed')
            status['sessions'][sid] = {'ready': True, 'provider':provider, 'condition':meta['condition'],
                                       'completed_api_turns':len(rows)}
        except (ValueError, OSError, KeyError):
            status['sessions'][sid] = {'ready': False}
    return status


def _responses_from_journal(folder, through_turn=None):
    rows=[]
    for path in sorted(folder.glob('turn-*.json')):
        raw=path.read_bytes()
        item=json.loads(raw)
        if through_turn is not None and item['turn'] > through_turn:
            break
        resolution_path=folder/f"resolution-{item['turn']:02d}.json"
        if resolution_path.exists():
            resolution=json.loads(resolution_path.read_text())
            if resolution.get('journal_sha256') != _digest(raw):
                raise ValueError('resolved journal changed after review')
            if item.get('status') != 'unresolved' or resolution.get('type') != 'accepted_reported_output_overrun':
                raise ValueError('invalid remote resolution')
            response=item.get('response',{})
            usage=response.get('usage',{})
            count=usage.get('completion_tokens')
            if (item.get('provider') != 'openrouter' or not isinstance(count,int) or
                    count <= item.get('request',{}).get('max_tokens',count) or
                    usage.get('cost') != resolution.get('reported_cost_usd') or
                    item.get('reserved_usd') != resolution.get('reserved_usd') or
                    usage.get('cost',float('inf')) > item['reserved_usd'] or
                    _extract('openrouter',response,count) != resolution.get('text')):
                raise ValueError('remote resolution does not match the original response')
            item=dict(item,status='completed',text=resolution['text'])
        elif item.get('status') != 'completed':
            raise ValueError('prior call has an unresolved outcome; inspect the private journal before proceeding')
        usage=item.get('response',{}).get('usage',{})
        is_openai=item.get('provider')=='openai'
        count=usage.get('output_tokens' if is_openai else 'completion_tokens')
        cap=item.get('request',{}).get('max_output_tokens' if is_openai else 'max_tokens')
        if isinstance(count,int) and isinstance(cap,int) and count > cap and not resolution_path.exists():
            raise ValueError('prior call reported output tokens above the requested cap; stop this session')
        rows.append(item)
    return rows


def reconcile_output_overrun(root, sid, turn):
    """Accept an already-returned Qwen overrun only when its reported cost fits its reservation.

    This never sends or repeats a request. It adds a separate audit record without
    modifying the original journal. Explicit operator action is required per turn.
    """
    root=Path(root)
    _session_paths(root,sid)
    if not isinstance(turn,int) or isinstance(turn,bool) or not 1 <= turn <= 5:
        raise ValueError('invalid turn number')
    remote_root=root/'private'/'remote_calls'
    if not remote_root.is_dir():
        raise ValueError('no remote journal exists')
    with _session_lock(remote_root):
        folder=remote_root/sid
        path=folder/f'turn-{turn:02d}.json'
        raw=path.read_bytes()
        event=json.loads(raw)
        resolution=folder/f'resolution-{turn:02d}.json'
        if resolution.exists():
            raise ValueError('turn has already been resolved')
        if event.get('status')!='unresolved' or event.get('provider')!='openrouter':
            raise ValueError('only an unresolved OpenRouter turn can be reviewed')
        response=event.get('response',{})
        usage=response.get('usage',{})
        reported=usage.get('completion_tokens')
        cap=event['request'].get('max_tokens')
        cost=usage.get('cost')
        if (isinstance(reported,bool) or not isinstance(reported,int) or
                not isinstance(cap,int) or reported <= cap):
            raise ValueError('turn is not a reported output-token overrun')
        if isinstance(cost,bool) or not isinstance(cost,(int,float)) or not math.isfinite(cost) or not 0 <= cost <= event['reserved_usd']:
            raise ValueError('reported cost is missing or exceeds the pre-call reservation')
        ledger=json.loads((remote_root/'ledger.json').read_text())
        if not any(x['session_id']==sid and x['turn']==turn and x['reserved_usd']==event['reserved_usd'] for x in ledger['reservations']):
            raise ValueError('matching reservation is missing')
        text=_extract('openrouter',response,reported)
        meta=json.loads((root/'private'/'review_sessions'/sid/'session.json').read_text())
        is_final=meta['condition']=='weights_only' or turn==5
        if not is_final and meta['condition']=='fixed_probes':
            if text.strip().upper() != 'READY':
                raise ValueError('returned fixed-probe acknowledgement is not READY')
        elif not is_final:
            from .probe import _inputs
            try:
                payload=json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError('returned probe batch is not JSON') from exc
            if not isinstance(payload,dict) or set(payload)-{'inputs','hypothesis'} or 'inputs' not in payload:
                raise ValueError('returned probe batch has an invalid shape')
            _inputs(payload['inputs'],8)
            if 'hypothesis' in payload and not isinstance(payload['hypothesis'],str):
                raise ValueError('returned hypothesis is invalid')
        record={'type':'accepted_reported_output_overrun', 'journal_sha256':_digest(raw),
                'reviewed_at':_utc(),'requested_output_tokens':cap,'reported_output_tokens':reported,
                'reported_cost_usd':cost,'reserved_usd':event['reserved_usd'],'text':text,
                'reason':'Operator accepted a reported token overrun after confirming its cost fit the reservation.'}
        with resolution.open('x') as file:
            json.dump(record,file,indent=2,allow_nan=False)
            file.write('\n')
        if is_final:
            final=folder/'final.txt'
            with final.open('x') as file: file.write(text)
        return {'session_id':sid,'turn':turn,'status':'resolved_without_retry',
                'reported_output_tokens':reported,'reported_cost_usd':cost,
                'reserved_usd':event['reserved_usd'], 'resolution':str(resolution)}


def _next_input(root,sid,meta,prompt,rows):
    if not rows:
        return prompt
    previous=rows[-1]['text']
    step=len(rows)
    if meta['condition']=='weights_only' or step > 4:
        raise ValueError('session has already returned its final response')
    if meta['condition']=='fixed_probes':
        if previous.strip().upper() != 'READY':
            raise ValueError('reviewer did not return READY; inspect the raw response')
        result=probe_round(root,sid,step)
    elif meta['condition']=='adaptive_probes':
        try:
            payload=json.loads(previous)
        except json.JSONDecodeError as exc:
            raise ValueError('reviewer did not return a valid probe JSON object; inspect the raw response') from exc
        result=probe_round(root,sid,step,payload)
    else:
        raise ValueError('invalid session condition')
    feedback=json.dumps(result,allow_nan=False,separators=(',',':'))
    if meta.get('relay_protocol') == 'staged_v3':
        if step == 4:
            instruction='All four batches are complete. Now return only your final edit JSON.'
        elif meta['condition'] == 'fixed_probes':
            instruction='This was fixed batch '+str(step)+' of 4. Your next response must be exactly READY; do not choose inputs.'
        else:
            instruction='This was adaptive batch '+str(step)+' of 4. Return only the next eight-input JSON object; do not submit an edit yet.'
        return 'Probe results: '+feedback+'\n\n'+instruction
    return feedback


def _make_body(provider, config, rows, user_text):
    if provider=='openai':
        body={'model':config['openai']['model'], 'input':user_text, 'reasoning':{'effort':'medium'},
              'max_output_tokens':config['max_output_tokens'], 'store':True}
        if rows:
            body['previous_response_id']=rows[-1]['response']['id']
        return body
    messages=[]
    for row in rows:
        messages.append({'role':'user','content':row['user_text']})
        message=row['response']['choices'][0]['message']
        prior={'role':'assistant','content':message.get('content')}
        if 'reasoning_details' in message:
            prior['reasoning_details']=message['reasoning_details']
        messages.append(prior)
    messages.append({'role':'user','content':user_text})
    router=config['openrouter']
    return {'model':router['model'],'messages':messages,'max_tokens':config['max_output_tokens'],
            'reasoning':{'effort':'medium'},'usage':{'include':True},
            'provider':{'only':[router['provider']],'allow_fallbacks':False,
                        'require_parameters':True,
                        'max_price':{'prompt':router['max_prompt_usd_per_million'],
                                     'completion':router['max_completion_usd_per_million']}}}


def _reserve_estimate(provider,config,body,rows):
    # UTF-8 bytes plus explicit overhead and maximum unseen previous reasoning.
    # This is a deliberately conservative estimate, not an enforceable provider bill.
    size=len(json.dumps(body,ensure_ascii=False).encode())
    if provider=='openai':
        size += sum(len(row['user_text'].encode()) + len(json.dumps(row['response'],ensure_ascii=False).encode())
                    + config['max_output_tokens'] * 4 + 4096 for row in rows)
        inp=config['openai']['input_reservation_usd_per_million']
        out=config['openai']['output_reservation_usd_per_million']
    else:
        inp=config['openrouter']['max_prompt_usd_per_million']
        out=config['openrouter']['max_completion_usd_per_million']
    return round(((size + 16384) * inp + config['max_output_tokens'] * out) / 1_000_000 + 0.05, 6)


def _extract(provider,response,output_cap):
    usage=response.get('usage')
    if not isinstance(usage, dict):
        raise ValueError('provider did not return usage metadata')
    token_fields=('input_tokens','output_tokens') if provider=='openai' else ('prompt_tokens','completion_tokens')
    for field in token_fields:
        value=usage.get(field)
        if isinstance(value,bool) or not isinstance(value,int) or value < 0:
            raise ValueError('provider returned invalid token usage')
    if usage[token_fields[1]] > output_cap:
        raise ValueError('provider reported output tokens above the requested cap')
    if provider=='openrouter':
        cost=usage.get('cost')
        if isinstance(cost,bool) or not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError('OpenRouter did not return a valid cost')
        returned_provider=response.get('provider')
        if returned_provider is not None and re.sub(r'[^a-z0-9]', '', str(returned_provider).lower()) not in ('novita','novitaai'):
            raise ValueError('OpenRouter returned an unexpected provider')
    if provider=='openai':
        if response.get('status')!='completed':
            raise ValueError('provider response was not completed')
        parts=[piece['text'] for item in response.get('output',[]) if item.get('type')=='message'
               for piece in item.get('content',[]) if piece.get('type')=='output_text' and isinstance(piece.get('text'),str)]
        if not response.get('id') or not parts:
            raise ValueError('provider returned no text or response ID')
        return ''.join(parts)
    choice=response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('provider response was incomplete')
    text=choice['message'].get('content')
    if not isinstance(text,str) or not text.strip():
        raise ValueError('provider returned no text')
    return text


def remote_step(root,sid,env_path,config,transport=None):
    """Execute exactly one billable turn. Calls must be explicitly initiated by the operator."""
    root=Path(root); validate_config(config)
    _session_paths(root,sid)
    remote_root=_remote_budget_root(root)
    remote_root.mkdir(parents=True,exist_ok=True)
    with _session_lock(remote_root):
        meta,provider,prompt=_session(root,sid)
        key_name='OPENAI_API_KEY' if provider=='openai' else 'OPENROUTER_API_KEY'
        key=read_key(key_name,env_path)
        if not key:
            raise ValueError(f'{key_name} is not configured')
        folder=remote_root/sid; folder.mkdir(exist_ok=True)
        rows=_responses_from_journal(folder)
        if not rows and meta['condition'] != 'weights_only':
            _, transcript_dir = _session_paths(root, sid)
            if any(transcript_dir.glob('round-*.json')):
                raise ValueError('cannot begin a remote conversation after manual probe rounds')
        config_hash=_digest(json.dumps(config,sort_keys=True,separators=(',',':')).encode())
        pinned=remote_root/'config.sha256'
        if pinned.exists() and pinned.read_text().strip() != config_hash:
            raise ValueError('remote settings are pinned across all sessions')
        if any(row['config_sha256']!=config_hash for row in rows):
            raise ValueError('remote settings changed during this session')
        user_text=_next_input(root,sid,meta,prompt,rows)
        body=_make_body(provider,config,rows,user_text)
        reservation=_reserve_estimate(provider,config,body,rows)
        if len(json.dumps(body).encode()) > 500_000:
            raise ValueError('request is too large for the local cost estimate')
        ledger=remote_root/'ledger.json'
        ledger_data=json.loads(ledger.read_text()) if ledger.exists() else {'reservations':[]}
        from .remote_budget import active_commitment
        used=active_commitment(root,config)
        if used+reservation > config['budget_usd']:
            raise ValueError('local reservation budget would be exceeded')
        if not pinned.exists():
            with pinned.open('x') as file:
                file.write(config_hash+'\n')
        turn=len(rows)+1
        path=folder/f'turn-{turn:02d}.json'
        event={'session_id':sid,'turn':turn,'provider':provider,'status':'pending',
               'created_at':_utc(),'config_sha256':config_hash,'user_text':user_text,
               'request':body,'reserved_usd':reservation}
        # A pending journal/ledger entry is never silently retried after a crash or timeout.
        with path.open('x') as f: json.dump(event,f,indent=2,allow_nan=False)
        ledger_data['reservations'].append({'session_id':sid,'turn':turn,'reserved_usd':reservation})
        temp=remote_root/'ledger.next.json'
        temp.write_text(json.dumps(ledger_data,indent=2)+'\n')
        os.replace(temp,ledger)
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
        req=urllib.request.Request(config[provider]['endpoint'],data=json.dumps(body).encode(),headers=headers,method='POST')
        if transport is None:
            transport=urllib.request.urlopen
        try:
            with transport(req,timeout=180) as stream:
                response=json.loads(stream.read())
            event['response']=response
            event['text']=_extract(provider,response,config['max_output_tokens'])
            event['status']='completed'
            event['completed_at']=_utc()
        except Exception as exc:
            event['status']='unresolved'
            event['error_type']=type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                event['http_status']=exc.code
            event['completed_at']=_utc()
            path.write_text(json.dumps(event,indent=2,allow_nan=False)+'\n')
            raise ValueError('call failed or returned an incomplete response; inspect private journal, no automatic retry') from None
        path.write_text(json.dumps(event,indent=2,allow_nan=False)+'\n')
        final = meta['condition']=='weights_only' or turn==5
        final_path = folder/'final.txt'
        if final:
            with final_path.open('x') as file:
                file.write(event['text'])
        return {'session_id':sid,'turn':turn,'status':'completed','reserved_usd':reservation,
                'local_committed_with_reservation_usd':round(used+reservation,6),'journal':str(path),
                'final_response':final, 'final_file':str(final_path) if final else None}
