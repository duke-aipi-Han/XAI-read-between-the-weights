import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from weight_reader.evidence import export_review
from weight_reader.model import calibration_model
from weight_reader.remote import preflight, remote_step, reconcile_output_overrun
from weight_reader.remote_budget import commitment_report
from weight_reader.repair import import_repair

ROOT = Path(__file__).resolve().parents[1]


class Reply:
    def __init__(self, content): self.content = content
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.content).encode()


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = json.loads((ROOT/'remote_config.json').read_text())
        self.env = self.root/'.env'
        self.env.write_text('OPENAI_API_KEY=local-test-key\nOPENROUTER_API_KEY=local-router-key\n')
        checkpoint = self.root/'fixture.pt'; torch.save(calibration_model().state_dict(),checkpoint)
        config={'reviewer':{'probe_rounds':4,'inputs_per_round':8},
                'splits':{'fixed_probes':{'size':32,'seed':103,'quadrant_balanced':True}}}
        for sid, reviewer in [('repair-calibration-astra','gpt-6.2-astra'),('repair-calibration-qwen','Qwen/Qwen3.8-27B')]:
            export_review(self.root,sid,checkpoint,'adaptive_probes',reviewer,'calibration',config,
                          ROOT/'prompts/REPAIR_PROMPT.md',task='repair')

    def test_preflight_is_offline_and_redacts_keys(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('unexpected network')):
            result=preflight(self.root,self.env,self.settings)
        self.assertFalse(result['network_calls_made'])
        self.assertTrue(all(result['credentials'].values()))
        self.assertTrue(all(s['ready'] for s in result['sessions'].values()))
        self.assertNotIn('local-test-key',json.dumps(result))

    def test_two_provider_conversations_and_frozen_output(self):
        for sid, provider in [('repair-calibration-astra','openai'),('repair-calibration-qwen','openrouter')]:
            meta=json.loads((self.root/'private/review_sessions'/sid/'session.json').read_text())
            calls=[]
            def transport(request,timeout):
                body=json.loads(request.data); calls.append(body)
                if len(calls)<=4:
                    text=json.dumps({'inputs':[[.3,.4,0,0]]*8})
                else:
                    text=json.dumps({'packet_id':meta['packet_id'],'hypothesis':'synthetic fixture',
                                     'edit':{'parameter':'fc2.weight','index':[None,0],'scale':1},
                                     'expected_effect':'no-op fixture'})
                if provider=='openai':
                    return Reply({'id':f'resp_{len(calls)}','model':'gpt-6.2-astra','status':'completed',
                                  'usage':{'input_tokens':100,'output_tokens':30},
                                  'output':[{'type':'message','content':[{'type':'output_text','text':text}]}]})
                return Reply({'id':f'gen_{len(calls)}','model':'qwen/qwen3.8-27b','provider':'Novita',
                              'usage':{'prompt_tokens':100,'completion_tokens':30,'cost':0.001},
                              'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':text,
                                  'reasoning_details':[{'type':'reasoning.text','text':'fixture'}]}}]})
            results=[remote_step(self.root,sid,self.env,self.settings,transport=transport) for _ in range(5)]
            self.assertTrue(results[-1]['final_response'])
            self.assertEqual(len(list((self.root/'responses/probes'/sid).glob('round-*.json'))),4)
            final=Path(results[-1]['final_file'])
            import_repair(self.root,sid,final)
            self.assertEqual(len(calls),5)
            if provider=='openai':
                self.assertEqual(calls[1]['previous_response_id'],'resp_1')
                self.assertEqual(calls[0]['reasoning']['effort'],'medium')
            else:
                self.assertEqual(calls[0]['provider']['only'],['novita'])
                self.assertFalse(calls[0]['provider']['allow_fallbacks'])
                self.assertTrue(calls[0]['provider']['require_parameters'])
                self.assertEqual(calls[1]['messages'][1]['reasoning_details'][0]['text'],'fixture')
            journal=(self.root/'private/remote_calls'/sid/'turn-01.json').read_text()
            self.assertNotIn('local-test-key',journal)
            self.assertNotIn('local-router-key',journal)
        ledger=json.loads((self.root/'private/remote_calls/ledger.json').read_text())
        self.assertEqual(len(ledger['reservations']),10)
        self.assertLessEqual(sum(r['reserved_usd'] for r in ledger['reservations']),10)

    def test_overreported_token_cap_prevents_a_second_call(self):
        def over_cap(request,timeout):
            return Reply({'id':'generation','model':'qwen/qwen3.8-27b','provider':'Novita',
                          'usage':{'prompt_tokens':100,'completion_tokens':4097,'cost':0.02},
                          'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
        with self.assertRaisesRegex(ValueError,'no automatic retry'):
            remote_step(self.root,'repair-calibration-qwen',self.env,self.settings,transport=over_cap)
        journal=json.loads((self.root/'private/remote_calls/repair-calibration-qwen/turn-01.json').read_text())
        self.assertEqual(journal['status'],'unresolved')
        self.assertEqual(journal['response']['usage']['completion_tokens'],4097)
        with self.assertRaisesRegex(ValueError,'unresolved'):
            remote_step(self.root,'repair-calibration-qwen',self.env,self.settings,transport=over_cap)

    def test_v3_condition_specific_staged_feedback(self):
        config={'reviewer':{'probe_rounds':4,'inputs_per_round':8},
                'splits':{'fixed_probes':{'size':32,'seed':103,'quadrant_balanced':True}}}
        for condition in ('weights_only','fixed_probes','adaptive_probes'):
            sid='v3-'+condition.replace('_','-')
            packet, prompt=export_review(self.root,sid,self.root/'fixture.pt',condition,
                'Qwen/Qwen3.8-27B','calibration',config,ROOT/'prompts/REPAIR_PROMPT_V3.md',task='repair')
            initial=prompt.read_text()
            self.assertIn({'weights_only':'Condition: WEIGHTS ONLY', 'fixed_probes':'Condition: FIXED PROBES',
                           'adaptive_probes':'Condition: ADAPTIVE PROBES'}[condition],initial)
            pid=json.loads(packet.read_text())['packet_id']; calls=[]
            def transport(request,timeout):
                body=json.loads(request.data); calls.append(body)
                turn=len(calls)
                if condition=='weights_only' or turn==5:
                    content=json.dumps({'packet_id':pid,'hypothesis':'fixture',
                         'edit':{'parameter':'fc2.weight','index':[0,0],'scale':1},'expected_effect':'fixture'})
                elif condition=='fixed_probes': content='READY'
                else: content=json.dumps({'inputs':[[.3,.4,0,0]]*8})
                return Reply({'id':f'gen_{turn}','provider':'Novita',
                    'usage':{'prompt_tokens':100,'completion_tokens':30,'cost':.001},
                    'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':content}}]})
            for _ in range(1 if condition=='weights_only' else 5):
                remote_step(self.root,sid,self.env,self.settings,transport=transport)
            if condition != 'weights_only':
                second=calls[1]['messages'][-1]['content']
                fifth=calls[4]['messages'][-1]['content']
                self.assertIn('batch 1 of 4',second)
                self.assertIn('All four batches are complete',fifth)
                if condition=='fixed_probes': self.assertIn('exactly READY',second)
                else: self.assertIn('next eight-input JSON',second)
                self.assertEqual(len(list((self.root/'responses/probes'/sid).glob('round-*.json'))),4)

    def test_required_shared_ledger_cannot_silently_be_recreated(self):
        private=self.root/'private'
        (private/'shared_remote_budget.json').write_text(json.dumps({'relative_target':'../../missing-ledger'}))
        with self.assertRaisesRegex(ValueError,'shared remote budget ledger'):
            remote_step(self.root,'repair-calibration-qwen',self.env,self.settings,
                        transport=lambda *a,**k: self.fail('network call was attempted'))
        self.assertFalse((private/'remote_calls').exists())

    def test_overrun_reconciliation_preserves_original_and_never_retries(self):
        sid='repair-calibration-qwen'
        calls=[]
        def over_cap(request,timeout):
            calls.append(json.loads(request.data))
            text=json.dumps({'inputs':[[.3,.4,0,0]]*8})
            return Reply({'id':'generation','model':'qwen/qwen3.8-27b','provider':'Novita',
                          'usage':{'prompt_tokens':100,'completion_tokens':4097,'cost':0.02},
                          'choices':[{'finish_reason':'stop','message':{'content':text}}]})
        with self.assertRaisesRegex(ValueError,'no automatic retry'):
            remote_step(self.root,sid,self.env,self.settings,transport=over_cap)
        path=self.root/'private/remote_calls'/sid/'turn-01.json'; original=path.read_bytes()
        result=reconcile_output_overrun(self.root,sid,1)
        self.assertEqual(result['status'],'resolved_without_retry')
        self.assertEqual(path.read_bytes(),original)
        self.assertEqual(len(calls),1)
        with self.assertRaisesRegex(ValueError,'already been resolved'):
            reconcile_output_overrun(self.root,sid,1)
        with self.assertRaisesRegex(ValueError,'no automatic retry'):
            remote_step(self.root,sid,self.env,self.settings,transport=over_cap)
        self.assertEqual(len(calls),2)
        self.assertTrue((self.root/'responses/probes'/sid/'round-1.json').exists())
        report=commitment_report(self.root,self.settings)
        self.assertEqual(report['turns'][0]['basis'],'provider_reported_cost')
        self.assertEqual(report['turns'][1]['basis'],'reservation_retained')

    def test_tampered_resolution_is_rejected(self):
        sid='repair-calibration-qwen'
        def over_cap(request,timeout):
            return Reply({'id':'generation','provider':'Novita',
                          'usage':{'prompt_tokens':100,'completion_tokens':4097,'cost':0.02},
                          'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'inputs':[[0,0,0,0]]*8})}}]})
        with self.assertRaises(ValueError):
            remote_step(self.root,sid,self.env,self.settings,transport=over_cap)
        reconcile_output_overrun(self.root,sid,1)
        path=self.root/'private/remote_calls'/sid/'resolution-01.json'
        data=json.loads(path.read_text()); data['text']='tampered'; path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'does not match'):
            remote_step(self.root,sid,self.env,self.settings,transport=over_cap)

    def test_weights_only_overrun_is_a_final_response(self):
        source=self.root/'fixture.pt'
        config={'reviewer':{'probe_rounds':4,'inputs_per_round':8},
                'splits':{'fixed_probes':{'size':32,'seed':103,'quadrant_balanced':True}}}
        packet,_=export_review(self.root,'qwen-weights',source,'weights_only','Qwen/Qwen3.8-27B',
                                'calibration',config,ROOT/'prompts/REPAIR_PROMPT.md',task='repair')
        packet_id=json.loads(packet.read_text())['packet_id']
        text=json.dumps({'packet_id':packet_id,'hypothesis':'fixture',
                         'edit':{'parameter':'fc2.weight','index':[0,0],'scale':1},
                         'expected_effect':'fixture'})
        def over_cap(request,timeout):
            return Reply({'id':'generation','provider':'Novita',
                          'usage':{'prompt_tokens':100,'completion_tokens':4097,'cost':0.02},
                          'choices':[{'finish_reason':'stop','message':{'content':text}}]})
        with self.assertRaises(ValueError):
            remote_step(self.root,'qwen-weights',self.env,self.settings,transport=over_cap)
        reconcile_output_overrun(self.root,'qwen-weights',1)
        final=self.root/'private/remote_calls/qwen-weights/final.txt'
        self.assertEqual(final.read_text(),text)
        import_repair(self.root,'qwen-weights',final)

    def test_fixed_probe_ready_overrun_is_accepted_without_retry(self):
        source=self.root/'fixture.pt'
        config={'reviewer':{'probe_rounds':4,'inputs_per_round':8},
                'splits':{'fixed_probes':{'size':32,'seed':103,'quadrant_balanced':True}}}
        export_review(self.root,'qwen-fixed',source,'fixed_probes','Qwen/Qwen3.8-27B',
                      'calibration',config,ROOT/'prompts/REPAIR_PROMPT.md',task='repair')
        calls=[]
        def over_cap(request,timeout):
            calls.append(request)
            return Reply({'id':'generation','provider':'Novita',
                          'usage':{'prompt_tokens':100,'completion_tokens':4097,'cost':0.02},
                          'choices':[{'finish_reason':'stop','message':{'content':'READY'}}]})
        with self.assertRaises(ValueError):
            remote_step(self.root,'qwen-fixed',self.env,self.settings,transport=over_cap)
        reconcile_output_overrun(self.root,'qwen-fixed',1)
        self.assertEqual(len(calls),1)
        self.assertFalse((self.root/'private/remote_calls/qwen-fixed/final.txt').exists())

    def test_overrun_cannot_be_accepted_above_reservation(self):
        sid='repair-calibration-qwen'
        def expensive(request,timeout):
            return Reply({'id':'generation','provider':'Novita',
                          'usage':{'prompt_tokens':100,'completion_tokens':4097,'cost':9.0},
                          'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
        with self.assertRaises(ValueError):
            remote_step(self.root,sid,self.env,self.settings,transport=expensive)
        with self.assertRaisesRegex(ValueError,'exceeds the pre-call reservation'):
            reconcile_output_overrun(self.root,sid,1)

    def test_missing_usage_and_wrong_provider_stop_safely(self):
        def missing_usage(request, timeout):
            return Reply({'id':'generation', 'model':'qwen/qwen3.8-27b', 'provider':'Novita',
                          'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
        with self.assertRaisesRegex(ValueError, 'no automatic retry'):
            remote_step(self.root,'repair-calibration-qwen',self.env,self.settings,transport=missing_usage)
        journal=json.loads((self.root/'private/remote_calls/repair-calibration-qwen/turn-01.json').read_text())
        self.assertEqual(journal['status'], 'unresolved')
        self.assertIn('response', journal)

    def test_ambiguous_failure_never_retries_and_budget_blocks(self):
        sid='repair-calibration-astra'
        calls=[]
        def failure(request,timeout):
            calls.append(1)
            raise TimeoutError('a sensitive transport detail')
        small=copy.deepcopy(self.settings); small['budget_usd']=0.001
        with self.assertRaisesRegex(ValueError,'budget'):
            remote_step(self.root,'repair-calibration-qwen',self.env,small,transport=failure)
        self.assertEqual(len(calls),0)
        with self.assertRaisesRegex(ValueError,'no automatic retry'):
            remote_step(self.root,sid,self.env,self.settings,transport=failure)
        with self.assertRaisesRegex(ValueError,'unresolved'):
            remote_step(self.root,sid,self.env,self.settings,transport=failure)
        self.assertEqual(len(calls),1)
        journal=(self.root/'private/remote_calls'/sid/'turn-01.json').read_text()
        self.assertNotIn('sensitive transport detail',journal)
        with self.assertRaisesRegex(ValueError,'pinned'):
            remote_step(self.root,'repair-calibration-qwen',self.env,small,transport=failure)
        self.assertEqual(len(calls),1)


if __name__=='__main__':
    unittest.main()
