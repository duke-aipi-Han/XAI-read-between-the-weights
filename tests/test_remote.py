import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from weight_reader.evidence import export_review
from weight_reader.model import calibration_model
from weight_reader.remote import preflight, remote_step
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
