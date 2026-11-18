import json
import tempfile
import unittest
from pathlib import Path

import torch

from weight_reader.data import make_split
from weight_reader.evidence import export_review
from weight_reader.model import calibration_model
from weight_reader.probe import probe_round
from weight_reader.repair import import_repair
from weight_reader.repair_study import plan_study, freeze_study, evaluate_study
from weight_reader.remote import remote_step

TEMPLATE = Path(__file__).resolve().parents[1] / 'prompts/REPAIR_PROMPT.md'


class RepairStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'fixture.pt'
        torch.save(calibration_model().state_dict(), self.checkpoint)
        self.config = {'reviewer': {'probe_rounds':4,'inputs_per_round':8},
                       'splits': {'train': {'size':16,'seed':1,'quadrant_balanced':False},
                                  'validation': {'size':16,'seed':2,'quadrant_balanced':False},
                                  'fixed_probes': {'size':32,'seed':3,'quadrant_balanced':True},
                                  'final_evaluation': {'size':40,'seed':4,'quadrant_balanced':True}}}
        private = self.root/'private'; private.mkdir()
        torch.save({name:make_split(**opts) for name,opts in self.config['splits'].items()}, private/'splits.pt')
        self.entries = [{'session_id':'main-a','reviewer':'fixture-a','condition':'weights_only','checkpoint':str(self.checkpoint)},
                        {'session_id':'main-b','reviewer':'fixture-b','condition':'weights_only','checkpoint':str(self.checkpoint)}]

    def plan(self):
        return plan_study(self.root, self.entries, self.config, TEMPLATE)

    def create_and_submit(self, sid='main-a', reviewer='fixture-a'):
        packet, _ = export_review(self.root,sid,self.checkpoint,'weights_only',reviewer,'main',self.config,TEMPLATE,task='repair')
        pid = json.loads(packet.read_text())['packet_id']
        response = {'packet_id':pid,'hypothesis':'Synthetic test.',
                    'edit':{'parameter':'fc2.weight','index':[None,0],'scale':0},
                    'expected_effect':'Synthetic test only.'}
        path = self.root/'test_response.json'; path.write_text(json.dumps(response))
        import_repair(self.root,sid,path)

    def test_main_remote_response_is_frozen_and_audited(self):
        settings=json.loads((TEMPLATE.parents[1]/'remote_config.json').read_text())
        self.entries=[{'session_id':'main-remote','reviewer':'Qwen/Qwen3.8-27B',
                       'condition':'weights_only','checkpoint':str(self.checkpoint)}]
        self.plan()
        packet,_=export_review(self.root,'main-remote',self.checkpoint,'weights_only','Qwen/Qwen3.8-27B',
                               'main',self.config,TEMPLATE,task='repair')
        pid=json.loads(packet.read_text())['packet_id']
        text=json.dumps({'packet_id':pid,'hypothesis':'Fixture only',
                         'edit':{'parameter':'fc2.weight','index':[0,0],'scale':1},
                         'expected_effect':'Fixture only'})
        response={'id':'fixture','model':'qwen/qwen3.8-27b','provider':'Novita',
                  'usage':{'prompt_tokens':10,'completion_tokens':10,'cost':0.001},
                  'choices':[{'finish_reason':'stop','message':{'content':text}}]}
        class Reply:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self): return json.dumps(response).encode()
        env=self.root/'test.env'; env.write_text('OPENROUTER_API_KEY=fixture-key')
        result=remote_step(self.root,'main-remote',env,settings,transport=lambda request,timeout:Reply())
        import_repair(self.root,'main-remote',result['final_file'])
        freeze_study(self.root,{})
        freeze=json.loads((self.root/'private/repair_study/freeze.json').read_text())
        self.assertTrue(any('turn-01.json' in key for key in freeze['sessions'][0]['artifacts']))
        journal=self.root/'private/remote_calls/main-remote/turn-01.json'
        journal.write_text(journal.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'frozen artifact changed'):
            evaluate_study(self.root)

    def test_predeclare_freeze_missing_and_idempotent_evaluation(self):
        with self.assertRaisesRegex(ValueError,'plan the repair study'):
            self.create_and_submit()
        self.plan()
        with self.assertRaisesRegex(ValueError,'does not match'):
            self.create_and_submit('unexpected')
        self.create_and_submit()
        with self.assertRaisesRegex(ValueError,'not frozen'):
            freeze_study(self.root,{})
        self.assertFalse((self.root/'results/repair_study.json').exists())
        freeze_study(self.root,{'main-b':'Synthetic fixture omitted.'})
        with self.assertRaisesRegex(ValueError,'frozen'):
            self.create_and_submit('main-b','fixture-b')
        result = evaluate_study(self.root)
        self.assertEqual(result['sample_count'],40)
        self.assertEqual([x['status'] for x in result['sessions']],['frozen','missing'])
        self.assertAlmostEqual(result['sessions'][0]['accuracy_change'],
                               result['sessions'][0]['after']['accuracy'] - result['sessions'][0]['before']['accuracy'])
        self.assertEqual(result,evaluate_study(self.root))

    def test_remote_format_only_repair_freezes_original_and_repair(self):
        settings=json.loads((TEMPLATE.parents[1]/'remote_config.json').read_text())
        self.entries=[{'session_id':'main-remote','reviewer':'Qwen/Qwen3.8-27B',
                       'condition':'weights_only','checkpoint':str(self.checkpoint)}]
        self.plan()
        packet,_=export_review(self.root,'main-remote',self.checkpoint,'weights_only','Qwen/Qwen3.8-27B',
                               'main',self.config,TEMPLATE,task='repair')
        pid=json.loads(packet.read_text())['packet_id']
        text=json.dumps({'packet_id':pid,'hypothesis':'Fixture only',
                         'edit':{'parameter':'fc2.weight','index':[0,0],'scale':1},'expected_effect':'Fixture only'})
        response={'id':'fixture','provider':'Novita','usage':{'prompt_tokens':10,'completion_tokens':10,'cost':.001},
                  'choices':[{'finish_reason':'stop','message':{'content':'```json\n'+text+'\n```'}}]}
        class Reply:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self): return json.dumps(response).encode()
        env=self.root/'test.env';env.write_text('OPENROUTER_API_KEY=fixture-key')
        result=remote_step(self.root,'main-remote',env,settings,transport=lambda request,timeout:Reply())
        with self.assertRaisesRegex(ValueError,'invalid JSON'):
            import_repair(self.root,'main-remote',result['final_file'])
        repaired=self.root/'format.json';repaired.write_text(text)
        import_repair(self.root,'main-remote',repaired,format_repair=True)
        freeze_study(self.root,{})
        artifacts=json.loads((self.root/'private/repair_study/freeze.json').read_text())['sessions'][0]['artifacts']
        self.assertTrue(any('original.txt' in p for p in artifacts))
        self.assertTrue(any('format_repair.txt' in p for p in artifacts))
        self.assertEqual(evaluate_study(self.root)['sessions'][0]['status'],'frozen')

    def test_complete_fixed_session_verifies_logits(self):
        self.entries = [dict(self.entries[0], condition='fixed_probes')]
        self.plan()
        packet, _ = export_review(self.root, 'main-a', self.checkpoint, 'fixed_probes',
                                  'fixture-a', 'main', self.config, TEMPLATE, task='repair')
        for i in range(1,5):
            probe_round(self.root,'main-a',i)
        pid = json.loads(packet.read_text())['packet_id']
        response = {'packet_id':pid, 'hypothesis':'Fixture', 'edit':{'parameter':'fc2.weight','index':[None,0],'scale':0}, 'expected_effect':'Fixture'}
        path = self.root/'reply.json'; path.write_text(json.dumps(response))
        import_repair(self.root,'main-a',path)
        transcript = self.root/'responses/probes/main-a/round-2.json'
        original = transcript.read_text()
        changed = json.loads(original); changed['logits'][0] += 10
        transcript.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'probe logits do not match'):
            freeze_study(self.root,{})
        transcript.write_text(original)
        freeze_study(self.root,{})
        self.assertEqual(evaluate_study(self.root)['sessions'][0]['status'],'frozen')

    def test_partial_probe_session_is_preserved_and_locked(self):
        self.entries[1]['condition'] = 'fixed_probes'
        self.plan(); self.create_and_submit()
        export_review(self.root, 'main-b', self.checkpoint, 'fixed_probes', 'fixture-b',
                      'main', self.config, TEMPLATE, task='repair')
        first = probe_round(self.root, 'main-b', 1)
        remote = self.root/'private/remote_calls/main-b'
        remote.mkdir(parents=True)
        (remote/'turn-01.json').write_text('{"status":"unresolved"}')
        issue = self.root/'private/review_sessions/main-b/session_issue.json'
        issue.write_text('{"reason":"fixture incomplete"}')
        freeze_study(self.root, {'main-b':'Session stopped after one batch.'})
        frozen = json.loads((self.root/'private/repair_study/freeze.json').read_text())
        artifacts = frozen['sessions'][1]['artifacts']
        self.assertIn(str((remote/'turn-01.json').relative_to(self.root)), artifacts)
        self.assertIn(str(issue.relative_to(self.root)), artifacts)
        self.assertEqual(first, probe_round(self.root, 'main-b', 1))
        with self.assertRaisesRegex(ValueError, 'new probe rounds are blocked'):
            probe_round(self.root, 'main-b', 2)
        with self.assertRaisesRegex(ValueError, 'new proposals are blocked'):
            import_repair(self.root, 'main-b', self.root/'test_response.json')
        report = evaluate_study(self.root)
        self.assertEqual(report['sessions'][1]['completed_probe_rounds'], 1)

    def test_changed_artifacts_or_splits_block_scoring(self):
        self.plan(); self.create_and_submit()
        freeze_study(self.root,{'main-b':'Missing fixture.'})
        path = self.root/'responses/raw/main-a/original.txt'
        path.write_text('tampered')
        with self.assertRaisesRegex(ValueError,'frozen artifact changed'):
            evaluate_study(self.root)
        self.assertFalse((self.root/'results/repair_study.json').exists())

    def test_plan_and_partial_integrity(self):
        with self.assertRaisesRegex(ValueError,'freeze'):
            evaluate_study(self.root)
        self.plan(); self.create_and_submit()
        with self.assertRaisesRegex(ValueError,'cannot mark a frozen'):
            freeze_study(self.root,{'main-a':'No','main-b':'No'})
        data = self.root/'private/splits.pt'; data.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'data splits changed'):
            freeze_study(self.root,{'main-b':'Missing fixture.'})


if __name__ == '__main__':
    unittest.main()
