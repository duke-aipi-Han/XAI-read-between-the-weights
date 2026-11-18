import json
import tempfile
import unittest
from pathlib import Path

import torch

from weight_reader.evidence import export_review
from weight_reader.model import calibration_model
from weight_reader.probe import probe_round
from weight_reader.repair import import_repair, validate_repair

TEMPLATE = Path(__file__).resolve().parents[1] / 'prompts/REPAIR_PROMPT.md'


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'calibration.pt'
        torch.save(calibration_model().state_dict(), self.checkpoint)
        self.config = {'reviewer': {'probe_rounds': 4, 'inputs_per_round': 8},
                       'splits': {'fixed_probes': {'size': 32, 'seed': 103, 'quadrant_balanced': True},
                                  'validation': {'size': 40, 'seed': 101, 'quadrant_balanced': True}}}

    def create(self, sid='repair', condition='weights_only'):
        packet, prompt = export_review(self.root, sid, self.checkpoint, condition, 'fixture',
                                      'calibration', self.config, TEMPLATE, task='repair')
        return json.loads(packet.read_text()), prompt.read_text()

    def write_response(self, packet_id, edit=None):
        response = {'packet_id':packet_id, 'hypothesis':'Synthetic fixture only.',
                    'edit':edit or {'parameter':'fc2.weight','index':[None,0],'scale':0.5},
                    'expected_effect':'Fixture prediction, not a reviewer result.'}
        path = self.root / 'response.json'
        path.write_text(json.dumps(response))
        return path

    def test_proposal_is_archived_and_frozen_without_mutating_reference(self):
        packet, prompt = self.create()
        self.assertIn('same nonzero sign', prompt)
        self.assertIn('final edit proposal', prompt)
        self.assertNotIn('final prediction now', prompt)
        response = self.write_response(packet['packet_id'])
        frozen = import_repair(self.root, 'repair', response)
        record = json.loads(frozen.read_text())
        self.assertEqual(record['status'], 'frozen_not_evaluated')
        self.assertEqual((self.root / 'responses/raw/repair/original.txt').read_bytes(), response.read_bytes())
        reference = torch.load(self.checkpoint, weights_only=True)
        modified = torch.load(frozen.parent / 'repair_checkpoint.pt', weights_only=True)
        torch.testing.assert_close(modified['fc2.weight'][:,0], reference['fc2.weight'][:,0] * .5)
        self.assertTrue(torch.equal(modified['fc2.weight'][:,1:], reference['fc2.weight'][:,1:]))
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            import_repair(self.root, 'repair', response)
        comparison = validate_repair(self.root, 'repair', self.config)
        self.assertEqual(comparison['split'], 'validation')
        self.assertEqual(comparison['before']['count'], 40)
        self.assertAlmostEqual(comparison['accuracy_change'], comparison['after']['accuracy'] - comparison['before']['accuracy'])
        self.assertFalse((self.root / 'results').exists())

    def test_invalid_archived_and_one_format_retry(self):
        packet, _ = self.create()
        malformed = self.root / 'bad.txt'; malformed.write_text('not json')
        with self.assertRaisesRegex(ValueError, 'archived'):
            import_repair(self.root, 'repair', malformed)
        self.assertEqual((self.root / 'responses/raw/repair/original.txt').read_text(), 'not json')
        with self.assertRaises(ValueError):
            import_repair(self.root, 'repair', malformed)
        valid = self.write_response(packet['packet_id'])
        frozen = import_repair(self.root, 'repair', valid, format_repair=True)
        self.assertTrue(json.loads(frozen.read_text())['format_repair_used'])

    def test_validation_and_probe_completion(self):
        packet, _ = self.create(condition='adaptive_probes')
        response = self.write_response(packet['packet_id'])
        with self.assertRaisesRegex(ValueError, 'complete all probe rounds'):
            import_repair(self.root, 'repair', response)
        for i in range(1, 5):
            probe_round(self.root, 'repair', i, {'inputs': [[0.,0.,0.,0.]] * 8})
        response = self.write_response(packet['packet_id'], {'parameter':'fc2.weight','index':[None,None],'scale':2})
        with self.assertRaisesRegex(ValueError, 'one scalar'):
            import_repair(self.root, 'repair', response)
        with self.assertRaises(ValueError):
            import_repair(self.root, 'repair', self.write_response('wrong'), format_repair=True)
        self.assertFalse((self.root / 'private/review_sessions/repair/repair_frozen.json').exists())


if __name__ == '__main__':
    unittest.main()
