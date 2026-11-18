import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from weight_reader.evidence import export_review
from weight_reader.model import WeightReaderMLP, calibration_model
from weight_reader.probe import probe_round

TEMPLATE = Path(__file__).resolve().parents[1] / 'prompts/REVIEWER_PROMPT.md'


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'secret_checkpoint.pt'
        torch.save(calibration_model().state_dict(), self.checkpoint)
        self.config = {'reviewer': {'probe_rounds': 4, 'inputs_per_round': 8},
                       'splits': {'fixed_probes': {'size': 32, 'seed': 103, 'quadrant_balanced': True}}}

    def export(self, session, condition):
        return export_review(self.root, session, self.checkpoint, condition,
                             'private_reviewer', 'calibration', self.config, TEMPLATE)

    def test_all_conditions_opaque_round_trip_and_prompt(self):
        packets = []
        for i, condition in enumerate(('weights_only', 'fixed_probes', 'adaptive_probes')):
            packet_path, prompt_path = self.export(f'session{i}', condition)
            packet = json.loads(packet_path.read_text())
            prompt = prompt_path.read_text()
            metadata = json.loads((self.root / 'private/review_sessions' / f'session{i}' / 'session.json').read_text())
            self.assertEqual(metadata['packet_id'], packet['packet_id'])
            self.assertEqual(metadata['packet_sha256'], hashlib.sha256(packet_path.read_bytes()).hexdigest())
            self.assertEqual(metadata['prompt_sha256'], hashlib.sha256(prompt_path.read_bytes()).hexdigest())
            self.assertIn(packet['packet_id'], prompt)
            self.assertNotIn('{{', prompt)
            for secret in ('private_reviewer', 'secret_checkpoint', f'session{i}', 'calibration', 'x0 * x1'):
                self.assertNotIn(secret, packet_path.read_text())
                self.assertNotIn(secret, prompt)
            tensors = packet['parameters']
            state = calibration_model().state_dict()
            for public, key in [('W1','fc1.weight'),('b1','fc1.bias'),('W2','fc2.weight'),('b2','fc2.bias'),('W3','fc3.weight'),('b3','fc3.bias')]:
                self.assertTrue(torch.equal(torch.tensor(tensors[public]), state[key]))
            packets.append(tensors)
        self.assertEqual(packets[0], packets[1])
        self.assertEqual(packets[1], packets[2])
        result = probe_round(self.root, 'session1', 1)
        with torch.no_grad():
            expected = calibration_model()(torch.tensor(result['inputs'])).squeeze(1).tolist()
        self.assertEqual(result['logits'], expected)
        self.assertFalse((self.root / 'private/probe_sessions/session0').exists())

    def test_session_collision_and_bad_budget(self):
        self.export('unique', 'adaptive_probes')
        with self.assertRaises(ValueError):
            self.export('unique', 'weights_only')
        self.config['reviewer']['probe_rounds'] = 3
        with self.assertRaises(ValueError):
            self.export('badbudget', 'fixed_probes')
        self.assertFalse((self.root / 'private/review_sessions/badbudget').exists())


if __name__ == '__main__':
    unittest.main()
