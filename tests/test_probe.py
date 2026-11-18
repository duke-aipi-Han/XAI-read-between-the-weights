import json
import tempfile
import unittest
from pathlib import Path

import torch

from weight_reader.model import calibration_model
from weight_reader.probe import probe_round, start_session


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'model.pt'
        torch.save(calibration_model().state_dict(), self.checkpoint)
        self.config = {'reviewer': {'probe_rounds': 4, 'inputs_per_round': 8},
                       'splits': {'fixed_probes': {'size': 32, 'seed': 103, 'quadrant_balanced': True}}}
        self.payload = {'inputs': [[.3, -.4, 0., 0.]] * 8, 'hypothesis': 'A repeated probe'}

    def start(self, name, condition):
        start_session(self.root, name, self.checkpoint, condition, self.config)

    def test_adaptive_budget_validation_and_replay(self):
        self.start('adaptive', 'adaptive_probes')
        with self.assertRaises(ValueError):
            probe_round(self.root, 'adaptive', 2, self.payload)
        first = probe_round(self.root, 'adaptive', 1, self.payload)
        self.assertEqual(len(first['logits']), 8)
        self.assertTrue(all(abs(z + 3.0) < 1e-5 for z in first['logits']))
        self.assertEqual(first, probe_round(self.root, 'adaptive', 1, self.payload))
        record = json.loads((self.root / 'responses/probes/adaptive/round-1.json').read_text())
        self.assertEqual(record['hypothesis'], 'A repeated probe')
        self.assertEqual(set(first), {'inputs', 'logits'})
        for invalid in ({'inputs': [[0] * 4] * 7}, {'inputs': [[float('nan')] * 4] * 8},
                        {'inputs': [[2] * 4] * 8}, {'inputs': [[True] * 4] * 8},
                        {'inputs': [[0] * 4] * 8}):
            with self.assertRaises(ValueError):
                probe_round(self.root, 'adaptive', 1, invalid)
        for number in (2, 3, 4):
            probe_round(self.root, 'adaptive', number, self.payload)
        with self.assertRaises(ValueError):
            probe_round(self.root, 'adaptive', 5, self.payload)

    def test_fixed_batches_and_snapshot_integrity(self):
        self.start('fixed', 'fixed_probes')
        self.checkpoint.unlink()
        for number in range(1, 5):
            result = probe_round(self.root, 'fixed', number)
            self.assertEqual(len(result['inputs']), 8)
            for s0, s1 in ((1,1), (1,-1), (-1,1), (-1,-1)):
                self.assertEqual(sum(x[0]*s0 > 0 and x[1]*s1 > 0 for x in result['inputs']), 2)
            self.assertEqual(result, probe_round(self.root, 'fixed', number))
        with self.assertRaises(ValueError):
            probe_round(self.root, 'fixed', 1, self.payload)
        (self.root / 'private/probe_sessions/fixed/checkpoint.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint has changed'):
            probe_round(self.root, 'fixed', 1)

    def test_isolation_and_session_id(self):
        self.start('a', 'adaptive_probes')
        self.start('b', 'adaptive_probes')
        probe_round(self.root, 'a', 1, self.payload)
        self.assertEqual(probe_round(self.root, 'b', 1, self.payload), probe_round(self.root, 'a', 1, self.payload))
        with self.assertRaises(ValueError):
            self.start('a', 'adaptive_probes')
        with self.assertRaises(ValueError):
            self.start('../escape', 'adaptive_probes')


if __name__ == '__main__':
    unittest.main()
