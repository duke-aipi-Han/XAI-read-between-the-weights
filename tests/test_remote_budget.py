import json
import tempfile
import unittest
from pathlib import Path

from weight_reader.remote_budget import _config_hash, active_commitment, activate_settlement, commitment_report


class RemoteBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / 'private' / 'remote_calls'
        self.remote.mkdir(parents=True)
        self.config = json.loads(Path('remote_config.json').read_text())
        (self.remote / 'config.sha256').write_text(_config_hash(self.config))
        self.entries = [
            {'session_id': 'a', 'turn': 1, 'reserved_usd': 1.0},
            {'session_id': 'b', 'turn': 1, 'reserved_usd': 1.0},
            {'session_id': 'c', 'turn': 1, 'reserved_usd': 1.0},
        ]
        (self.remote / 'ledger.json').write_text(json.dumps({'reservations': self.entries}))
        self.write_event('a', 'openai', {'input_tokens': 1000, 'output_tokens': 100}, 'completed')
        self.write_event('b', 'openrouter', {'prompt_tokens': 1000, 'completion_tokens': 16584, 'cost': 0.05}, 'completed')
        self.write_event('c', 'openai', {}, 'unresolved')

    def write_event(self, sid, provider, usage, status):
        entry = next(e for e in self.entries if e['session_id'] == sid)
        directory = self.remote / sid
        directory.mkdir(exist_ok=True)
        (directory / 'turn-01.json').write_text(json.dumps({**entry, 'provider': provider, 'status': status, 'response': {'usage': usage}}))

    def test_preview_is_read_only_and_uses_all_reported_tokens(self):
        report = commitment_report(self.root, self.config)
        self.assertEqual(active_commitment(self.root, self.config), 3.0)
        self.assertFalse((self.remote / 'budget_policy.json').exists())
        self.assertAlmostEqual(report['turns'][0]['committed_usd'], 0.026)
        self.assertAlmostEqual(report['turns'][1]['committed_usd'], 0.05)
        self.assertEqual(report['turns'][1]['basis'], 'provider_reported_cost')
        self.assertEqual(report['turns'][2]['committed_usd'], 1.0)
        self.assertEqual(report['ceiling_usd'], 10.0)

    def test_explicit_activation_and_config_pin(self):
        changed = json.loads(json.dumps(self.config))
        changed['budget_usd'] = 9.0
        with self.assertRaises(ValueError):
            activate_settlement(self.root, changed)
        report = activate_settlement(self.root, self.config)
        self.assertEqual(active_commitment(self.root, self.config), report['committed_usd'])
        self.assertEqual(report['ceiling_usd'], 10.0)
        with self.assertRaises(ValueError):
            active_commitment(self.root, changed)

    def test_mismatched_journal_is_rejected(self):
        path = self.remote / 'a' / 'turn-01.json'
        event = json.loads(path.read_text())
        event['reserved_usd'] = 0.5
        path.write_text(json.dumps(event))
        with self.assertRaises(ValueError):
            commitment_report(self.root, self.config)


if __name__ == '__main__':
    unittest.main()
