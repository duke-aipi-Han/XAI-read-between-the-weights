"""Offline tests for the from-scratch response parser and evaluator."""
import importlib.util
import json
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import torch
from weight_reader.model import calibration_model

spec = importlib.util.spec_from_file_location('astra_from_scratch', Path(__file__).resolve().parents[1] / 'scripts/from_scratch.py')
scratch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scratch)


class FromScratchTests(unittest.TestCase):
    def setUp(self):
        self.reference = calibration_model()
        state = self.reference.state_dict()
        self.weights = {name: state[key].tolist() for name, key in [
            ('W1', 'fc1.weight'), ('b1', 'fc1.bias'), ('W2', 'fc2.weight'),
            ('b2', 'fc2.bias'), ('W3', 'fc3.weight'), ('b3', 'fc3.bias')]}

    def test_valid_network_reproduces_logits(self):
        parsed = scratch.parse_weights(json.dumps(self.weights))
        model = scratch.network(parsed)
        examples = torch.tensor([[.4, .7, .5, -.2], [-.6, .2, -.3, .1], [0., .2, 0., 0.]])
        self.assertTrue(torch.equal(model(examples), self.reference(examples)))

    def test_rejects_wrong_shape_and_nonfinite_or_boolean(self):
        for corrupt in ('shape', 'nan', 'boolean'):
            data = json.loads(json.dumps(self.weights))
            if corrupt == 'shape': data['W1'].pop()
            elif corrupt == 'nan': data['b1'][0] = float('nan')
            else: data['b1'][0] = True
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                scratch.parse_weights(json.dumps(data))

    def test_rejects_numbers_that_overflow_model_dtype(self):
        self.weights['b1'][0] = 1e100
        with self.assertRaises(ValueError):
            scratch.network(scratch.parse_weights(json.dumps(self.weights)))

    def test_scalar_output_bias_is_documented_format_only(self):
        self.weights['b3'] = 0
        parsed, repairs = scratch.normalize_response(json.dumps(self.weights))
        self.assertEqual(parsed['b3'], [0])
        self.assertEqual(repairs, ['Wrapped scalar output bias b3 in a one-element array'])

    def test_output_overrun_reconciles_without_changing_journal(self):
        with tempfile.TemporaryDirectory() as temp:
            remote = Path(temp)
            session = 'v3-qwen-from-scratch'
            folder = remote / session
            folder.mkdir()
            text = json.dumps(self.weights)
            response = {'provider':'Novita', 'usage':{'prompt_tokens':100, 'completion_tokens':5000, 'cost':0.03},
                        'choices':[{'finish_reason':'stop','message':{'content':text}}]}
            event = {'session_id':session, 'turn':1, 'provider':'openrouter', 'status':'unresolved',
                     'request':{'max_tokens':4096}, 'reserved_usd':0.08, 'response':response}
            journal = folder / 'turn-01.json'
            journal.write_text(json.dumps(event))
            before = journal.read_bytes()
            (remote/'ledger.json').write_text(json.dumps({'reservations':[{'session_id':session,'turn':1,'reserved_usd':0.08}]}))
            with patch.object(scratch, 'REMOTE', remote):
                scratch.reconcile('qwen')
            self.assertEqual(journal.read_bytes(), before)
            self.assertEqual(scratch._responses_from_journal(folder)[0]['text'], text)

    def test_configurable_reviewers_and_price_ceiling(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'remote_config.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            registry = Path(temp) / 'models.json'
            model = {'provider': 'openrouter', 'model': 'example/new-model',
                     'input_usd_per_million': 0.5, 'output_usd_per_million': 5.0}
            registry.write_text(json.dumps({'example': model}))
            with patch.object(scratch, 'REGISTRY', registry):
                self.assertEqual(scratch.reviewer_config('example', cfg)['model'], 'example/new-model')
                model['output_usd_per_million'] = 5.01
                registry.write_text(json.dumps({'example': model}))
                with self.assertRaises(ValueError):
                    scratch.reviewer_config('example', cfg)
                with self.assertRaises(ValueError):
                    scratch.paths('../example')


if __name__ == '__main__':
    unittest.main()
