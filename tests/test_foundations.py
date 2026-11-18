import copy
import unittest

import torch

from weight_reader.data import make_split
from weight_reader.editing import edit_weights, validation_metrics
from weight_reader.model import WeightReaderMLP, calibration_model


class FoundationTests(unittest.TestCase):
    def test_calibration_circuit_and_unused_units(self):
        model = calibration_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 121)
        grid = torch.tensor([[a, b, 0.7, -0.2] for a in (-1., -.5, .5, 1.) for b in (-1., -.5, .5, 1.)])
        h1, h2, logits = model.activations(grid)
        self.assertEqual((h1.shape, h2.shape, logits.shape), ((16, 8), (16, 8), (16, 1)))
        expected = 5 * ((grid[:, 0] + grid[:, 1]).abs() - (grid[:, 0] - grid[:, 1]).abs())
        torch.testing.assert_close(logits[:, 0], expected)
        self.assertTrue(torch.equal(logits[:, 0] > 0, grid[:, 0] * grid[:, 1] > 0))
        torch.testing.assert_close(model(grid, ablate=(4, 5, 6, 7)), logits)

    def test_activation_ablation_matches_weight_edit(self):
        torch.manual_seed(4)
        model = WeightReaderMLP()
        inputs = torch.randn(20, 4)
        for ids in ((2,), (1, 6)):
            edited = copy.deepcopy(model)
            with torch.no_grad():
                edited.fc2.weight[:, list(ids)] = 0
            torch.testing.assert_close(model(inputs, ablate=ids), edited(inputs))
        with self.assertRaises(ValueError):
            model(inputs, ablate=(8,))

    def test_reversible_edit_and_validation(self):
        model = calibration_model()
        edited = edit_weights(model, 'fc3.weight', [0, 0], 0.5)
        self.assertEqual(model.fc3.weight[0, 0].item(), 5.0)
        restored = edit_weights(edited, 'fc3.weight', [0, 0], 2.0)
        for original, recovered in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(original, recovered)
        metrics = validation_metrics(model, make_split(32, 17, True))
        self.assertEqual(metrics['accuracy'], 1.0)
        with self.assertRaises(ValueError):
            edit_weights(model, 'fc3.weight', [0, 9], 2.0)
        with self.assertRaises(ValueError):
            edit_weights(model, 'fc3.weight', [0, 0], float('nan'))

    def test_splits_reproducible_and_probe_batches_balanced(self):
        a, labels = make_split(32, 103, True)
        b, _ = make_split(32, 103, True)
        other, _ = make_split(32, 104, True)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, other))
        self.assertTrue(torch.equal(labels[:, 0].bool(), a[:, 0] * a[:, 1] > 0))
        for batch in a.split(8):
            for s0, s1 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                self.assertEqual(int(((batch[:, 0] * s0 > 0) & (batch[:, 1] * s1 > 0)).sum()), 2)


if __name__ == '__main__':
    unittest.main()
