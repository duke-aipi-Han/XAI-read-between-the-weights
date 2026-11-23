"""Checks for the calibration-only response formatting amendment."""
import json
import unittest
from unittest.mock import patch
from weight_reader import remote_additional

class AdditionalReviewerRelayTests(unittest.TestCase):
    def test_fenced_probe_request_preserves_all_coordinates(self):
        inputs=[[0.1*i, -0.2, 0.3, -0.4] for i in range(8)]
        raw='```json\n'+json.dumps({'inputs':inputs,'hypothesis':'test'})+'\n```'
        with patch.object(remote_additional,'probe_round',return_value={'pairs':[]}) as probe:
            remote_additional._next_input('.', 'test', {'condition':'adaptive_probes'}, 'prompt', [{'text':raw}])
        self.assertEqual(probe.call_args.args[3]['inputs'],inputs)
        self.assertEqual(probe.call_args.args[3]['hypothesis'],'test')

    def test_extra_prose_is_not_silently_removed(self):
        raw='Here are my probes:\n```json\n'+json.dumps({'inputs':[[0]*4]*8})+'\n```'
        with self.assertRaisesRegex(ValueError,'valid probe JSON'):
            remote_additional._next_input('.', 'test', {'condition':'adaptive_probes'}, 'prompt', [{'text':raw}])

if __name__ == '__main__': unittest.main()
