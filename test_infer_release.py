import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import infer_release


class InferenceEntrypointTests(unittest.TestCase):
    def test_loaded_weights_invoke_documented_codepath(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); weights = p / 'weights.safetensors'; weights.write_bytes(b'weights')
            source = p / 'planes16.npy'; source.touch()
            with patch.object(infer_release.subprocess, 'run') as run:
                infer_release.main(['--input', str(source), '--output', str(p / 'ink.npy'),
                                    '--checkpoint', str(weights), '--checkpoint-sha256', hashlib.sha256(b'weights').hexdigest()])
            args = run.call_args.args[0]
            self.assertTrue(args[1].endswith('studio-multiscroll-cpt-20261023/hf/inference.py'))
            self.assertEqual(args[args.index('--checkpoint') + 1], str(weights.resolve()))
            self.assertEqual(args[args.index('--stride') + 1], '64')
            self.assertEqual(args[args.index('--blend') + 1], 'hann')

    def test_bad_weights_never_run_inference(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); weights = p / 'weights.safetensors'; weights.write_bytes(b'wrong weights')
            source = p / 'planes16.npy'; source.touch()
            with patch.object(infer_release.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                    infer_release.main(['--input', str(source), '--output', str(p / 'ink.npy'), '--checkpoint', str(weights)])
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
