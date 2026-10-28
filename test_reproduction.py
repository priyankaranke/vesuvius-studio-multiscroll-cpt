"""CPU-only integration tests for dataset validation and training orchestration."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import reproduction


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'code'
        self.root.mkdir()
        self.cpt = self.root / 'studio-multiscroll-cpt-20261023'
        self.cpt.mkdir()
        for name in ('donor-studio-resunet-scratch-20260926', 'studio-softloss-20261023'):
            (self.root / name).mkdir()
        rows = []
        for family, count in [('expanded9/prepared', 69), ('reader-rows', 47)]:
            for i in range(count):
                folder = self.root.parent / family / str(i)
                folder.mkdir(parents=True)
                (folder / 'ready.json').write_text(json.dumps({'status': 'complete', 'missing_chunks': 0}))
                for name in ('tiles.npy', 'coordinates.npy', 'label.png', 'mask.png', 'ct-tiles.npy'):
                    (folder / name).touch()
                rows.append('../../' + family + '/' + str(i))
        (self.cpt / 'mixture.json').write_text(json.dumps({'rows': rows, 'groups': {'PHerc0500P2': {'weight': .1}}}))
        self.calls = []
        self.fail_at = None

    def command(self, args, cwd=None, **env):
        args = list(map(str, args))
        self.calls.append((args, cwd, env))
        program = Path(args[1]).name if len(args) > 1 else args[0]
        if program == self.fail_at:
            raise subprocess.CalledProcessError(75, args)
        if program in ('train_cpt.py', 'train_softloss.py'):
            if program == 'train_softloss.py' or env['CPT_STOP_AFTER_UPDATE'] == 30000:
                folder = Path(cwd) / 'checkpoints'
                folder.mkdir(exist_ok=True)
                (folder / 'model-ema.safetensors').write_bytes(program.encode())
        if args[0] == 'gh':
            Path(args[args.index('-O') + 1]).write_bytes(b'released intermediate')

    def run_pipeline(self, mode='all'):
        runner = reproduction.Reproduction(self.root, mode)
        runner.command = self.command
        with contextlib.redirect_stdout(io.StringIO()), patch.object(reproduction, 'export_scratch',
                side_effect=lambda checkpoint, output: output.write_bytes(b'new scratch weights')):
            runner.run()

    def test_full_chain_uses_new_scratch_weights_and_all_datasets(self):
        self.run_pipeline()
        training = [(a, cwd, e) for a, cwd, e in self.calls if a[1].endswith(('train.py', 'train_cpt.py', 'train_softloss.py'))]
        self.assertEqual([a[1] for a, _, _ in training], ['train.py', 'train_cpt.py', 'train_cpt.py', 'train_cpt.py', 'train_softloss.py'])
        self.assertEqual(training[0][2]['STUDIO_MAX_UPDATES'], 74250)
        phases = [e for _, _, e in training[1:4]]
        self.assertEqual([e['CPT_STOP_AFTER_UPDATE'] for e in phases], [2024, 2167, 30000])
        self.assertEqual([e['CPT_ENABLE_EMA'] for e in phases], ['0', '0', '1'])
        self.assertEqual([e['CPT_PEAK_LR'] for e in phases], ['1e-4', '1e-4', '6e-5'])
        self.assertTrue(all(e['CPT_BASE'] == self.root / '.reproduction/studio-epoch9.safetensors' for e in phases))
        self.assertEqual(training[-1][2]['CPT_BASE'], self.cpt / 'checkpoints/model-ema.safetensors')
        early = json.loads((self.cpt / 'mixture-stage1.json').read_text())
        self.assertEqual(len(early['rows']), 69)
        self.assertEqual(early['groups']['PHerc0500P2']['weight'], .04)
        # Full mode must never fetch released weights or the original pretrained model.
        self.assertFalse(any(a[0] == 'gh' or any('resunet-studio-20260926' in v for v in a) for a, _, _ in self.calls))
        downloads = [a for a, _, _ in self.calls if 'download' in a]
        self.assertEqual(len(downloads), 2)
        self.assertTrue(all('--revision' in a for a in downloads))

    def test_resume_does_not_repeat_completed_training(self):
        self.run_pipeline()
        self.calls.clear()
        self.run_pipeline()
        self.assertFalse(any(a[1].startswith('train') for a, _, _ in self.calls))

    def test_interrupted_stage_never_advances_to_softloss(self):
        self.fail_at = 'train_cpt.py'
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline()
        self.assertFalse(any(a[1] == 'train_softloss.py' for a, _, _ in self.calls))
        self.fail_at = None
        self.calls.clear()
        self.run_pipeline()
        self.assertFalse(any(a[1] == 'train.py' for a, _, _ in self.calls))
        self.assertTrue(any(a[1] == 'train_softloss.py' for a, _, _ in self.calls))

    def test_missing_reader_data_stops_before_any_training(self):
        (self.root.parent / 'reader-rows/46/ct-tiles.npy').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete training row'):
            self.run_pipeline()
        self.assertFalse(any(a[1].startswith('train') for a, _, _ in self.calls))

    def test_missing_ct_chunks_are_rejected(self):
        folder = self.root.parent / 'expanded9/prepared/0'
        (folder / 'ready.json').write_text(json.dumps({'status': 'complete', 'missing_chunks': 1}))
        with self.assertRaisesRegex(RuntimeError, 'Incomplete CT data'):
            reproduction.validate_rows([folder])

    def test_unrelated_checkpoints_are_not_adopted(self):
        (self.cpt / 'checkpoints').mkdir()
        with self.assertRaisesRegex(RuntimeError, 'fresh clone'):
            self.run_pipeline()
        self.assertEqual(self.calls, [])

    def test_modified_completed_weights_are_rejected(self):
        self.run_pipeline()
        (self.root / '.reproduction/studio-epoch9.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'missing or changed'):
            self.run_pipeline()

    def test_softloss_mode_is_explicitly_separate(self):
        self.run_pipeline('softloss')
        self.assertTrue(any(a[0] == 'gh' for a, _, _ in self.calls))
        self.assertFalse(any(a[1] == 'train.py' for a, _, _ in self.calls))
        with self.assertRaisesRegex(RuntimeError, 'different reproduction mode'):
            self.run_pipeline('all')


if __name__ == '__main__':
    unittest.main()
