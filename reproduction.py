"""Run the complete training chain. Invoked by reproduce.sh; no GPU work at import time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXPANDED_REVISION = '6049be1c45ddb7f659db8cb4daad023c75e1f98f'
READER_REVISION = '112fe71eb71bcd382a793f61bb53ab4b18313b47'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(rows):
    """Reject incomplete downloads instead of silently changing the training mixture."""
    for folder in map(Path, rows):
        required = ['ready.json', 'tiles.npy', 'coordinates.npy', 'label.png', 'mask.png']
        if not (folder / 'ct-tiles.npy').is_file():
            required += ['ct-tiles.blosc', 'ct-tile-offsets.npy']
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            raise RuntimeError(f'Incomplete training row {folder}: missing {missing}')
        meta = json.loads((folder / 'ready.json').read_text())
        if meta.get('status') != 'complete' or meta.get('missing_chunks', 0):
            raise RuntimeError(f'Incomplete CT data in {folder}: {meta}')


def export_scratch(checkpoint, output):
    import torch
    from safetensors.torch import save_file
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if saved.get('initialization') != 'scratch' or saved['step'] != 74250 or saved.get('next_epoch') != 9:
        raise RuntimeError('Scratch training did not reach the required nine-epoch checkpoint (74,250 updates)')
    save_file({k: v.contiguous() for k, v in saved['state_dict'].items()}, str(output))


class Reproduction:
    def __init__(self, root, mode):
        self.root = Path(root).resolve()
        self.mode = mode
        self.work = self.root.parent
        self.python = str(Path(sys.executable).absolute())
        self.hf = str(Path(self.python).with_name('hf'))
        self.cpt = self.root / 'studio-multiscroll-cpt-20261023'
        self.soft = self.root / 'studio-softloss-20261023'
        self.scratch = self.root / 'donor-studio-resunet-scratch-20260926'
        self.state_path = self.root / '.reproduction' / 'state.json'
        self.base = self.state_path.parent / 'studio-epoch9.safetensors'
        self.env = dict(os.environ, STUDIO_WANDB=os.environ.get('STUDIO_WANDB', '0'))
        self.state = dict(recipe=1, mode=mode, stages={})

    def command(self, args, cwd=None, **env):
        print('+', ' '.join(map(str, args)), flush=True)
        subprocess.run(list(map(str, args)), cwd=cwd or self.root, env=self.env | {k: str(v) for k, v in env.items()}, check=True)

    def save(self):
        self.state_path.parent.mkdir(exist_ok=True)
        tmp = self.state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=2) + '\n')
        tmp.replace(self.state_path)

    def stage(self, name, action, output=None):
        receipt = self.state['stages'].get(name)
        if receipt is not None:
            if output is not None and (not output.is_file() or sha256(output) != receipt['sha256']):
                raise RuntimeError(f'{name}: completed stage output is missing or changed: {output}')
            print('Already complete:', name, flush=True)
            return
        action()
        self.state['stages'][name] = ({'sha256': sha256(output)} if output else {})
        self.save()

    def run(self):
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get('recipe') != 1 or self.state['mode'] != self.mode:
                raise RuntimeError('Use a separate clean checkout for a different reproduction mode')
        else:
            for folder in (self.scratch, self.cpt, self.soft):
                if (folder / 'checkpoints').exists() or (folder / 'training.jsonl').exists() or (folder / 'training-result.json').exists():
                    raise RuntimeError(f'Existing training output in {folder}; use a fresh clone to preserve it')
            self.save()

        self.command([self.python, 'download_data.py'])
        self.command([self.hf, 'download', 'paranke/vesuvius-expanded9-training-data', '--repo-type', 'dataset',
                      '--revision', EXPANDED_REVISION, '--include', 'prepared/*', '--local-dir', self.work / 'expanded9'])
        self.command([self.hf, 'download', 'domenicor046/reader-v2-labels', '--repo-type', 'dataset',
                      '--revision', READER_REVISION, '--local-dir', self.work / 'reader-v2'])
        self.command(['tar', '-xf', self.work / 'reader-v2/reader-v2-dense-native-labels.tar', '-C', self.work / 'reader-v2'])
        # Always run the preparer: an existing output directory can still be partial.
        self.command([self.python, self.cpt / 'prepare_reader.py', '--labels', self.work / 'reader-v2/reader-v2-labels',
                      '--manifest', self.work / 'reader-v2/manifest.csv', '--out', self.work / 'reader-rows',
                      '--objects', '0139', '0500P2', '0343P', '0172', '--max-patches', '0172=700'])
        mixture = json.loads((self.cpt / 'mixture.json').read_text())
        rows = [(self.cpt / p).resolve() for p in mixture['rows']]
        validate_rows(rows)
        if (len(set(rows)) != 116 or sum('expanded9/prepared/' in p for p in mixture['rows']) != 69
                or sum('reader-rows/' in p for p in mixture['rows']) != 47):
            raise RuntimeError('Expected all 69 Expanded9 and 47 Reader training rows')

        if self.mode == 'all':
            def scratch():
                self.command([self.python, 'train.py'], cwd=self.scratch, STUDIO_MAX_UPDATES=74250)
                export_scratch(self.scratch / 'checkpoints/resume.pt', self.base)
            self.stage('scratch-nine-epochs', scratch, self.base)
            early = dict(mixture, rows=[p for p in mixture['rows'] if 'reader-rows/' not in p],
                         groups=json.loads(json.dumps(mixture['groups'])))
            early['groups']['PHerc0500P2']['weight'] = 0.04
            stage1 = self.cpt / 'mixture-stage1.json'
            stage1.write_text(json.dumps(early, indent=2) + '\n')
            for name, manifest, stop, peak, ema in [
                ('cpt-expanded9', stage1, 2024, '1e-4', '0'),
                ('cpt-all-data', self.cpt / 'mixture.json', 2167, '1e-4', '0'),
                ('cpt-final', self.cpt / 'mixture.json', 30000, '6e-5', '1'),
            ]:
                def train(manifest=manifest, stop=stop, peak=peak, ema=ema):
                    self.command([self.python, 'train_cpt.py'], cwd=self.cpt, CPT_BASE=self.base,
                                 CPT_MIXTURE_PATH=manifest, CPT_STOP_AFTER_UPDATE=stop, CPT_PEAK_LR=peak,
                                 CPT_ENABLE_EMA=ema, CPT_TOTAL_UPDATES=30000, CPT_WARMUP=600,
                                 CPT_REPLAY=20, CPT_EVAL_EVERY=2000, CPT_EMA_DECAY='0.9995')
                output = self.cpt / 'checkpoints/model-ema.safetensors' if name == 'cpt-final' else None
                self.stage(name, train, output)
        else:
            def download_intermediate():
                (self.cpt / 'checkpoints').mkdir(exist_ok=True)
                self.command(['gh', 'release', 'download', 'studio-multiscroll-cpt-20261023', '-R',
                              'priyankaranke/vesuvius-studio-multiscroll-cpt', '-p',
                              'studio-multiscroll-cpt-20261023-ema.safetensors', '-O',
                              self.cpt / 'checkpoints/model-ema.safetensors', '--clobber'])
            self.stage('released-intermediate', download_intermediate, self.cpt / 'checkpoints/model-ema.safetensors')

        def softloss():
            self.command([self.python, 'train_softloss.py'], cwd=self.soft,
                         CPT_BASE=self.cpt / 'checkpoints/model-ema.safetensors', CPT_TOTAL_UPDATES=3000,
                         CPT_PEAK_LR='6e-5', CPT_WARMUP=200, CPT_REPLAY=20, CPT_EVAL_EVERY=1000,
                         CPT_EMA_DECAY='0.998', SOFT_SMOOTH='0.25')
        final = self.soft / 'checkpoints/model-ema.safetensors'
        self.stage('softloss', softloss, final)
        print('DONE:', final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='all', choices=['all', 'softloss'])
    args = parser.parse_args()
    Reproduction(Path(__file__).resolve().parent, args.mode).run()
