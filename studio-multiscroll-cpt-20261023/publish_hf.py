"""Stage and upload the model repository (private) from the training host. HF token on stdin (never written to disk).

  python publish_hf.py --model checkpoints/<selected>.safetensors [--extra NAME=checkpoints/<file> ...] --readme hf/README.md \
      --repo paranke/vesuvius-studio-multiscroll-cpt-20261023 < token
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--extra', nargs='*', default=[])
    p.add_argument('--readme', required=True); p.add_argument('--repo', required=True); p.add_argument('--stage', default=str(HERE / 'hf-stage'))
    a = p.parse_args(); token = sys.stdin.readline().strip()
    stage = Path(a.stage); shutil.rmtree(stage, ignore_errors=True); (stage / 'records').mkdir(parents=True)
    files = {'model.safetensors': Path(a.model)}
    for e in a.extra:
        name, path = e.split('=', 1); files[name] = Path(path)
    hashes = {}
    for name, src in files.items():
        dst = stage / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        hashes[name] = dict(source=src.name, sha256=hashlib.sha256(dst.read_bytes()).hexdigest(), bytes=dst.stat().st_size)
    for name in ('inference.py', 'model.py', 'requirements.txt'):
        shutil.copy2(HERE / 'hf' / name, stage / name)
    shutil.copy2(a.readme, stage / 'README.md')
    for name in ('config.json', 'mixture.json', 'training-result.json', 'training.jsonl'):
        if (HERE / name).exists(): shutil.copy2(HERE / name, stage / 'records' / name)
    evaluations = [json.loads(l) for l in (HERE / 'training.jsonl').read_text().splitlines() if '"evaluation' in l]
    (stage / 'records' / 'evaluations.json').write_text(json.dumps(evaluations, indent=1) + '\n')
    for f in (HERE.parent.parent / 'evals').glob('*.json'):
        shutil.copy2(f, stage / 'records' / f.name)
    (stage / 'records' / 'files.json').write_text(json.dumps(hashes, indent=1) + '\n')
    from huggingface_hub import HfApi
    api = HfApi(token=token); api.create_repo(a.repo, private=True, exist_ok=True, repo_type='model')
    info = api.upload_folder(folder_path=str(stage), repo_id=a.repo, repo_type='model', commit_message='Studio multi-scroll continued pretraining (2026-10-23): weights, inference package, run records')
    print('UPLOADED', a.repo, info.oid if hasattr(info, 'oid') else info, json.dumps(hashes), flush=True)


if __name__ == '__main__':
    main()
