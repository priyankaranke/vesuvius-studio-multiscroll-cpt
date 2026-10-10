"""Download the Studio training data from Hugging Face into this checkout and verify every file.

The dataset's paths mirror this repository's layout, so files land next to the code. Each file is checked against the
Hub's record of that revision (SHA-256 for large files, git blob id for small ones); files already present and correct
are skipped, so an interrupted download resumes. A Hugging Face token is read from HF_TOKEN (.env or environment) or
from `hf auth login`, and is needed only while the dataset is private.

  python download_data.py                      # training data: plans, labels, CT tiles, run records (~62 GB)
  python download_data.py --resume-checkpoint  # + the resumable epoch-9 training state (2.7 GB)
  python download_data.py --all                # everything, incl. the geometry backup used to re-render the CT
  python download_data.py --max-mib-per-s 35   # optional download-rate cap
"""
import argparse, hashlib, os, sys, time
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK / 'donor-studio-resunet-scratch-20260926'))
from studio_env import load_env  # noqa: E402

REPO = 'paranke/vesuvius-814-841-9b-studio-training-data'
REVISION = 'ad919d288ede1c4daf92725288bb6fe29c558591'   # dataset revision verified by the 2026-10-10 fresh-machine reproduction
RUN = 'donor-studio-resunet-scratch-20260926/'
TRAINING = (RUN + 'prepared/', RUN + 'data.json', RUN + 'config.json', RUN + 'label-snapshot.json', RUN + 'qualification.json')


def git_blob_sha1(path):
    h = hashlib.sha1(); h.update(b'blob %d\0' % path.stat().st_size)
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def ok(path, s):
    if not path.exists() or path.stat().st_size != s.size: return False
    return sha256(path) == s.lfs.sha256 if s.lfs else git_blob_sha1(path) == s.blob_id


def fetch_throttled(url, dest, token, rate):
    import requests
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    tmp = dest.with_name(dest.name + '.part'); dest.parent.mkdir(parents=True, exist_ok=True); t0 = time.monotonic(); n = 0
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r, tmp.open('wb') as f:
        r.raise_for_status()
        for chunk in r.iter_content(4 << 20):
            f.write(chunk); n += len(chunk); wait = n / rate - (time.monotonic() - t0)
            if wait > 0: time.sleep(wait)
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--revision', default=REVISION); ap.add_argument('--all', action='store_true')
    ap.add_argument('--resume-checkpoint', action='store_true'); ap.add_argument('--max-mib-per-s', type=float, default=0)
    a = ap.parse_args(); load_env()
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get('HF_TOKEN') or None
    info = HfApi(token=token).dataset_info(REPO, revision=a.revision, files_metadata=True); rev = info.sha
    want = [s for s in info.siblings if s.rfilename != '.gitattributes' and (a.all or s.rfilename.startswith(TRAINING)
            or (a.resume_checkpoint and s.rfilename == RUN + 'checkpoints/resume.pt'))]
    total = sum(s.size for s in want); print(f'{REPO}@{rev[:12]}: {len(want)} files, {total / 1e9:.1f} GB', flush=True)
    done = 0
    for i, s in enumerate(sorted(want, key=lambda s: s.rfilename)):
        dest = WORK / s.rfilename
        if not ok(dest, s):
            if a.max_mib_per_s:
                fetch_throttled(f'https://huggingface.co/datasets/{REPO}/resolve/{rev}/{s.rfilename}', dest, token, a.max_mib_per_s * 2 ** 20)
            else:
                hf_hub_download(REPO, s.rfilename, repo_type='dataset', revision=rev, local_dir=WORK, token=token)
            if not ok(dest, s): sys.exit(f'CHECKSUM MISMATCH {s.rfilename}')
        done += s.size
        if i % 50 == 0 or done == total: print(f'  {i + 1}/{len(want)} files, {done / 1e9:.1f}/{total / 1e9:.1f} GB verified', flush=True)
    print(f'DONE: {len(want)} files verified against {REPO}@{rev}', flush=True)


if __name__ == '__main__':
    main()
