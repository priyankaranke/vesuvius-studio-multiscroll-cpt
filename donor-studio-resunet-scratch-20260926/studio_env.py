"""Portability helpers shared by the training, data and rendering scripts.

load_env()  reads KEY=VALUE lines from a .env file (repository root or this folder) into os.environ without overriding
            variables already set. Every key is optional; see .env.example.
localize(p) the frozen records (data.json, label-snapshot.json, scan configs) keep the original training host's absolute
            paths (/home/ubuntu/<folder>/...). They are provenance and part of the run's hashed configuration, so they are
            not rewritten; localize() maps such a path onto this checkout: the first component that names a top-level
            folder of the repository layout (or its HF data layout) decides where it lands.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FOLDERS = ('donor-studio-resunet-scratch-20260926', 'donor-spirals-20260921', 'donor-continued-training-20260910',
           'donor-original-recipe-20260910', 'donor-resunet-scratch-20260912', 'donor-labels-20260906',
           'pherc1447-architecture-study-20260906')


def load_env():
    for f in (WORK / '.env', ROOT / '.env'):
        if not f.exists(): continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line: continue
            k, v = line.split('=', 1); k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ: os.environ[k] = v


def localize(path):
    p = Path(path); parts = p.parts
    for i, part in enumerate(parts):
        if part in FOLDERS: return WORK.joinpath(*parts[i:])
    return p
