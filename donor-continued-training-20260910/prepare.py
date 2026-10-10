"""Prepare native-volume donor patches on the immutable labeled UV grids."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
DONOR = ROOT.parent / 'donor-labels-20260906'
sys.path[:0] = [str(DONOR / 'expansion')]
from volume_sampling import Surface
from fast_volume_sampling import FastVolume
from expanded_inference import bounded_map
Image.MAX_IMAGE_PIXELS = None
IDENTITY = np.eye(4)
SEED = 20260910


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def affine(value):
    result = np.eye(4)
    if value is not None:
        result[:3] = value
    return result


def surface_for(row, catalog, native=True):
    m = row['source_meta']
    if not m.get('mesh'):
        return None, None
    scroll = row['scroll']
    transform = affine(m.get('matrix_to_hr'))
    source_volume = m['source_volume']
    volume_id = source_volume
    level = 2
    ct_um = m['hr_um'] * 4
    kind = 'area_mean_high_resolution'
    if native and scroll == 'PHerc0814':
        assert source_volume == '20260309142202'
        t = next(t for t in catalog[scroll][source_volume]['properties']['transforms'] if t['to_volume_id'] == '20250804134230')
        transform = affine(t['transformation_matrix']) @ transform
        volume_id, level, ct_um, kind = '20250804134230', 0, 9.362, 'native_low_resolution'
    elif native and scroll == 'PHerc0009B':
        # Both HR exports share the same acquisition, shape and voxel frame;
        # their intensity windows differ. Native->HR is published on native.
        native_id = '20250521125136'
        t = next(t for t in catalog[scroll][native_id]['properties']['transforms'] if t['to_volume_id'] == '20250820154339')
        assert source_volume in ['20250820154339', '20260319104112']
        assert catalog[scroll][source_volume]['scan_id'] == catalog[scroll]['20250820154339']['scan_id']
        transform = np.linalg.inv(affine(t['transformation_matrix'])) @ transform
        volume_id, level, ct_um, kind = native_id, 0, 8.64, 'native_low_resolution'
    elif native and scroll == 'PHerc0841' and (ROOT/'841-transform/transform.json').exists():
        recovered=json.loads((ROOT/'841-transform/transform.json').read_text())
        if not recovered['fit_success'] or recovered['heldout_error_native_voxels']['p95']>.5:
            raise RuntimeError('841 published mesh transform is not sufficiently resolved')
        transform=affine(recovered['matrix'])@transform
        volume_id,level,ct_um,kind='20250821151531',0,9.366,'native_low_resolution_mesh_derived_affine'
    if np.linalg.det(transform[:3, :3]) <= 0:
        raise ValueError('Reflected or degenerate source transform')
    if level:
        transform[:3, 3] -= (2**level - 1) / 2
        transform[:3] /= 2**level
    v = catalog[scroll][volume_id]
    url = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/' + next(d['origins'][0]['path'] for d in v['data'] if d['type'] == 'ome-zarr') + str(level) + '/'
    sign = (row.get('orientation') or {}).get('normal_sign', 1)
    s = Surface(ROOT / row['label_directory'] / 'mesh', m['geometry_um'], ct_um, transform[:3], sign, m.get('geometry_shape'))
    assert list(s.shape(m['target_um'])) == row['shape'], (row['id'], s.shape(m['target_um']), row['shape'])
    recipe = dict(kind=kind, volume_id=volume_id, source_volume=source_volume, url=url,
                  level=level, xy_pitch_um=m['target_um'], z_pitch_um=ct_um,
                  matrix=transform[:3].tolist(), normal_sign=sign,
                  offsets_um=(np.arange(-8, 8)*ct_um).tolist(),
                  pixel_centers='(index+0.5)*original_label_pitch; unchanged labeled UV',
                  limitation=('841 affine recovered from corresponding published meshes; heldout geometric p95 < 0.5 native voxel' if kind=='native_low_resolution_mesh_derived_affine' else 'Measured area-mean pyramid' if level else None))
    return s, recipe


def coordinates(ink, known, validation=False):
    """Original 256-parent/64-stride/four-128-child sparse sampler, including multiplicity."""
    h, w = ink.shape
    ph, pw = ((h+255)//256)*256, ((w+255)//256)*256
    a = np.pad(ink, ((0,ph-h),(0,pw-w)))
    b = np.pad(known, ((0,ph-h),(0,pw-w)))
    rows = []
    seen = set()
    for y in range(0, ph-255, 64):
        for x in range(0, pw-255, 64):
            if not b[y:y+256, x:x+256].any():
                continue
            if not validation and not a[y:y+256, x:x+256].any():
                continue
            for dy in [0,128]:
                for dx in [0,128]:
                    yy, xx = y+dy, x+dx
                    if not b[yy:yy+128,xx:xx+128].any():
                        continue
                    if validation and (yy,xx) in seen:
                        continue
                    seen.add((yy,xx))
                    rows.append((yy,xx,int(a[yy:yy+128,xx:xx+128].any())))
    return np.asarray(rows, np.int32).reshape(-1,3)


def select():
    rows = json.loads((ROOT / 'labels.json').read_text())
    catalog = json.loads((ROOT / 'volume-catalog.json').read_text())
    # A held-out new donor segment has no historical S1 checkpoint exposure.
    candidates = sorted(r['id'] for r in rows if r['scroll']=='PHerc0841' and r['orientation'] and r['ink_pixels']>50000 and r['source_meta'].get('mesh'))
    validation_id = str(np.random.default_rng(SEED).choice(candidates))
    holdout = next(r for r in rows if r['id']==validation_id)
    hs, _ = surface_for(holdout, catalog, native=False)
    hk = np.asarray(Image.open(ROOT/holdout['label_directory']/'supervision.png'))>0
    # Dense physical surface reference of the heldout *whole* segment. A 2mm
    # exclusion around every node covers both 128px contexts and registration.
    xyz = hs.a[hs.valid] * hs.ct_um
    tree = cKDTree(xyz)
    output=[]
    for r in rows:
        folder = ROOT/r['label_directory']
        ink=np.asarray(Image.open(folder/'inklabels.png'))>0
        known=np.asarray(Image.open(folder/'supervision.png'))>0
        coords=coordinates(ink,known,r['id']==validation_id)
        excluded=0
        if r['scroll']==holdout['scroll'] and r['id']!=validation_id:
            s,_=surface_for(r,catalog,native=False)
            if s is None:
                # Its old released-frame mapping cannot prove disjointness.
                excluded=len(coords);coords=coords[:0]
            elif len(coords):
                # points() is Cartesian; query the paired centers in bounded batches.
                centers=[]
                for y,x,_ in coords:
                    p,_,v=s.points([int(y)+64],[int(x)+64],r['source_meta']['target_um'])
                    centers.append(p[0,0]*s.ct_um if v[0,0] else np.full(3,np.inf))
                centers=np.asarray(centers)
                finite=np.isfinite(centers).all(1)
                keep=np.zeros(len(coords),bool)
                keep[finite]=tree.query(centers[finite],workers=4)[0]>2000
                excluded=int((~keep).sum());coords=coords[keep]
        out=ROOT/'prepared'/r['id'];out.mkdir(parents=True,exist_ok=True)
        np.save(out/'coordinates.npy',coords)
        info=dict(r,role='validation' if r['id']==validation_id else 'train',
                  samples=len(coords),positive_samples=int(coords[:,2].sum()),
                  negative_samples=int((coords[:,2]==0).sum()),excluded_overlap_samples=excluded)
        atomic(out/'selection.json',info)
        output.append(info)
        print('SELECT',r['id'],info['role'],len(coords),'overlap_excluded',excluded,flush=True)
    atomic(ROOT/'dataset.json',dict(seed=SEED,validation_id=validation_id,validation_overlap_exclusion_um=2000,segments=output))


def prepare_one(row,catalog,volumes):
    ident=row['id'];out=ROOT/'prepared'/ident
    if (out/'prepared.json').exists() or not row['samples']:
        return
    coords=np.load(out/'coordinates.npy')
    h,w=row['shape'];ph,pw=((h+255)//256)*256,((w+255)//256)*256
    tiles=set()
    for y,x,_ in coords:
        for ty in range(int(y)//128*128,int(y)+128,128):
            for tx in range(int(x)//128*128,int(x)+128,128):
                if ty<h and tx<w:tiles.add((ty,tx))
    tiles=sorted(tiles)
    s,recipe=surface_for(row,catalog)
    ctpath=out/'ct.npy'
    raw=np.lib.format.open_memmap(ctpath,mode='r+' if ctpath.exists() else 'w+',dtype=np.uint8,shape=(16,ph,pw))
    if s is None:
        # Retain the already measured, centered released stack on its exact XY grid.
        old=(ROOT.parent / 'donor-labels-20260906/prepared')/ident/'ct.npy'
        if not old.exists() and ident.startswith('source-'):
            old=(ROOT.parent / 'donor-labels-20260906/prepared')/ident.removeprefix('source-').split('-approved-')[0]/'ct.npy'
        if not old.exists() and (ROOT/'legacy-ct'/ident/'ct.npy').exists():old=ROOT/'legacy-ct'/ident/'ct.npy'
        if not old.exists():old=Path(row['original_ct'])
        a=np.load(old,mmap_mode='r');n=len(a);selected=a[n//2-8:n//2+8]
        if (row.get('orientation') or {}).get('normal_sign')==-1:selected=selected[::-1]
        for y,x in tiles:raw[:,y:min(y+128,h),x:min(x+128,w)]=selected[:,y:min(y+128,h),x:min(x+128,w)]
        recipe=dict(kind=row['source_meta']['kind'],source='existing measured released stack',xy_pitch_um=row['source_meta']['spacing_um'][1],z_pitch_um=row['source_meta']['spacing_um'][0],normal_sign=(row.get('orientation') or {}).get('normal_sign'))
    else:
        url=recipe['url']
        if url not in volumes:
            for old in volumes.values():old.pool.shutdown(wait=True)
            volumes.clear();volumes[url]=FastVolume(url,cache_gb=12,workers=16)
        volume=volumes[url]
        progress=out/'render-progress.json'
        done=json.loads(progress.read_text())['done'] if progress.exists() else 0
        def one(yx):
            y,x=yx;ys=np.arange(y,min(y+128,h));xs=np.arange(x,min(x+128,w))
            a,valid=s.render(volume,ys,xs,recipe['xy_pitch_um'],recipe['offsets_um'])
            a[:,~valid]=0
            return y,x,a
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            for n,(y,x,a) in enumerate(bounded_map(pool,one,tiles[done:],8),done+1):
                raw[:,y:y+a.shape[1],x:x+a.shape[2]]=a
                if n%64==0 or n==len(tiles):
                    raw.flush();atomic(progress,dict(done=n,total=len(tiles)))
                    print('RENDER',ident,n,'/',len(tiles),flush=True)
    raw.flush()
    # Exclude genuinely absent target CT from the loss, without intensity thresholding.
    labeldir=ROOT/row['label_directory']
    known=np.asarray(Image.open(labeldir/'supervision.png'))>0
    support=np.zeros((h,w),bool);covered=np.zeros((h,w),bool)
    for y,x in tiles:
        covered[y:min(y+128,h),x:min(x+128,w)]=True
        support[y:min(y+128,h),x:min(x+128,w)]=raw[:,y:min(y+128,h),x:min(x+128,w)].any(0)
    missing=int((known&covered&~support).sum())
    active=known&support
    Image.fromarray(active.astype('uint8')*255).save(out/'supervision.png')
    active_coords=np.array([c for c in coords if active[int(c[0]):int(c[0])+128,int(c[1]):int(c[1])+128].any()],dtype=np.int32).reshape(-1,3)
    ink=(np.asarray(Image.open(labeldir/'inklabels.png'))>0)&active
    for c in active_coords:c[2]=int(ink[c[0]:c[0]+128,c[1]:c[1]+128].any())
    np.save(out/'coordinates.npy',active_coords)
    atomic(out/'prepared.json',dict(id=ident,shape=[16,ph,pw],recipe=recipe,samples=len(active_coords),missing_labeled_ct_pixels=missing,rendered_tiles=len(tiles),orientation_resolved=row['orientation'] is not None))
    print('READY',ident,'samples',len(active_coords),'missing_ct',missing,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','render']);p.add_argument('--scroll');args=p.parse_args()
    if args.stage=='select':select();return
    data=json.loads((ROOT/'dataset.json').read_text());catalog=json.loads((ROOT/'volume-catalog.json').read_text());volumes={}
    for row in data['segments']:
        if not args.scroll or row['scroll']==args.scroll:prepare_one(row,catalog,volumes)
    for v in volumes.values():v.pool.shutdown(wait=True)


if __name__=='__main__':main()
