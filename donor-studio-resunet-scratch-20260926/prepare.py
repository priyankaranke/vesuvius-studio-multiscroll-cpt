"""Frozen sparse-label eligibility, geometry exclusion, and native tile plan."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys
import zlib
import numpy as np
from PIL import Image
import tifffile
from scipy.spatial import cKDTree
from studio_env import localize

ROOT = Path(__file__).resolve().parent
HOST = ROOT.parent
STUDIO = HOST / 'donor-spirals-20260921'
BASE = HOST / 'donor-resunet-scratch-20260912'
DONOR = HOST / 'donor-continued-training-20260910'
HOLDOUT = '841-expanded-layer-03-approved-r000001'
SCROLLS = {'814': 'PHerc0814', '841': 'PHerc0841', '9B': 'PHerc0009B'}
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, str(DONOR))
spec = importlib.util.spec_from_file_location('accepted_native', DONOR / 'prepare.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
spec = importlib.util.spec_from_file_location('original_selection', HOST / 'donor-original-recipe-20260910/prepare.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def decode(row):
    folder = localize(row['snapshot_directory'])
    path = folder / row['labels_file']
    if row['category'] == 'full-canvases':
        h, w = row['metadata']['shape'][-2:]
        state = np.zeros((h, w), np.uint8)
        # Same data_parent ancestry and ordering as authoritative label_tile.
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            records = db.execute('WITH RECURSIVE ancestors(id) AS (SELECT ? UNION ALL SELECT r.data_parent FROM revisions r JOIN ancestors a ON r.id=a.id WHERE r.data_parent>0) SELECT t.x,t.y,t.data FROM tiles t JOIN ancestors a ON t.rev=a.id WHERE t.level=0 ORDER BY t.rev DESC', (row['revision'],))
            seen = set()
            for x, y, payload in records:
                if (y, x) in seen:
                    continue
                seen.add((y, x))
                tile = np.frombuffer(zlib.decompress(payload), np.uint8).reshape(256, 256)
                state[y*256:min(h,(y+1)*256), x*256:min(w,(x+1)*256)] = tile[:min(256,h-y*256), :min(256,w-x*256)]
    elif path.suffix == '.tif':
        state = tifffile.imread(path, key=0, maxworkers=1)
    else:
        state = np.asarray(Image.open(path)).copy()
    assert set(np.unique(state).tolist()) <= {0, 1, 2}, row['id']
    assert int((state == 2).sum()) == row['ink'], row['id']
    assert int((state == 1).sum()) == row['background'], row['id']
    return state


def geometry(row, baseline):
    if row['category'] == 'original-labels':
        if row['id'] not in baseline:
            return None
        source = baseline[row['id']]
        surface, recipe = native.surface_for(source, read(DONOR / 'volume-catalog.json'))
        return dict(kind='original', baseline=source, recipe=source['recipe'],
                    runtime_reverse=source['runtime_reverse'], surface=surface,
                    mesh=str(DONOR / source['label_directory'] / 'mesh') if surface is not None else None,
                    offset_yx=[0, 0], pitch=source['recipe']['xy_pitch_um'])
    m = row['metadata']
    sid = m['parent_segment_id']
    folder = STUDIO / row['scroll']
    surface = read(folder / 'results' / sid / 'surface.json')
    prepared = read(folder / 'prepared' / sid / 'prepared.json')
    ori = row['orientation']
    if not ori or ori.get('normal_basis') != 'native_renderer_column_cross_row':
        raise ValueError('Missing native label direction binding')
    expected = m['geometry_fingerprint']
    assert ori['geometry_fingerprint'] == expected, 'Saved label geometry binding changed'
    assert (m.get('parent_geometry_fingerprint') or expected) == surface['geometry']
    assert prepared['geometry_fingerprint'] == surface['geometry']
    teachers = {t['id']: t for t in m['teachers']}
    teacher = teachers[ori['teacher_id']]
    assert teacher['checkpoint_sha256'] == ori['checkpoint_sha256']
    assert teacher['normal_sign'] == ori['normal_sign']
    direction = 'forward' if ori['normal_sign'] == 1 else 'reverse'
    if row['scroll'] in ('814', '841'):
        production = surface['productionDepthOrder']
        assert production['training_contract'] == 'native_donor_inward_v1'
        assert production['geometry_fingerprint'] == surface['geometry']
        assert production['direction'] == direction, 'Label direction conflicts with accepted physical training convention'
    mesh = Path(prepared.get('source') if prepared.get('source_kind') == 'native_spiral_fit' or prepared.get('winding') else prepared.get('mesh', prepared.get('source', '')))
    mesh = localize(mesh) if mesh.is_absolute() else STUDIO / mesh
    assert (mesh / 'meta.json').exists(), str(mesh)
    scan = read(folder / 'config/scan.json')
    offset = [0, 0]
    if row['category'] == 'hr-rois':
        box = m['parent_crop_xyxy']
        assert [box[3]-box[1], box[2]-box[0]] == m['shape'][-2:]
        offset = [box[1], box[0]]
    return dict(kind='campaign', mesh=str(mesh), parent=sid, direction=direction,
                geometry_fingerprint=surface['geometry'], scan=scan, offset_yx=offset,
                pitch=scan['voxel_size_um'], parent_shape=[surface['height'], surface['width']],
                recipe=dict(renderer='accepted vc_render_tifxyz', native17_offsets=list(range(-8,9)),
                            model_offsets=list(range(-8,8)) if direction=='forward' else list(range(8,-8,-1)),
                            normalization='clip200/div255', shape=[surface['height'],surface['width']],
                            geometry_fingerprint=surface['geometry']))


def exclusion(coords, geom, holdout):
    if not len(coords):
        return coords, {}
    hs = holdout
    hxyz = hs.a[hs.valid] * hs.ct_um
    tree = cKDTree(hxyz)
    # Cover the entire holdout surface between its nodes, including the CT depth
    # interval, rather than only measuring center-to-center distances.
    def diagonal(a, valid):
        good = valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
        d = np.maximum(np.linalg.norm(a[1:,1:]-a[:-1,:-1],axis=-1),
                       np.linalg.norm(a[1:,:-1]-a[:-1,1:],axis=-1))
        return d, good
    hd,hv = diagonal(hs.a*hs.ct_um,hs.valid)
    holdout_cover = float(hd[hv].max())
    if geom['kind'] == 'original':
        s = geom['surface']
        if s is None:
            return coords[:0], dict(reason='Released 841 source lacks proven registration to holdout')
        a, valid, scale = s.a*s.ct_um, s.valid, np.array([geom['pitch']/s.pitch[1], geom['pitch']/s.pitch[0]])
    else:
        mesh = localize(geom['mesh'])
        a = np.stack([np.asarray(Image.open(mesh/(k+'.tif')),np.float32) for k in 'xyz'],-1)*geom['pitch']
        valid = np.isfinite(a).all(-1)&(a>=0).all(-1)
        scale = np.asarray(read(mesh/'meta.json')['scale'],np.float32)[::-1]
    distance = np.full(valid.shape, np.inf)
    distance[valid] = tree.query(a[valid],workers=1)[0]
    d, good = diagonal(a,valid)
    radius = np.zeros(valid.shape)
    for oy,ox in ((0,0),(0,1),(1,0),(1,1)):
        np.maximum(radius[oy:oy+d.shape[0],ox:ox+d.shape[1]],np.where(good,d,0),out=radius[oy:oy+d.shape[0],ox:ox+d.shape[1]])
    # Triangle inequality: if any CT sample could be <2mm from any point in
    # heldout geometry, its neighboring vertex enters this conservative mask.
    unsafe = valid & (distance <= 2000 + holdout_cover + radius + 8*(hs.ct_um+geom['pitch']))
    integral = np.pad(unsafe.astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0)))
    unique,inverse = np.unique(coords[:,:2],axis=0,return_inverse=True)
    lower = np.floor((unique + np.asarray(geom['offset_yx']) -112-.5)*scale).astype(int)-2
    upper = np.ceil((unique + np.asarray(geom['offset_yx']) +240+.5)*scale).astype(int)+3
    lower=np.maximum(lower,0); upper=np.minimum(upper,unsafe.shape)
    lower=np.minimum(lower,unsafe.shape); upper=np.maximum(upper,0)
    y0,x0=lower.T;y1,x1=upper.T
    keep=(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])==0
    return coords[keep[inverse]],dict(method='All mesh cells intersecting full352 measured context plus normal halo; conservative triangle-inequality distance bound',
                                    holdout_node_cover_um=holdout_cover, exclusion_um=2000,
                                    excluded_entries=int((~keep[inverse]).sum()))


def select_one(row):
    baseline={r['id']:r for r in read(BASE/'data.json')['segments']}
    out = ROOT/'prepared'/row['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():
        return read(out/'selection.json')
    result={k:v for k,v in row.items() if k!='metadata'}
    result.update(scroll=SCROLLS[row['scroll']], role='validation' if row['id']==HOLDOUT else 'train',samples=0)
    try:
        geom=geometry(row,baseline)
    except (AssertionError,ValueError,KeyError) as exc:
        result.update(excluded_reason='Unresolved geometry/orientation: '+str(exc));atomic(out/'selection.json',result);return result
    if geom is None:
        result.update(excluded_reason='No accepted source-specific input/orientation recipe; checking source alias separately')
        atomic(out/'selection.json',result);return result
    state=decode(row)
    folder=localize(row['snapshot_directory']);m=row['metadata']
    maskfile=folder/(m['validMaskFile'] if row['category']=='original-labels' else m.get('geometry_file','geometry-valid.png'))
    physical=tifffile.imread(maskfile,key=0,maxworkers=1)>0 if maskfile.suffix=='.tif' else np.asarray(Image.open(maskfile))>0
    assert physical.shape==state.shape
    state[~physical]=0
    # Exact raster aliases only. Independently painted/reflattened overlaps stay.
    alias_removed=0
    if row['id']=='9b':
        alias=next(r for r in read(ROOT/'label-snapshot.json')['datasets'] if r['id']=='source-9b-approved-r000001')
        other=decode(alias);same=(state>0)&(state==other);alias_removed=int(same.sum());state[same]=0
    coords=original.coordinates(state==2,physical,result['role']=='validation')
    result['eligible_before_holdout']=len(coords)
    if row['scroll']=='841' and result['role']=='train':
        hs,_=native.surface_for(baseline[HOLDOUT],read(DONOR/'volume-catalog.json'))
        coords,evidence=exclusion(coords,geom,hs)
        result['holdout_exclusion']=evidence
    unique,inverse=np.unique(coords[:,:2],axis=0,return_inverse=True)
    np.save(out/'labels.npy',state)
    np.save(out/'coordinates.npy',unique)
    np.save(out/'entry-to-patch.npy',inverse)
    # 512 tiles share native pixels between overlapping352 parents. Only tiles
    # actually reachable by the unchanged random224 context fetch are prepared.
    h,w=state.shape;tiles=set()
    for y,x in unique:
        for yy in range(max(0,int(y)-112)//512*512,min(h,int(y)+240),512):
            for xx in range(max(0,int(x)-112)//512*512,min(w,int(x)+240),512):
                tiles.add((yy,xx))
    if result['role']=='validation':
        tiles={(yy,xx) for y,x in unique for yy in range(int(y)//512*512,min(h,int(y)+128),512) for xx in range(int(x)//512*512,min(w,int(x)+128),512)}
    np.save(out/'tiles.npy',np.array(sorted(tiles),np.int32).reshape(-1,2))
    geom.pop('surface',None)
    result.update(shape=list(state.shape),samples=len(coords),distinct_patches=len(unique),tiles=len(tiles),
                  folder=str(out),geometry=geom,supervised_ink=int((state==2).sum()),supervised_background=int((state==1).sum()),
                  exact_alias_pixels_removed=alias_removed,
                  empty_parent_policy='Any ink in physically valid256 parent; all four128 children, original multiplicity')
    if not len(coords):
        result['excluded_reason']='No eligible any-ink parent after validity and physical holdout exclusion'
    atomic(out/'selection.json',result)
    print('SELECTED',row['id'],len(coords),len(unique),len(tiles),flush=True)
    return result


def select(workers=12):
    snapshot=read(ROOT/'label-snapshot.json')
    with ProcessPoolExecutor(workers) as pool:
        rows=list(pool.map(select_one,snapshot['datasets']))
    entries=patches=0
    for row in rows:
        if row['role']=='train' and row['samples']:
            row.update(entry_offset=entries,patch_offset=patches)
            entries+=row['samples'];patches+=row['distinct_patches']
    counts={s:sum(r['samples'] for r in rows if r['scroll']==s and r['role']=='train') for s in SCROLLS.values()}
    assert min(counts.values())>0
    draws=math.ceil(max(counts.values())/64)*64
    doc=dict(status='selected',initialization='scratch',label_snapshot=str(ROOT/'label-snapshot.json'),cutoff_utc=snapshot['cutoff_utc'],
             validation_id=HOLDOUT,validation_overlap_exclusion_um=2000,entry_counts=counts,
             distinct_counts={s:sum(r.get('distinct_patches',0) for r in rows if r['scroll']==s and r['role']=='train') for s in SCROLLS.values()},
             training_entries=entries,distinct_training_patches=patches,segments=rows,epochs=10,
             epoch_draws=3*draws,updates_per_epoch=3*draws//64,total_updates=30*draws//64,
             supervision_mask=True,s1_samples=0,
             storage=dict(labels_bytes=sum(math.prod(r.get('shape',[0])) for r in rows),
                          ct_tiles_bytes=sum(r.get('tiles',0)*512*512*16 for r in rows),
                          renderer_scratch_bound_bytes=12*32*512*512,checkpoint_reserve_bytes=12*2**30,
                          ct_download_bytes=0,policy='Reuse complete native acquisition caches; no HR canvases;512 shared native tiles'))
    atomic(ROOT/'data.json',doc)
    print('SELECT_COMPLETE',json.dumps({k:v for k,v in doc.items() if k!='segments'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select']);p.add_argument('--workers',type=int,default=12)
    args=p.parse_args();select(args.workers)
