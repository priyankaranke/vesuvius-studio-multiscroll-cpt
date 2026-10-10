"""Bind region approvals to one physical surface and one output pixel grid."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
ROOT=Path(__file__).resolve().parent

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def geometry_fingerprint(case):
    values={k:case.get(k) for k in ['geometry_um','geometry_shape','matrix_to_hr','hr_um','target_um','volume_url']}
    values['uv_origin']='first mesh vertex; output centers=(index+0.5)*target_um'
    values['mesh']={n:file_sha256(ROOT/case['mesh']/n) for n in ['x.tif','y.tif','z.tif','meta.json']}
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_inference_approval(case,shape):
    required=['approval_mask_file','approval_revision','approval_geometry_fingerprint','approval_mask_sha256']
    if any(k not in case for k in required):
        raise RuntimeError('Ink inference requires a saved user approval mask for this exact surface: '+case['id'])
    if int(case['approval_revision'])<1:raise RuntimeError('Unreviewed surface cannot run ink inference')
    if geometry_fingerprint(case)!=case['approval_geometry_fingerprint']:
        raise RuntimeError('Surface geometry changed since approval; review it again')
    p=ROOT/case['approval_mask_file']
    if file_sha256(p)!=case['approval_mask_sha256']:raise RuntimeError('Approval mask differs from its saved revision')
    mask=np.asarray(Image.open(p))
    if mask.shape!=tuple(shape) or mask.ndim!=2:raise RuntimeError('Approval mask and inference UV shapes differ')
    if not np.isin(mask,[0,1,2]).all():raise RuntimeError('Invalid approval classes')
    approved=mask==1
    if not approved.any():raise RuntimeError('No approved pixels for ink inference')
    return approved

def approved_tile(approved,y,x,h,w,hr_um,target_um):
    """Exact pixel-center mapping; outside-canvas high-resolution pixels are denied."""
    yy=np.floor((np.arange(y,y+h,dtype=np.float64)+.5)*hr_um/target_um).astype(np.int64)
    xx=np.floor((np.arange(x,x+w,dtype=np.float64)+.5)*hr_um/target_um).astype(np.int64)
    good_y=(yy>=0)&(yy<approved.shape[0]);good_x=(xx>=0)&(xx<approved.shape[1])
    return approved[np.clip(yy,0,approved.shape[0]-1)[:,None],np.clip(xx,0,approved.shape[1]-1)[None,:]]&good_y[:,None]&good_x[None,:]
