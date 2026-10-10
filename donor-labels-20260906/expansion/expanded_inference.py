"""Explicit surface rendering and canonical R152 inference with a shared GPU lock.

The CPU stage writes whole 65-plane 9.36 um CT. The GPU stage evaluates the
measured 2.4 um source, then maps scores onto the identical physical UV canvas.
Original labels and user revisions in the starter dashboard are never modified.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse,concurrent.futures,json,time,sys,fcntl,contextlib
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,gaussian_filter
from volume_sampling import Volume,Surface
from approval_geometry import load_inference_approval,approved_tile
from review_inference_gate import require_review_targets
Image.MAX_IMAGE_PIXELS=None
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'prepared-expanded'
def atomic(p,data):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def grid(n,tile=256,stride=128):
    a=list(range(0,max(1,n-tile+1),stride));last=max(0,n-tile)
    if a[-1]!=last:a.append(last)
    return a
def bounded_map(pool,fn,items,limit):
    items=iter(items);pending=deque()
    for _ in range(limit):
        try:pending.append(pool.submit(fn,next(items)))
        except StopIteration:break
    while pending:
        yield pending.popleft().result()
        try:pending.append(pool.submit(fn,next(items)))
        except StopIteration:pass
def make_surface(case,level=0,sign=1):
    m=np.asarray(case.get('matrix_to_hr') or [[1,0,0,0],[0,1,0,0],[0,0,1,0]],np.float64)
    if level:
        # The raw pyramid is an area mean. Its voxel k represents L0 samples
        # [factor*k, factor*k+factor-1], whose physical center is offset below.
        factor=2**level;m=m.copy();m[:,3]-=(factor-1)/2;m/=factor
    return Surface(ROOT/case['mesh'],case['geometry_um'],case['hr_um']*2**level,m,sign,case.get('geometry_shape'))
def cpu_case(case):
    folder=OUT/case['id'];folder.mkdir(parents=True,exist_ok=True)
    complete=folder/'ct-ready.json'
    if complete.exists():return
    s=make_surface(case,2);v=Volume(case['volume_url']+'2/',cache_gb=5,workers=6);pitch=case['target_um'];h,w=s.shape(pitch)
    p=folder/'ct.npy';mode='r+' if p.exists() else 'w+';a=np.lib.format.open_memmap(p,mode=mode,dtype=np.uint8,shape=(65,h,w))
    gp=folder/'geometry-valid.npy';valid=np.lib.format.open_memmap(gp,mode='r+' if gp.exists() else 'w+',dtype=np.uint8,shape=(h,w))
    cp=folder/'ct-progress.json';done=json.loads(cp.read_text())['tiles'] if cp.exists() else 0
    approved=load_inference_approval(case,(h,w)) if 'approval_mask_file' in case else None
    tiles=[(y,x) for y in range(0,h,128) for x in range(0,w,128)]
    if approved is not None:
        # Preserve native training context around approved pixels while avoiding
        # CT rendering of distant, unreviewed parts of the canvas.
        tiles=[(y,x) for y,x in tiles if approved[max(0,y-128):min(h,y+256),max(0,x-128):min(w,x+256)].any()]
    def one(item):
        y,x=item;ys=np.arange(y,min(y+128,h));xs=np.arange(x,min(x+128,w))
        stack,good=s.render(v,ys,xs,pitch,np.arange(-32,33)*pitch)
        good&=stack.any(0);return y,x,stack,good
    started=time.time()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for i,(y,x,stack,good) in enumerate(bounded_map(pool,one,tiles[done:],8),done+1):
            hh,ww=good.shape;a[:,y:y+hh,x:x+ww]=stack;valid[y:y+hh,x:x+ww]=good
            if i%64==0 or i==len(tiles):
                a.flush();valid.flush();atomic(cp,dict(tiles=i,total=len(tiles),shape=[65,h,w]));print('CT',case['id'],i,'/',len(tiles),'elapsed',round(time.time()-started),flush=True)
    Image.fromarray(np.asarray(valid)*255).save(folder/'geometry-valid.png')
    for filename in ['source-state.png','validation.png']:Image.new('L',(w,h)).save(folder/filename)
    count=int(valid.sum());record=dict(shape=[65,h,w],valid_pixels=count,supported_uv_area_cm2=count*(pitch/10000)**2)
    if approved is not None:
        record.update({k:case[k] for k in ['approval_revision','approval_geometry_fingerprint','approval_mask_sha256']})
    atomic(complete,record)
    del a,valid;gp.unlink()
def load_model():
    # Called only after acquiring the shared lock, including before CUDA initialization.
    import torch
    sys.path.insert(0,str(ROOT/'canonical-code'))
    from model_resnet3d_3d_decoder import RegressionModel,_extract_state_dict,_strip_known_prefixes
    torch.set_num_threads(8)
    checkpoint=torch.load(ROOT/'models/canonical/r152_3ddec_v2_l5_epoch13.ckpt',map_location='cpu',weights_only=False)
    state=_strip_known_prefixes(_extract_state_dict(checkpoint));model=RegressionModel(with_norm=any(k.startswith('normalization.') for k in state));model.load_state_dict(state,strict=True)
    del checkpoint,state
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    return model.eval().to('cuda')
def gpu_case(case,model,batch=32):
    import torch
    import torch.nn.functional as F
    folder=OUT/case['id'];ct=json.loads((folder/'ct-ready.json').read_text());lh,lw=ct['shape'][1:];s=make_surface(case);v=Volume(case['volume_url']+'0/',cache_gb=24,workers=16)
    approved=load_inference_approval(case,(lh,lw))
    if any(ct.get(k)!=case[k] for k in ['approval_revision','approval_geometry_fingerprint','approval_mask_sha256']):
        raise RuntimeError('Prepared CT does not belong to this saved approval revision; use a new output case')
    h,w=s.shape(case['hr_um']);xs=grid(w);ys=grid(h);weight=np.outer(np.hanning(256),np.hanning(256)).astype(np.float32);weight/=weight.sum()
    for sign in case['normal_signs']:
        name='canonical-forward' if sign==1 else 'canonical-reverse';teacher=folder/(name+'.png')
        if teacher.exists():continue
        s.sign=sign;cp=folder/(name+'-progress.json');start=json.loads(cp.read_text())['rows'] if cp.exists() else 0
        ap=folder/(name+'-sum.npy');wp=folder/(name+'-weight.npy')
        sums=np.lib.format.open_memmap(ap,mode='r+' if ap.exists() else 'w+',dtype=np.float32,shape=(h,w));weights=np.lib.format.open_memmap(wp,mode='r+' if wp.exists() else 'w+',dtype=np.float32,shape=(h,w))
        def one(item):
            y,x=item;hh=min(256,h-y);ww=min(256,w-x);offsets=np.arange(-31,31) if sign==1 else np.arange(-30,32)
            allowed=approved_tile(approved,y,x,hh,ww,case['hr_um'],case['target_um'])
            if not allowed.any():return y,x,None,None
            image,good=s.render(v,np.arange(y,y+hh),np.arange(x,x+ww),case['hr_um'],offsets*case['hr_um']);good&=image.any(0)
            good&=allowed
            if not good.any():return y,x,None,None
            data=np.zeros((62,256,256),np.uint8);data[:,:hh,:ww]=image;mask=np.zeros((256,256),bool);mask[:hh,:ww]=good;return y,x,data,mask
        def predict(items):
            items=[i for i in items if i[2] is not None]
            if not items:return
            data=np.minimum(np.stack([i[2] for i in items]),200).astype(np.float32)/200
            x=torch.from_numpy(data[:,None]).to('cuda')
            with torch.inference_mode(),torch.autocast('cuda'):
                pred=torch.sigmoid(model(x))
            pred=F.interpolate(pred.float(),size=(256,256),mode='bilinear',align_corners=False).cpu().numpy()[:,0]
            for (y,x,_,valid),p in zip(items,pred):
                hh=min(256,h-y);ww=min(256,w-x);wt=weight[:hh,:ww]*valid[:hh,:ww]
                row_sum[:hh,x:x+ww]+=p[:hh,:ww]*wt;row_weight[:hh,x:x+ww]+=wt
        backup=folder/(name+'-pending-row.npz')
        if backup.exists():
            old=np.load(backup);r=int(old['row']);y=int(old['y']);hh=len(old['sum'])
            if r>=start:
                sums[y:y+hh]=old['sum'];weights[y:y+hh]=old['weight'];sums.flush();weights.flush()
            old.close();backup.unlink()
        with concurrent.futures.ThreadPoolExecutor(12) as pool:
            for row in range(start,len(ys)):
                row_started=time.time()
                yy=ys[row];hh=min(256,h-yy);row_sum=np.zeros((hh,w),np.float32);row_weight=np.zeros_like(row_sum)
                items=[]
                for item in bounded_map(pool,one,((ys[row],x) for x in xs),batch*2):
                    items.append(item)
                    if len(items)==batch:predict(items);items=[]
                predict(items)
                with backup.with_suffix('.tmp').open('wb') as f:np.savez(f,row=row,y=yy,sum=np.asarray(sums[yy:yy+hh]),weight=np.asarray(weights[yy:yy+hh]))
                backup.with_suffix('.tmp').replace(backup)
                sums[yy:yy+hh]+=row_sum;weights[yy:yy+hh]+=row_weight;sums.flush();weights.flush();atomic(cp,dict(rows=row+1,total_rows=len(ys),tiles_per_row=len(xs),highres_shape=[h,w],sign=sign));backup.unlink()
                print('INK',case['id'],name,'row',row+1,'/',len(ys),'tiles/row',len(xs),'seconds',round(time.time()-row_started,2),flush=True)
        # Smooth probabilities before center-aligned downsampling. Process stripes
        # with a halo so resampling never shifts the UV origin or needs full RAM.
        output=np.zeros((lh,lw),np.uint8);validout=np.zeros((lh,lw),np.uint8);ratio=case['target_um']/case['hr_um'];sigma=max(0,(ratio-1)/2);halo=int(np.ceil(4*sigma))+2
        for y in range(0,lh,128):
            yy=(np.arange(y,min(y+128,lh))+.5)*ratio-.5;xx=(np.arange(lw)+.5)*ratio-.5;low=max(0,int(np.floor(yy.min()))-halo);high=min(h,int(np.ceil(yy.max()))+halo+1)
            ww=np.asarray(weights[low:high]);q=np.divide(np.asarray(sums[low:high]),ww,out=np.zeros_like(ww),where=ww>0)
            support=(ww>0).astype(np.float32);q=gaussian_filter(q,sigma,mode='nearest');den=gaussian_filter(support,sigma,mode='nearest');q=np.divide(q,den,out=np.zeros_like(q),where=den>0)
            coords=np.meshgrid(yy-low,xx,indexing='ij');vals=map_coordinates(q,coords,order=1,mode='constant',cval=0,prefilter=False);ok=map_coordinates(support,coords,order=0,mode='constant',cval=0,prefilter=False)>0
            output[y:y+len(yy)]=np.rint(vals*255).clip(0,255).astype(np.uint8);validout[y:y+len(yy)]=ok*255
        geometry=np.asarray(Image.open(folder/'geometry-valid.png'))>0
        # Antialiasing can spread scores over a mask boundary; enforce the exact
        # approved low-resolution support again after the final resampling.
        output[~(geometry&approved)]=0;validout[~(geometry&approved)]=0
        Image.fromarray(output).save(teacher);Image.fromarray(validout).save(folder/(name+'-valid.png'))
        del sums,weights;ap.unlink();wp.unlink()
    publish_metadata(case,folder,ct)
def publish_metadata(case,folder,ct):
    pitch=case['target_um'];h,w=ct['shape'][1:];teachers=[]
    for sign in case['normal_signs']:
        name='canonical-forward' if sign==1 else 'canonical-reverse';teachers.append(dict(id=name,name='New canonical · '+('forward' if sign==1 else 'reversed depth'),file=name+'.png',valid_file=name+'-valid.png',model='scrollprize/ink_canonical_2um',checkpoint_sha256='36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0',normal_sign=sign))
    meta=dict(case,shape=ct['shape'],spacing_um=[pitch]*3,split_group=case['scroll'],source_spacing_um=case['hr_um'],source_url=case['volume_url'],source_coverage='complete grown or published surface; unlabelled regions remain unknown',source_counts=dict(ink=0,background=0,validation=0,valid=ct['valid_pixels']),valid_area_cm2=ct['supported_uv_area_cm2'],canvas_area_cm2=h*w*(pitch/10000)**2,focus=[w//2,h//2],teachers=teachers,default_teacher=teachers[0]['id'],training_split_policy='donor',published_training_pixels=0,alignment_confidence='High confidence in common CT/teacher UV coordinates; segmentation seating and ink quality require manual review. Acquisition remains high-resolution CT resampled to 9.36 um, not the native low-resolution scan.',native_domain_confidence='Resampled source; native acquisition equivalence is unverified.',method='Bilinear 3D surface; canonical R152 at measured high resolution in both depth orientations; Gaussian-filtered probabilities resampled at exact physical pixel centers; 65-plane low-resolution CT from an area-mean raw pyramid.',sampling=dict(uv_origin='first mesh vertex',pixel_centers='(index+0.5)*pitch_um',ct_offsets_um=(np.arange(-32,33)*pitch).tolist(),teacher_offsets_voxels=list(range(-31,31)),raw_pyramid_level=2,raw_pyramid_center_offset_L0=1.5),label_alignment_note='CT and new predictions use one 3D mesh and one UV origin. Teacher scores are uncalibrated predictions; unknown pixels must be excluded from training loss.')
    meta['ink_quality_confidence']='Unreviewed model predictions. Extracted area is not verified readable ink area; inspect and refine before training.'
    meta['readability_status']='unreviewed'
    if case['id']=='841-expanded-layer-01':
        meta['readability_status']='rejected'
        meta['readability_evidence']='User rejected the complete forward prediction: no recognizable letter forms. Assistant also found no convincing letters in reversed depth.'
        meta['ink_quality_confidence']='Rejected as an ink-label teacher; neither orientation demonstrates recognizable text.'
        teachers[0]['ink_quality_confidence']='Rejected by user: no recognizable letter forms.'
    atomic(folder/'dataset.json',meta)
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cpu','gpu']);p.add_argument('--ids',nargs='*');p.add_argument('--batch',type=int,default=32);p.add_argument('--cases-file');args=p.parse_args();cases=json.loads((Path(args.cases_file) if args.cases_file else ROOT/'expansion-cases.json').read_text())
    if args.ids:
        by_id={c['id']:c for c in cases};cases=[by_id[ident] for ident in args.ids]
    OUT.mkdir(exist_ok=True)
    if args.stage=='cpu':
        for case in cases:cpu_case(case)
    else:
        require_review_targets()
        gate=Path(os.environ.get('STUDIO_GPU_LEASE_FILE',str(Path.home()/'.cache/vesuvius-gpu/donor-priority.json')))
        if not gate.exists() or json.loads(gate.read_text()).get('status')!='running' or 'DONOR_GPU_LOCK_FD' not in os.environ:raise RuntimeError('Start GPU stage through the lock-owning launcher')
        if any(not (OUT/c['id']/'ct-ready.json').exists() for c in cases):raise RuntimeError('All selected CT inputs must be ready before GPU ownership')
        os.fstat(int(os.environ['DONOR_GPU_LOCK_FD']))
        # Approval validation precedes CUDA initialization and model loading.
        for case in cases:
            ct=json.loads((OUT/case['id']/'ct-ready.json').read_text())
            load_inference_approval(case,ct['shape'][1:])
            if any(ct.get(k)!=case[k] for k in ['approval_revision','approval_geometry_fingerprint','approval_mask_sha256']):
                raise RuntimeError('Prepared CT approval revision mismatch: '+case['id'])
        model=load_model()
        for case in cases:gpu_case(case,model,args.batch)
if __name__=='__main__':main()
