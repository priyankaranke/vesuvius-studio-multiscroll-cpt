"""Bounded reconstruction of shared512 tiles using accepted source samplers."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import tifffile
from prepare import ROOT,HOST,STUDIO,DONOR,BASE,read,atomic,native
from studio_env import localize

IMAGE='ghcr.io/scrollprize/villa/volume-cartographer@sha256:bad516f66001abca759454cc43e4fd11e5b19aa55d36bdc2043817291c8083c4'
# the run's vc_render_tifxyz build (HF data: render-runtime/); originally first-letters-ten-20260909/pipeline/render-runtime
RUNTIME=Path(os.environ.get('STUDIO_RENDER_RUNTIME') or HOST/'render-runtime').resolve()
DOCKER=os.environ.get('STUDIO_DOCKER','docker').split()   # e.g. 'sudo -n docker' where docker needs root


def campaign_tile(row,y,x,stage):
    g=row['geometry'];oy,ox=g['offset_yx'];h,w=row['shape']
    hh,ww=min(512,h-y),min(512,w-x)
    target=stage/'planes'
    # CLI crop is expressed in the unchanged full parent's UV frame. No mesh
    # crop, normalization, flattening, rotation or shape recomputation occurs.
    command=[*DOCKER,'run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
             '--cpus','2','--memory','4g','--memory-swap','4g','--label','vesuvius.task=donor-studio-training-20260926',
             '-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=1','-e','CUDA_VISIBLE_DEVICES=',
             '-v',f'{STUDIO}:/production:ro','-v',f'{ROOT}:/training','-v',f'{RUNTIME}:/runtime:ro',
             '--entrypoint','/runtime/vc_render_tifxyz',IMAGE,
             '-v','/production/'+str(localize(g['scan']['destination']).relative_to(STUDIO)),
             '-g','0','--scale','1','-s','/production/'+str(localize(g['mesh']).relative_to(STUDIO)),
             '-n','17','--slice-step','1','--pyramid','0',
             '--tif-output','/training/'+str(target.relative_to(ROOT)),
             '--cache-gb','1','--sampling-column-block','128','--chunks-already-staged',
             '--voxel-size',str(g['pitch']),'--voxel-unit','micrometer',
             '--crop-x',str(int(x+ox)),'--crop-y',str(int(y+oy)),'--crop-width',str(ww),'--crop-height',str(hh)]
    with (stage/'render.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    raw=np.stack([tifffile.imread(target/f'{i:02d}.tif')[:hh,:ww] for i in range(17)])
    assert raw.shape==(17,hh,ww) and raw.dtype==np.uint8
    if g['direction']=='reverse':raw=raw[::-1]
    result=np.zeros((512,512,16),np.uint8);result[:hh,:ww]=raw[:16].transpose(1,2,0)
    shutil.rmtree(target)
    return result


def original_source(row):
    g=row['geometry'];old=g['baseline'];s,_=native.surface_for(old,read(DONOR/'volume-catalog.json'))
    if s is None:
        if row['id']=='814-native':
            # Exact released28 source; baseline selected centered planes6:22.
            from volume_sampling import Volume
            volume=Volume(old['source_meta']['source_url'],cache_gb=.3,workers=8)
            assert volume.shape.tolist()==[28,*row['shape']]
            def fetch(y,x,hh,ww):
                return volume.block([6,y,x],[22,y+hh,x+ww])
            return fetch,volume
        ident=row['id'].removeprefix('source-').split('-approved-')[0]
        source=np.load(HOST/'donor-labels-20260906/prepared'/ident/'ct.npy',mmap_mode='r')
        source=source[len(source)//2-8:len(source)//2+8]
        if (old.get('orientation') or {}).get('normal_sign')==-1:source=source[::-1]
        if old.get('runtime_reverse'):source=source[::-1]
        return lambda y,x,hh,ww:np.array(source[:,y:y+hh,x:x+ww]),None
    # The interpolation and quantization are the original accepted sampler;
    # replace only transport with the already inventoried native CT cache.
    import volume_sampling,fast_volume_sampling
    remote_fetch=volume_sampling.fetch
    def local_fetch(url):
        if url.startswith('/'):
            path=Path(url)
            return path.read_bytes() if path.exists() else None
        return remote_fetch(url)
    volume_sampling.fetch=local_fetch;fast_volume_sampling.fetch=local_fetch
    scroll={'PHerc0814':'814','PHerc0841':'841','PHerc0009B':'9B'}[row['scroll']]
    scan=read(STUDIO/scroll/'config/scan.json')
    assert g['recipe']['volume_id']==scan['scan_id'] and g['recipe']['level']==0
    volume=native.FastVolume(str(localize(scan['destination'])/'0'),cache_gb=1,workers=4)
    def fetch(y,x,hh,ww):
        a,valid=s.render(volume,np.arange(y,y+hh),np.arange(x,x+ww),g['recipe']['xy_pitch_um'],g['recipe']['offsets_um'])
        a[:,~valid]=0
        if old.get('runtime_reverse'):a=a[::-1]
        return a
    return fetch,volume


def render_row(row):
    folder=localize(row['folder']);ready=folder/'render.json'
    if ready.exists() and read(ready).get('status')=='complete':return row['id']
    coordinates=np.load(folder/'tiles.npy');path=folder/'ct-tiles.npy'
    data=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.uint8,shape=(len(coordinates),512,512,16))
    progress=read(ready) if ready.exists() else {};done=progress.get('tiles_done',0)
    began=time.monotonic();stage=folder/'render-scratch';stage.mkdir(exist_ok=True)
    fetch=volume=None
    if row['geometry']['kind']=='original':fetch,volume=original_source(row)
    try:
        for index,(y,x) in enumerate(coordinates[done:],done):
            y,x=int(y),int(x)
            if fetch is None:
                if (stage/'planes').exists():shutil.rmtree(stage/'planes')
                a=campaign_tile(row,y,x,stage)
            else:
                h,w=row['shape'];hh,ww=min(512,h-y),min(512,w-x)
                a=np.zeros((512,512,16),np.uint8);a[:hh,:ww]=fetch(y,x,hh,ww).transpose(1,2,0)
            data[index]=a
            if (index+1)%16==0 or index+1==len(coordinates):
                data.flush();atomic(ready,dict(status='rendering',tiles_done=index+1,total_tiles=len(coordinates),seconds=time.monotonic()-began))
                print('RENDER',row['id'],index+1,len(coordinates),round(time.monotonic()-began,1),flush=True)
        data.flush()
        atomic(ready,dict(status='complete',tiles_done=len(coordinates),total_tiles=len(coordinates),seconds=time.monotonic()-began,
                         recipe=row['geometry']['recipe'],layout='NHWD uint8; already in model depth order',
                         renderer_sha256=hashlib.sha256((RUNTIME/'vc_render_tifxyz').read_bytes()).hexdigest() if fetch is None else None))
        shutil.rmtree(stage)
    finally:
        if volume is not None:volume.pool.shutdown(wait=True)
    return row['id']


def main(workers):
    doc=read(ROOT/'data.json')
    required=doc['storage']['ct_tiles_bytes']+doc['storage']['checkpoint_reserve_bytes']+2*2**30
    assert shutil.disk_usage(ROOT).free>required+100*2**30,'Preserve100GiB reserve before rendering'
    rows=[r for r in doc['segments'] if r['samples']]
    # Required final dataset tiles, not disposable performance runs.
    with ProcessPoolExecutor(workers) as pool:
        for name in pool.map(render_row,sorted(rows,key=lambda r:r['tiles'])):
            print('READY',name,flush=True)
    doc['status']='complete';atomic(ROOT/'data.json',doc)
    print('ALL_NATIVE_INPUTS_READY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);args=p.parse_args();main(args.workers)
