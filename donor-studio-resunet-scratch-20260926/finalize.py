"""Finish alias accounting and check reconstructed input bindings before training."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from prepare import ROOT,read,atomic,decode,native,DONOR


def check(row):
    if not row['samples']:return row
    folder=Path(row['folder']);render=read(folder/'render.json')
    assert render['status']=='complete' and render['tiles_done']==row['tiles']
    tiles=np.load(folder/'tiles.npy');raw=np.load(folder/'ct-tiles.npy',mmap_mode='r');labels=np.load(folder/'labels.npy',mmap_mode='r')
    assert raw.shape==(len(tiles),512,512,16) and raw.dtype==np.uint8
    # Sampling support is geometry/acquisition based; CT intensity never makes
    # new negatives. Inspect every supervised source coordinate against bounds.
    g=row['geometry'];support_failures=0
    if g['kind']=='campaign':
        mesh=Path(g['mesh']);xyz=np.stack([np.asarray(Image.open(mesh/(k+'.tif')),np.float32) for k in 'xyz'],-1)
        scale=np.asarray(read(mesh/'meta.json')['scale'],np.float32)
        for y,x in tiles:
            state=labels[y:y+512,x:x+512];yy,xx=np.nonzero(state)
            if not len(yy):continue
            iy=np.floor((yy+y+g['offset_yx'][0])*scale[1]).astype(int)
            ix=np.floor((xx+x+g['offset_yx'][1])*scale[0]).astype(int)
            iy=np.clip(iy,0,xyz.shape[0]-2);ix=np.clip(ix,0,xyz.shape[1]-2)
            corners=np.stack([xyz[iy,ix],xyz[iy+1,ix],xyz[iy,ix+1],xyz[iy+1,ix+1]])
            # Convex interpolation and unit normals: a nine-voxel bound contains
            # all centered17 samples plus trilinear support. This is conservative.
            supported=(corners>=9).all((0,2))&(corners<np.array(g['scan']['shape_zyx'])[::-1]-10).all((0,2))
            support_failures+=int((~supported).sum())
    else:
        s,_=native.surface_for(g['baseline'],read(DONOR/'volume-catalog.json'))
        if s is not None:
            from prepare import STUDIO
            short={'PHerc0814':'814','PHerc0841':'841','PHerc0009B':'9B'}[row['scroll']]
            bounds=np.array(read(STUDIO/short/'config/scan.json')['shape_zyx'])[::-1]-1
            offsets=np.asarray(g['recipe']['offsets_um'])/s.ct_um
            for y,x in tiles:
                h,w=labels[y:y+512,x:x+512].shape
                p,n,v=s.points(np.arange(y,y+h),np.arange(x,x+w),g['pitch'])
                supported=v.copy()
                for z in (offsets.min(),offsets.max()):
                    q=p+n*z;supported&=(q>=0).all(-1)&(q<=bounds).all(-1)
                support_failures+=int(((labels[y:y+h,x:x+w]>0)&~supported).sum())
    if support_failures:raise ValueError((row['id'],'Supervised pixels need acquisition-support masking',support_failures))
    # Compare source-specific re-rendered tiles with previously measured accepted
    # context tiles. Read only a bounded overlap, respecting saved reversal once.
    if g['kind']=='original':
        old=Path(g['baseline']['context_folder']);coords=np.load(old/'extra-tile-coordinates.npy')
        if len(coords):
            values=np.load(old/'extra-ct.npy',mmap_mode='r');lookup={tuple(t):i for i,t in enumerate(tiles)}
            compared=0
            for i,(y,x) in enumerate(coords):
                key=(int(y)//512*512,int(x)//512*512)
                if key not in lookup or y+128>labels.shape[0] or x+128>labels.shape[1]:continue
                actual=raw[lookup[key],y-key[0]:y-key[0]+128,x-key[1]:x-key[1]+128].transpose(2,0,1)
                expected=values[i][::-1] if g.get('runtime_reverse') else values[i]
                assert np.array_equal(actual,expected),(row['id'],'Accepted context pixels changed',int(np.abs(actual.astype(int)-expected).max()))
                compared+=actual.size
                if compared>=16*128*128:break
            row['accepted_context_compared_values']=compared
    row['supervised_acquisition_support_failures']=support_failures
    unique=np.load(folder/'coordinates.npy');covered=np.zeros(((labels.shape[0]+63)//64,(labels.shape[1]+63)//64),bool)
    for y,x in unique:covered[y//64:y//64+2,x//64:x//64+2]=True
    missing=0
    for y in range(0,labels.shape[0],512):
        mask=np.repeat(np.repeat(covered[y//64:(y+512)//64],64,axis=0),64,axis=1)[:min(512,labels.shape[0]-y),:labels.shape[1]]
        missing+=int(((labels[y:y+512]==1)&~mask).sum())
    row['background_outside_eligible_parent_footprints']=missing
    return row


def main():
    doc=read(ROOT/'data.json');snapshot=read(ROOT/'label-snapshot.json');by={r['id']:r for r in snapshot['datasets']}
    for old,new in [('814','source-814-approved-r000001'),('source-814-native-approved-r000001','814-native')]:
        a,b=decode(by[old]),decode(by[new]);assert not ((a>0)&(a!=b)).any()
        row=next(r for r in doc['segments'] if r['id']==old)
        row.update(excluded_reason='Exact inherited annotation alias; all supervised pixels already retained',alias_of=new,
                   exact_alias_pixels_removed=int((a>0).sum()))
    for row in doc['segments']:
        if row['id'].startswith(('841-grow-','841-w00','source-841-')) and not row['samples']:
            row['excluded_reason']='Released841 source/approved alias has no accepted native physical registration to the fixed holdout; excluded as in accepted baseline'
        if row['id']=='9b' and not row['samples']:
            row['excluded_reason']='Ink annotations duplicate source-9b-approved-r000001; remaining13071 background pixels have no any-ink parent and are omitted by inherited sampler'
    with ProcessPoolExecutor(12) as pool:doc['segments']=list(pool.map(check,doc['segments']))
    doc['storage'].update(renderer_scratch_bound_bytes=2*2**30,ct_download_bytes_bound=250*2**20,
                          ct_download_note='Only missing published814-native28-plane source, bounded250MiB; all volumetric nativeCT already local',
                          gpu_shared_tiles_and_masks_bytes=doc['storage']['ct_tiles_bytes']*17//16)
    doc['baseline_growth']=dict(training_entries=doc['training_entries']/73146,distinct_patches=doc['distinct_training_patches']/25452,
                                saved_supervised_pixels=(102111967+204782564)/(18773013+40179395),not_tenfold=True)
    doc['implementation_changes']=['Sparse supervision explicitly requested; no unknown negatives',
        'Conservative physical exclusion covers full352 input context and CT depth, replacing old center-only test',
        'Exact inherited annotation deduplication; independently labeled reflattened overlaps retained',
        'Shared512 nativeCT and categorical tiles in VRAM rather than duplicated352 parents',
        'Current run_gpu exclusive lease; retired historical prerequisites removed']
    doc['verification']='Categorical counts match pinned revisions; source geometry/checkpoint/depth bindings resolved; supervised acquisition support and accepted original context bytes checked'
    atomic(ROOT/'data.json',doc)
    print('FINALIZED',doc['training_entries'],doc['distinct_training_patches'],doc['total_updates'],flush=True)


if __name__=='__main__':main()
