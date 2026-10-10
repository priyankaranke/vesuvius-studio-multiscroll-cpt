"""Shared native tiles, overlapped CPU fetch, unchanged coupled augmentation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from studio_env import localize

ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent/'pherc1447-architecture-study-20260906'
sys.path.insert(0,str(STUDY))
from scroll_augment import augment as scroll_shift


def masked_terms(logits, categorical):
    z=logits[...,16:112,16:112].float()
    state=categorical[...,16:112,16:112]
    y=(state==2).float();m=(state>0).float();axes=(1,2,3)
    count=m.sum(axes);active=count>0
    bce=(m*F.binary_cross_entropy_with_logits(z,y,reduction='none')).sum(axes)/count.clamp_min(1)
    p=z.sigmoid()
    dice=1-(2*(m*p*y).sum(axes)+1)/((m*p).sum(axes)+(m*y).sum(axes)+1)
    return ((bce+dice)*.5*active).sum(),active.sum()


def augment(raw,state,parameters,augmentation_rng):
    gain,bias,rotation,flip,drop=parameters
    raw=raw.permute(0,3,1,2)[:,None].float()
    state=state[:,None].float()
    gain=torch.as_tensor(gain,device=raw.device,dtype=torch.float32)
    bias=torch.as_tensor(bias,device=raw.device,dtype=torch.float32)
    raw=(raw*gain+bias).clamp_min_(0)
    raw,state=raw.rot90(rotation,[-2,-1]),state.rot90(rotation,[-2,-1])
    if flip:
        raw,state=raw.flip(-1),state.flip(-1)
    if drop is not None:
        raw[:,:,drop]=raw.mean(2)
    # One categorical state plane transports target AND supervision in the
    # exact same nearest-neighbor warp, with no additional random calls.
    raw,state=scroll_shift(raw,state,128,augmentation_rng,'scroll-shift-v2')
    return raw.clamp_max_(200).mul_(1/255).contiguous(),state.contiguous()


def parameters(rng,n=64):
    offsets=rng.integers(0,129,size=(n,2))
    gain=rng.uniform(.8,1.2,(n,1,1,1,1))
    bias=rng.uniform(-15,15,(n,1,1,1,1))
    rotation,flip=int(rng.integers(4)),bool(rng.integers(2))
    drop=int(rng.integers(16)) if rng.random()<.2 else None
    return offsets,(gain,bias,rotation,flip,drop)


class Data:
    def __init__(self,doc):
        self.doc=doc;self.rows=[];self.arenas=[];self.labels=[];self.coords=[];self.lookup=[]
        self.entry_row=np.empty(doc['training_entries'],np.int32)
        self.entry_patch=np.empty(doc['training_entries'],np.int32)
        self.workers=ThreadPoolExecutor(8)
        total=0
        for row in doc['segments']:
            if not row['samples']:
                continue
            path=localize(row['folder']);index=len(self.rows);self.rows.append(row)
            tiles=np.load(path/'tiles.npy')
            values=np.load(path/'ct-tiles.npy',mmap_mode='r')
            assert values.shape==(len(tiles),512,512,16) and values.dtype==np.uint8
            # Compact shared tiles fit host memory; eagerly cache once so later
            # epochs never wait on filesystem reads. No352-parent duplication.
            arena=np.array(values,copy=True);self.arenas.append(arena);total+=arena.nbytes
            self.labels.append(np.load(path/'labels.npy',mmap_mode='r'))
            self.coords.append(np.load(path/'coordinates.npy'))
            self.lookup.append({tuple(map(int,t)):i for i,t in enumerate(tiles)})
            if row['role']=='train':
                sl=slice(row['entry_offset'],row['entry_offset']+row['samples'])
                self.entry_row[sl]=index;self.entry_patch[sl]=np.load(path/'entry-to-patch.npy')
            print('TILE_CACHE',row['id'],len(tiles),'total_GiB',round(total/2**30,2),flush=True)
        self.bytes=total

    def crop(self,item):
        index,y,x,width=item;state=self.labels[index];h,w=state.shape
        ys=np.arange(y,y+width);xs=np.arange(x,x+width)
        # Match the baseline's reflected measured CT and zero label padding.
        def reflect(a,n):
            a=a%(2*n-2)
            return np.where(a<n,a,2*n-2-a)
        yy=reflect(ys,h);xx=reflect(xs,w)
        raw=np.empty((width,width,16),np.uint8)
        for ty in np.unique(yy//512*512):
            iy=np.flatnonzero(yy//512*512==ty)
            for tx in np.unique(xx//512*512):
                ix=np.flatnonzero(xx//512*512==tx)
                tile=self.arenas[index][self.lookup[index][(int(ty),int(tx))]]
                raw[np.ix_(iy,ix)]=tile[np.ix_(yy[iy]-ty,xx[ix]-tx)]
        label=np.zeros((width,width),np.uint8)
        iy=np.flatnonzero((ys>=0)&(ys<h));ix=np.flatnonzero((xs>=0)&(xs<w))
        label[np.ix_(iy,ix)]=state[np.ix_(ys[iy],xs[ix])]
        return raw,label

    def fetch(self,entries,offsets,pin=True):
        items=[]
        for entry,offset in zip(entries,offsets):
            index=int(self.entry_row[entry]);y,x=self.coords[index][self.entry_patch[entry]]
            items.append((index,int(y)-112+int(offset[0]),int(x)-112+int(offset[1]),224))
        pairs=list(self.workers.map(self.crop,items))
        raw=torch.from_numpy(np.stack([p[0] for p in pairs]));state=torch.from_numpy(np.stack([p[1] for p in pairs]))
        return (raw.pin_memory(),state.pin_memory()) if pin else (raw,state)

    def validation(self):
        index=next(i for i,r in enumerate(self.rows) if r['role']=='validation')
        coords=self.coords[index]
        pairs=list(self.workers.map(self.crop,[(index,int(y),int(x),128) for y,x in coords]))
        return (torch.from_numpy(np.stack([p[0] for p in pairs])),
                torch.from_numpy(np.stack([p[1] for p in pairs])),coords)


class GpuData:
    """All shared measured tiles and categorical tiles fit in GH200 VRAM."""
    def __init__(self,doc):
        self.doc=doc;self.rows=[r for r in doc['segments'] if r['samples']]
        count=sum(r['tiles'] for r in self.rows);required=count*512*512*17
        assert required+36*2**30 < torch.cuda.mem_get_info()[0], 'Use bounded host loader if shared tiles exceed VRAM budget'
        self.raw=torch.empty((count,512,512,16),dtype=torch.uint8,device='cuda')
        self.state=torch.empty((count,512,512),dtype=torch.uint8,device='cuda')
        h=max((r['shape'][0]+511)//512 for r in self.rows);w=max((r['shape'][1]+511)//512 for r in self.rows)
        lookup=np.full((len(self.rows),h,w),-1,np.int32)
        self.coords=[];entry_row=np.empty(doc['training_entries'],np.int32);entry_yx=np.empty((doc['training_entries'],2),np.int32)
        start=0
        for i,row in enumerate(self.rows):
            path=localize(row['folder']);tiles=np.load(path/'tiles.npy');values=np.load(path/'ct-tiles.npy',mmap_mode='r')
            labels=np.load(path/'labels.npy',mmap_mode='r');coords=np.load(path/'coordinates.npy');self.coords.append(coords)
            for k,(y,x) in enumerate(tiles):lookup[i,y//512,x//512]=start+k
            for offset in range(0,len(tiles),32):
                n=min(32,len(tiles)-offset)
                raw=torch.from_numpy(np.array(values[offset:offset+n],copy=True))
                categorical=np.zeros((n,512,512),np.uint8)
                for j,(y,x) in enumerate(tiles[offset:offset+n]):
                    block=labels[y:y+512,x:x+512];categorical[j,:block.shape[0],:block.shape[1]]=block
                self.raw[start+offset:start+offset+n].copy_(raw)
                self.state[start+offset:start+offset+n].copy_(torch.from_numpy(categorical))
            if row['role']=='train':
                sl=slice(row['entry_offset'],row['entry_offset']+row['samples'])
                entry_row[sl]=i;entry_yx[sl]=coords[np.load(path/'entry-to-patch.npy')]
            start+=len(tiles)
            print('GPU_SHARED_TILES',row['id'],start,'of',count,flush=True)
        self.lookup=torch.from_numpy(lookup).cuda()
        self.entry_row=entry_row;self.entry_yx=entry_yx
        self.shapes=torch.tensor([r['shape'] for r in self.rows],device='cuda')
        self.bytes=required

    def crops(self,indices,yx,width):
        indices=torch.as_tensor(indices,device='cuda',dtype=torch.long)
        yx=torch.as_tensor(yx,device='cuda',dtype=torch.long)
        ys=yx[:,0,None]+torch.arange(width,device='cuda');xs=yx[:,1,None]+torch.arange(width,device='cuda')
        h,w=self.shapes[indices].unbind(1)
        outside=(ys[:,:,None]<0)|(ys[:,:,None]>=h[:,None,None])|(xs[:,None,:]<0)|(xs[:,None,:]>=w[:,None,None])
        def reflect(a,n):
            a=a%(2*n[:,None]-2)
            return torch.where(a<n[:,None],a,2*n[:,None]-2-a)
        yy=reflect(ys,h);xx=reflect(xs,w)
        tile=self.lookup[indices[:,None,None],yy[:,:,None]//512,xx[:,None,:]//512].long()
        # Missing tiles are an implementation error, never an implicit CT fill.
        if bool((tile<0).any()):raise RuntimeError('Unprepared input context tile')
        raw=self.raw[tile,yy[:,:,None]%512,xx[:,None,:]%512]
        state=self.state[tile,yy[:,:,None]%512,xx[:,None,:]%512].masked_fill_(outside,0)
        return raw,state

    def fetch(self,entries,offsets):
        return self.crops(self.entry_row[entries],self.entry_yx[entries]-112+offsets,224)

    def validation_batch(self,start,count=16):
        index=next(i for i,r in enumerate(self.rows) if r['role']=='validation')
        coords=self.coords[index][start:start+count]
        return self.crops(np.full(len(coords),index),coords,128)

    def validation_coords(self):
        return self.coords[next(i for i,r in enumerate(self.rows) if r['role']=='validation')]
