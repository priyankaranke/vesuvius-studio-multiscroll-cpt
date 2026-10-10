"""Bounded remote Zarr sampling and explicit bilinear surface coordinates."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import concurrent.futures,json,threading,time
from collections import OrderedDict
import requests,numpy as np,numcodecs
from scipy.ndimage import map_coordinates
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
local=threading.local()
def fetch(url):
    if not hasattr(local,'session'):local.session=requests.Session()
    for attempt in range(6):
        try:
            r=local.session.get(url,timeout=(20,90))
            if r.status_code==404:return None
            r.raise_for_status();return r.content
        except requests.RequestException:
            if attempt==5:raise
            time.sleep(min(20,2**attempt))
class Volume:
    def __init__(self,url,cache_gb=2,workers=4):
        self.url=url.rstrip('/')+'/';self.meta=json.loads(fetch(self.url+'.zarray'));self.shape=np.array(self.meta['shape']);self.chunks=np.array(self.meta['chunks']);self.dtype=np.dtype(self.meta['dtype']);self.cache=OrderedDict();self.bytes=0;self.limit=int(cache_gb*1e9);self.lock=threading.Lock();self.fetchlocks=[threading.Lock() for _ in range(64)];self.pool=concurrent.futures.ThreadPoolExecutor(workers)
        self.codec=numcodecs.get_codec(self.meta['compressor']) if self.meta.get('compressor') else None
    def get_chunk(self,key):
        key=tuple(key)
        with self.fetchlocks[hash(key)%64]:
            with self.lock:
                if key in self.cache:
                    a=self.cache.pop(key);self.cache[key]=a;return a
            sep=self.meta.get('dimension_separator','.');b=fetch(self.url+sep.join(map(str,key)))
            if b is None:a=np.zeros(tuple(self.chunks),self.dtype)
            else:
                if self.codec:b=self.codec.decode(b)
                a=np.frombuffer(b,self.dtype).reshape(tuple(self.chunks))
            with self.lock:
                self.cache[key]=a;self.bytes+=a.nbytes
                while self.bytes>self.limit and len(self.cache)>1:
                    _,old=self.cache.popitem(last=False);self.bytes-=old.nbytes
            return a
    def block(self,low,high):
        low=np.maximum(0,np.asarray(low,dtype=int));high=np.minimum(self.shape,np.asarray(high,dtype=int));out=np.zeros(tuple(high-low),self.dtype)
        lo=low//self.chunks;hi=(high+self.chunks-1)//self.chunks
        keys=[(z,y,x) for z in range(lo[0],hi[0]) for y in range(lo[1],hi[1]) for x in range(lo[2],hi[2])]
        for key,a in zip(keys,self.pool.map(self.get_chunk,keys)):
            origin=np.array(key)*self.chunks;b=np.maximum(low,origin);e=np.minimum(high,origin+self.chunks)
            out[tuple(slice(i,j) for i,j in zip(b-low,e-low))]=a[tuple(slice(i,j) for i,j in zip(b-origin,e-origin))]
        return out
    def sample(self,xyz):
        shape=xyz.shape[:-1];points=np.asarray(xyz,np.float32).reshape(-1,3)[:,::-1];valid=np.isfinite(points).all(1)&(points>=0).all(1)&(points<=self.shape-1).all(1);out=np.zeros(len(points),np.float32)
        if valid.any():
            out[valid]=self._sample_valid(points[valid])
        return out.reshape(shape)
    def _sample_valid(self,p,max_block_voxels=200_000_000):
        lower=p.min(0);upper=p.max(0)
        low=np.floor(lower).astype(int);high=np.minimum(self.shape,np.ceil(upper).astype(int)+1)
        if np.prod(high-low)<=max_block_voxels:
            a=self.block(low,high)
            return map_coordinates(a,(p-low.astype(np.float32)).T,output=np.float32,order=1,mode='nearest',prefilter=False)
        # Bent or disconnected parts of a UV tile can span a huge empty CT
        # box. Partition reads spatially, preserving every sample and its order.
        axis=int(np.argmax(upper-lower));left=p[:,axis]<=(lower[axis]+upper[axis])*.5
        if left.all() or not left.any():raise ValueError('Cannot partition oversized CT sampling footprint')
        out=np.empty(len(p),np.float32)
        out[left]=self._sample_valid(p[left],max_block_voxels)
        out[~left]=self._sample_valid(p[~left],max_block_voxels)
        return out
class Surface:
    def __init__(self,path,geometry_um,ct_um,matrix=None,normal_sign=1,geometry_shape=None):
        from pathlib import Path
        p=Path(path);self.meta=json.loads((p/'meta.json').read_text());self.a=np.stack([np.asarray(Image.open(p/(k+'.tif')),np.float32) for k in 'xyz'],-1)
        self.valid=np.isfinite(self.a).all(-1)&(self.a>=0).all(-1)
        if geometry_shape is not None:self.valid&=(self.a<np.asarray(geometry_shape)[::-1]).all(-1)
        if matrix is not None:
            m=np.asarray(matrix,np.float32);self.a=self.a@m[:,:3].T+m[:,3]
        self.scale=np.asarray(self.meta['scale']);self.pitch=np.array([geometry_um/self.scale[0],geometry_um/self.scale[1]])
        self.ct_um=ct_um;self.sign=normal_sign
    def shape(self,pitch_um):return (int(np.ceil((self.a.shape[0]-1)*self.pitch[1]/pitch_um)),int(np.ceil((self.a.shape[1]-1)*self.pitch[0]/pitch_um)))
    def points(self,ys,xs,pitch_um):
        # All products use this same physical UV origin and pixel-center convention.
        gy,gx=np.meshgrid((np.asarray(ys,dtype=np.float32)+.5)*np.float32(pitch_um/self.pitch[1]),(np.asarray(xs,dtype=np.float32)+.5)*np.float32(pitch_um/self.pitch[0]),indexing='ij')
        iy=np.floor(gy).astype(int);ix=np.floor(gx).astype(int);valid=(iy>=0)&(ix>=0)&(iy<self.a.shape[0]-1)&(ix<self.a.shape[1]-1)
        iy=np.clip(iy,0,self.a.shape[0]-2);ix=np.clip(ix,0,self.a.shape[1]-2);v=gy-iy.astype(np.float32);u=gx-ix.astype(np.float32)
        a,b,c,d=self.a[iy,ix],self.a[iy,ix+1],self.a[iy+1,ix],self.a[iy+1,ix+1]
        valid&=self.valid[iy,ix]&self.valid[iy,ix+1]&self.valid[iy+1,ix]&self.valid[iy+1,ix+1]
        xyz=(1-u)[...,None]*(1-v)[...,None]*a+u[...,None]*(1-v)[...,None]*b+(1-u)[...,None]*v[...,None]*c+u[...,None]*v[...,None]*d
        dx=(1-v)[...,None]*(b-a)+v[...,None]*(d-c);dy=(1-u)[...,None]*(c-a)+u[...,None]*(d-b)
        n=np.cross(dy,dx);length=np.linalg.norm(n,axis=-1);valid&=length>1e-6;n*=self.sign/np.maximum(length,1e-6)[...,None]
        # Exclude geometric bridges over holes, without filtering CT appearance.
        max_edge=np.maximum.reduce([np.linalg.norm(b-a,axis=-1),np.linalg.norm(c-a,axis=-1),np.linalg.norm(d-b,axis=-1),np.linalg.norm(d-c,axis=-1)])
        valid&=max_edge<5*max(self.pitch)/self.ct_um
        xyz[~valid]=-1;n[~valid]=0
        return xyz,n,valid
    def render(self,volume,ys,xs,pitch_um,offsets_um):
        xyz,n,valid=self.points(ys,xs,pitch_um)
        offsets=np.asarray(offsets_um,np.float32)/self.ct_um
        coords=xyz[None]+n[None]*offsets[:,None,None,None];coords[:,~valid]=-1
        out=np.rint(volume.sample(coords)).clip(0,255).astype(np.uint8)
        return out,valid
