"""Exact Zarr sampling with shared in-flight chunk requests and bounded RAM."""
import concurrent.futures
import time
import numpy as np
from scipy.ndimage import map_coordinates
from volume_sampling import Volume,fetch


class FastVolume(Volume):
    def __init__(self,url,cache_gb=8,workers=24):
        super().__init__(url,cache_gb=cache_gb,workers=workers)
        self.inflight={}
        self.stats=dict(cache_hits=0,joined_requests=0,chunk_misses=0,compressed_bytes=0,decoded_bytes=0,fetch_seconds=0.,decode_seconds=0.,block_seconds=0.,interpolation_seconds=0.)

    def _load(self,key):
        began=time.monotonic()
        try:
            sep=self.meta.get('dimension_separator','.')
            payload=fetch(self.url+sep.join(map(str,key)))
            fetched=time.monotonic();compressed=0 if payload is None else len(payload)
            if payload is None:a=np.zeros(tuple(self.chunks),self.dtype)
            else:
                if self.codec:payload=self.codec.decode(payload)
                a=np.frombuffer(payload,self.dtype).reshape(tuple(self.chunks))
            decoded=time.monotonic()
            with self.lock:
                self.stats['chunk_misses']+=1;self.stats['compressed_bytes']+=compressed;self.stats['decoded_bytes']+=a.nbytes
                self.stats['fetch_seconds']+=fetched-began;self.stats['decode_seconds']+=decoded-fetched
                self.cache[key]=a;self.bytes+=a.nbytes
                while self.bytes>self.limit and len(self.cache)>1:
                    _,old=self.cache.popitem(last=False);self.bytes-=old.nbytes
            return a
        finally:
            with self.lock:self.inflight.pop(key,None)

    def request(self,key):
        key=tuple(key)
        with self.lock:
            if key in self.cache:
                a=self.cache.pop(key);self.cache[key]=a;self.stats['cache_hits']+=1
                return a
            if key in self.inflight:
                self.stats['joined_requests']+=1
                return self.inflight[key]
            future=self.pool.submit(self._load,key);self.inflight[key]=future
            return future

    def get_chunk(self,key):
        value=self.request(key)
        return value.result() if isinstance(value,concurrent.futures.Future) else value

    def block(self,low,high):
        began=time.monotonic()
        low=np.maximum(0,np.asarray(low,dtype=int));high=np.minimum(self.shape,np.asarray(high,dtype=int));out=np.zeros(tuple(high-low),self.dtype)
        lo=low//self.chunks;hi=(high+self.chunks-1)//self.chunks
        keys=[(z,y,x) for z in range(lo[0],hi[0]) for y in range(lo[1],hi[1]) for x in range(lo[2],hi[2])]
        # Cache hits never wait behind network requests in the executor. Every
        # missing chunk has a single shared future across all renderer threads.
        pending=[self.request(key) for key in keys]
        for key,value in zip(keys,pending):
            a=value.result() if isinstance(value,concurrent.futures.Future) else value
            origin=np.array(key)*self.chunks;b=np.maximum(low,origin);e=np.minimum(high,origin+self.chunks)
            out[tuple(slice(i,j) for i,j in zip(b-low,e-low))]=a[tuple(slice(i,j) for i,j in zip(b-origin,e-origin))]
        with self.lock:self.stats['block_seconds']+=time.monotonic()-began
        return out

    def _sample_valid(self,p,max_block_voxels=200_000_000):
        lower=p.min(0);upper=p.max(0)
        low=np.floor(lower).astype(int);high=np.minimum(self.shape,np.ceil(upper).astype(int)+1)
        if np.prod(high-low)<=max_block_voxels:
            a=self.block(low,high);began=time.monotonic()
            result=map_coordinates(a,(p-low.astype(np.float32)).T,output=np.float32,order=1,mode='nearest',prefilter=False)
            with self.lock:self.stats['interpolation_seconds']+=time.monotonic()-began
            return result
        axis=int(np.argmax(upper-lower));left=p[:,axis]<=(lower[axis]+upper[axis])*.5
        if left.all() or not left.any():raise ValueError('Cannot partition oversized CT sampling footprint')
        out=np.empty(len(p),np.float32)
        out[left]=self._sample_valid(p[left],max_block_voxels)
        out[~left]=self._sample_valid(p[~left],max_block_voxels)
        return out
