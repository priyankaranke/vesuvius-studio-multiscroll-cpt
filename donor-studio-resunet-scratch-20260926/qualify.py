"""Decisive sparse-gradient/reduction checks and fixed-recipe kernel preparation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','16')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','16')
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sparse_data import masked_terms,augment,parameters,ROOT
sys.path.insert(0,str(ROOT.parent/'donor-resunet-scratch-20260912'))
from packed_data import NativeCanvas
from small_unets import DeepResidualUNet


def main():
    torch.set_num_threads(16)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(2026090613)
    # Deliberately unequal supervised-example counts across microbatches.
    z=torch.randn(64,1,128,128,device='cuda',requires_grad=True)
    state=torch.zeros_like(z);state[:7,:,30:90,20:80]=1;state[2:6,:,45:70,30:55]=2;state[40:58,:,50:80,50:80]=2
    numerator,n=masked_terms(z,state);(numerator/n).backward();reference=z.grad.clone()
    unknown_max=float(reference[state==0].abs().max());assert unknown_max==0
    z.grad=None
    for i in range(0,64,16):
        term,_=masked_terms(z[i:i+16],state[i:i+16]);(term/n).backward()
    reduction_error=float((reference-z.grad).abs().max());assert reduction_error<1e-8
    zero,count=masked_terms(z,torch.zeros_like(state));assert zero==0 and count==0
    print('MASKED_LOSS_CHECK',unknown_max,reduction_error,flush=True)
    del z,state,reference
    doc=json.loads((ROOT.parent/'donor-resunet-scratch-20260912/data.json').read_text())
    names=['814-wide-20260908-028-approved-r000001','841-more-20260907-113-approved-r000001','9b-wide-20260908-025-approved-r000001']
    pairs=[]
    for name in names:
        row=next(r for r in doc['segments'] if r['id']==name)
        folder=Path(row['context_folder'])
        values=np.load(folder/'extra-ct.npy',mmap_mode='r')
        coords=np.load(folder/'extra-tile-coordinates.npy')
        label=np.load(ROOT/'prepared'/name/'labels.npy',mmap_mode='r')
        for raw,(y,x) in zip(values[:24],coords[:24]):
            target=np.zeros((128,128),np.uint8)
            block=label[int(y):int(y)+128,int(x):int(x)+128]
            target[:block.shape[0],:block.shape[1]]=block
            pairs.append((raw.copy(),target))
    assert len(pairs)>=64
    pairs=pairs[:64]
    x=torch.from_numpy(np.stack([a for a,b in pairs])).cuda()[:,None].float().clamp_max_(200)/255
    state=torch.from_numpy(np.stack([b for a,b in pairs])).cuda()[:,None].float()
    model=DeepResidualUNet(deeper=False).cuda();assert sum(p.numel() for p in model.parameters())==223355921
    path=ROOT.parent/'donor-resunet-scratch-20260912/checkpoints/last.pt'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest=='a16e462c1f4159537c5c156a55fa0630dbc0781992e7d6e8cd6a1a485ec84d71'
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(saved['state_dict']);del saved
    fn=torch.compile(model,dynamic=False)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=model(x[:16]).float().sigmoid();actual=fn(x[:16]).float().sigmoid()
    difference=float((expected-actual).abs().max());assert difference<=.02,difference
    print('MODEL_CONFIRMED',digest,'max_probability_difference',difference,flush=True)
    del expected,actual
    timings=[]
    # Compile backward once; a second identical-size pass measures the actual
    # uncapped augmentation+micro16 memory demand. No optimizer update occurs.
    for repeat in range(2):
        fn.train();model.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.monotonic()
        count=(state[...,16:112,16:112].flatten(1)>0).any(1).sum()
        for i in range(0,64,16):
            with torch.autocast('cuda',dtype=torch.bfloat16):term,n=masked_terms(fn(x[i:i+16]),state[i:i+16])
            (term/count).backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        torch.cuda.synchronize();timings.append(time.monotonic()-begin)
        print('FIXED_RECIPE_BACKWARD',repeat,timings[-1],float(norm),'peak_GiB',torch.cuda.max_memory_allocated()/2**30,flush=True)
    result=dict(architecture_parameters=223355921,baseline_checkpoint_sha256=digest,
                unknown_gradient_max=unknown_max,microbatch_gradient_max_difference=reduction_error,
                eager_compile_probability_max_difference=difference,backward_seconds=timings,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=0,
                purpose='Sparse loss correctness and unchanged fixed-recipe compiler/memory qualification')
    (ROOT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
