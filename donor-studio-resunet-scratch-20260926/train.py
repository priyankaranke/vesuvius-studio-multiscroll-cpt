"""Ten complete equal-scroll epochs; sparse loss; final-epoch scratch checkpoint."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','16')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','16')
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
from studio_env import load_env
load_env()
from sparse_data import ROOT,GpuData,augment,parameters,masked_terms
from small_unets import DeepResidualUNet

RUN_ID='three-donor-studio-sparse-scratch-20260926'
# Optional integrations (portability; none is needed to reproduce the run, see .env.example):
#   STUDIO_WANDB=1          log to Weights & Biases (WANDB_ENTITY / WANDB_PROJECT; credentials from `wandb login`)
#   DONOR_SPIRAL_GPU_LEASE_TOKEN (+ STUDIO_GPU_LEASE_FILE)  the original host's exclusive-GPU lease check
#   STUDIO_MAX_UPDATES=N    stop after N optimizer updates, with a resumable checkpoint (0 = full run)
ENTITY,PROJECT=os.environ.get('WANDB_ENTITY') or None,os.environ.get('WANDB_PROJECT','studio-814-841-9b')
MAX_UPDATES=int(os.environ.get('STUDIO_MAX_UPDATES','0') or 0)
STOP=False


class NoRun:
    """Stand-in for a W&B run when logging is off."""
    url=None

    def __init__(self):self.summary={}

    def log(self,*a,**k):pass

    def finish(self):pass


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def epoch_indices(doc,epoch):
    rng=np.random.default_rng(2026091201+epoch);draws=[]
    count=math.ceil(max(doc['entry_counts'].values())/64)*64
    for scroll in ('PHerc0814','PHerc0841','PHerc0009B'):
        pool=np.concatenate([np.arange(r['entry_offset'],r['entry_offset']+r['samples']) for r in doc['segments'] if r['role']=='train' and r['samples'] and r['scroll']==scroll])
        assert len(pool)==doc['entry_counts'][scroll]
        draws.append(np.concatenate([pool,rng.choice(pool,count-len(pool),replace=True)]))
    result=np.concatenate(draws);rng.shuffle(result)
    assert len(np.unique(result))==doc['training_entries']
    return result


def main():
    token=os.environ.get('DONOR_SPIRAL_GPU_LEASE_TOKEN')
    gate=Path(os.environ.get('STUDIO_GPU_LEASE_FILE',str(Path.home()/'.cache/vesuvius-gpu/donor-priority.json')))
    def check_lease():
        if not token:return  # no lease outside the original host
        value=json.loads(gate.read_text())
        assert value['status']=='running' and value['ownership_token']==token
    check_lease()
    doc=json.loads((ROOT/'data.json').read_text());assert doc['status']=='complete'
    qualification=json.loads((ROOT/'qualification.json').read_text())
    assert qualification['unknown_gradient_max']==0 and qualification['eager_compile_probability_max_difference']<=.02
    checkpoint_dir=ROOT/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    config=dict(run_id=RUN_ID,architecture='resunet-deep',architecture_class='DeepResidualUNet(deeper=False)',parameters=223355921,
                initialization='scratch',pretrained_checkpoint=None,epochs=10,effective_batch=64,microbatch=16,accumulation=4,
                precision='bfloat16',implementation='torch.compile(dynamic=False)',loss_implementation='eager',layout='contiguous',
                optimizer='fused AdamW',peak_lr=.0002,weight_decay=.01,gradient_clip_norm=1.,cpu_threads=16,
                training_seed=2026090613,augmentation_seed=2026090719,sampler_seed=2026091201,
                input_depth=16,model_window=128,measured_parent=352,augmentation_context=224,
                augmentation='Unchanged uncapped scroll-shift-v2 and legacy ordering',
                loss='Per-example masked0.5BCE+0.5Dice smooth1 central96; actual contributing-example accumulation',
                validation_metric_core=32,holdout=doc['validation_id'],holdout_exclusion_um=2000,
                training_entries=doc['training_entries'],distinct_patches=doc['distinct_training_patches'],
                entry_counts=doc['entry_counts'],epoch_draws=doc['epoch_draws'],planned_updates=doc['total_updates'],
                data_sha256=hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest(),
                label_snapshot_sha256=hashlib.sha256((ROOT/'label-snapshot.json').read_bytes()).hexdigest(),
                final_checkpoint_policy='Final epoch; no validation-selected checkpoint',s1_samples=0,
                throughput_changes='Shared512 native tiles and categorical masks resident in VRAM; coordinates/RNG/augmentation unchanged',
                learning_rate_formula='2e-4*(.01+.99*min(1.,(step+1)/max(1,steps*.03)))*(.1+.9*(1+cos(pi*min(step,steps)/steps))/2)')
    atomic(ROOT/'config.json',config)
    if os.environ.get('STUDIO_WANDB')=='1':
        import wandb
        run=wandb.init(entity=ENTITY,project=PROJECT,id=RUN_ID,resume='allow',dir=str(ROOT),name='814 / 841 / 9B · Studio sparse labels · scratch',config=config)
    else:
        wandb=None;run=NoRun()
    torch.set_num_threads(16);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.set_float32_matmul_precision('high')
    torch.manual_seed(2026090613)
    model=DeepResidualUNet(deeper=False).cuda();assert sum(p.numel() for p in model.parameters())==223355921
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01,fused=True)
    rng=np.random.default_rng(2026090613);augmentation_rng=np.random.default_rng(2026090719)
    step=epoch=offset=empty_batches=empty_examples=0;compute_seconds=wall_before=0.
    if (checkpoint_dir/'resume.pt').exists():
        saved=torch.load(checkpoint_dir/'resume.pt',map_location='cpu',weights_only=False,mmap=True)
        assert saved['config']==config and saved['run_id']==RUN_ID
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        step,epoch,offset=saved['step'],saved['next_epoch'],saved['next_offset']
        empty_batches,empty_examples=saved['empty_batches'],saved['empty_examples']
        compute_seconds,wall_before=saved['training_seconds'],saved['wall_seconds']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        rng.bit_generator.state=saved['numpy_rng'];augmentation_rng.bit_generator.state=saved['augmentation_rng'];del saved
    else:
        torch.manual_seed(2026090613)
    began=time.monotonic();last_checkpoint=began
    atomic(ROOT/'progress.json',dict(status='loading_shared_tiles',wandb_url=run.url,updated_at=time.time()))
    data=GpuData(doc);fn=torch.compile(model,dynamic=False)
    print('DATA_RESIDENT',data.bytes,'free_gpu',torch.cuda.mem_get_info()[0],flush=True)
    if step==0:
        # Decisive check of the only throughput change: compare the new shared
        # tile fetch and coupled labels with the old packed-parent batch code.
        sys.path.insert(0,str(ROOT.parent/'donor-resunet-scratch-20260912'))
        from packed_data import batch as old_batch
        entries=epoch_indices(doc,0)[:64]
        parents,states=data.crops(data.entry_row[entries],data.entry_yx[entries]-112,352)
        reference=dict(x=parents,y=(states==2).to(torch.uint8)*255,refs=np.arange(64),augmentation_rng=np.random.default_rng(2026090719))
        expected_x,expected_y=old_batch(reference,np.arange(64),np.random.default_rng(2026090613))
        check_offsets,check_params=parameters(np.random.default_rng(2026090613))
        raw,categorical=data.fetch(entries,check_offsets)
        actual_x,actual_state=augment(raw,categorical,check_params,np.random.default_rng(2026090719))
        assert torch.equal(expected_x,actual_x),'Shared tile loader changed augmented CT'
        assert torch.equal(expected_y, (actual_state==2).float()),'Shared categorical warp changed ink alignment'
        reference.update(y=(states>0).to(torch.uint8)*255,augmentation_rng=np.random.default_rng(2026090719))
        _,expected_mask=old_batch(reference,np.arange(64),np.random.default_rng(2026090613))
        assert torch.equal(expected_mask,(actual_state>0).float()),'Shared categorical warp changed supervision alignment'
        run.summary['verification/shared_tile_augmented_CT_exact']=True
        run.summary['verification/coupled_ink_mask_exact']=True
        print('SHARED_TILE_AUGMENTATION_EXACT',64,'examples',flush=True)
        del parents,states,reference,expected_x,expected_y,expected_mask,raw,categorical,actual_x,actual_state,_
    log=(ROOT/'training.jsonl').open('a',buffering=1)

    def checkpoint(final=False):
        payload=dict(run_id=RUN_ID,kind='resunet-deep',initialization='scratch',s1_samples=0,state_dict=model.state_dict(),
                     step=step,epoch=epoch,input_window=128,augmentation='scroll-shift-v2',amp_dtype='bfloat16',config=config,
                     training_seconds=compute_seconds,wall_seconds=wall_before+time.monotonic()-began)
        if not final:
            payload.update(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                           numpy_rng=copy.deepcopy(rng.bit_generator.state),augmentation_rng=copy.deepcopy(augmentation_rng.bit_generator.state),
                           next_epoch=epoch,next_offset=offset,empty_batches=empty_batches,empty_examples=empty_examples)
        path=checkpoint_dir/('last.pt' if final else 'resume.pt');tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

    @torch.inference_mode()
    def evaluate():
        from sklearn.metrics import average_precision_score,roc_auc_score
        fn.eval();loss_sum=0.;examples=0;probabilities=[];targets=[];states=[]
        for start in range(0,len(data.validation_coords()),16):
            raw,state=data.validation_batch(start);n=len(raw)
            x=raw.permute(0,3,1,2)[:,None].float().clamp_max_(200)/255
            if n<16:x=torch.cat([x,x[-1:].expand(16-n,*x.shape[1:])])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=fn(x)[:n];total,count=masked_terms(z,state[:,None])
            loss_sum+=float(total);examples+=int(count)
            probabilities.append(z[:,0,48:80,48:80].float().sigmoid().cpu().numpy())
            states.append(state[:,48:80,48:80].cpu().numpy())
        probability=np.concatenate(probabilities);state=np.concatenate(states);mask=state>0;y=state[mask]==2;p=probability[mask]
        both=bool(len(y) and y.any() and (~y).any())
        scores=dict(loss=loss_sum/examples if examples else None,ap=float(average_precision_score(y,p)) if both else None,
                    auc=float(roc_auc_score(y,p)) if both else None,patches=len(probability),contributing_loss_examples=examples,
                    supervised_ink=int(y.sum()),supervised_background=int((~y).sum()),unknown_pixels=int((~mask).sum()),
                    metric_core=32,loss_core=96,scope='Fixed841 holdout; not unseen-scroll generalization')
        return scores,probability,state

    try:
        while epoch<10:
            entries=epoch_indices(doc,epoch)
            while offset<len(entries):
                if STOP or (MAX_UPDATES and step>=MAX_UPDATES):
                    checkpoint();run.finish();print('STOPPED_AT_UPDATE',step,flush=True);return
                if step%25==0:check_lease()
                torch.cuda.synchronize();start=time.monotonic()
                offsets,params=parameters(rng)
                raw,state=data.fetch(entries[offset:offset+64],offsets)
                x,state=augment(raw,state,params,augmentation_rng);del raw
                contributing=(state[...,16:112,16:112].flatten(1)>0).any(1).sum()
                count=int(contributing);empty_examples+=64-count
                if not count:
                    empty_batches+=1;offset+=64
                    log.write(json.dumps(dict(epoch=epoch+1,offset=offset,empty_batch=True,step=step))+'\n')
                    del x,state
                    continue
                lr=2e-4*(.01+.99*min(1.,(step+1)/max(1,doc['total_updates']*.03)))*(.1+.9*(1+math.cos(math.pi*min(step,doc['total_updates'])/doc['total_updates']))/2)
                for group in optimizer.param_groups:group['lr']=lr
                fn.train();optimizer.zero_grad(set_to_none=True);total_loss=torch.zeros((),device='cuda')
                for micro in range(0,64,16):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        numerator,_=masked_terms(fn(x[micro:micro+16]),state[micro:micro+16]);loss=numerator/contributing
                    loss.backward();total_loss+=loss.detach()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if not bool(torch.isfinite(total_loss)):raise RuntimeError('Nonfinite masked loss')
                optimizer.step();torch.cuda.synchronize();duration=time.monotonic()-start
                step+=1;offset+=64;compute_seconds+=duration
                row=dict(step=step,epoch=epoch+1,offset=offset,loss=float(total_loss),gradient_norm=float(norm),lr=lr,
                         update_seconds=duration,training_seconds=compute_seconds,contributing_examples=count,
                         empty_batches=empty_batches,empty_examples=empty_examples,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(row)+'\n')
                if step<=3 or step%25==0:
                    atomic(ROOT/'progress.json',dict(status='training',**row,planned_updates=doc['total_updates'],updated_at=time.time(),wandb_url=run.url))
                    run.log({'train/'+k:v for k,v in row.items()});print('UPDATE',json.dumps(row),flush=True)
                if time.monotonic()-last_checkpoint>=300:
                    checkpoint();last_checkpoint=time.monotonic()
                del x,state,loss,numerator,total_loss,norm
            scores,probability,validation_state=evaluate()
            run.log({'validation/'+k:v for k,v in scores.items() if isinstance(v,(int,float))})
            log.write(json.dumps(dict(step=step,epoch=epoch+1,validation=scores))+'\n');print('VALIDATION',epoch+1,json.dumps(scores),flush=True)
            epoch+=1;offset=0;checkpoint();last_checkpoint=time.monotonic()
        scores,probability,validation_state=evaluate() if 'scores' not in locals() else (scores,probability,validation_state)
        checkpoint(final=True)
        np.savez_compressed(ROOT/'validation-final.npz',probability=probability,state=validation_state,coords=data.validation_coords())
        path=checkpoint_dir/'last.pt';digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
        result=dict(status='complete',run_id=RUN_ID,initialization='scratch',completed_epochs=epoch,global_step=step,
                    planned_updates=doc['total_updates'],empty_batches=empty_batches,empty_examples=empty_examples,
                    last_checkpoint=str(path),checkpoint_sha256=digest.hexdigest(),final_validation=scores,wandb_url=run.url,
                    training_seconds=compute_seconds,wall_seconds=wall_before+time.monotonic()-began,
                    config=str(ROOT/'config.json'),label_snapshot=doc['label_snapshot'])
        atomic(ROOT/'training-result.json',result);atomic(ROOT/'progress.json',dict(result,updated_at=time.time()))
        if wandb is not None:
            artifact=wandb.Artifact(RUN_ID+'-final',type='model',metadata=result);artifact.add_file(str(path))
            for name in ('config.json','data.json','label-snapshot.json','train.py','sparse_data.py','validation-final.npz'):
                artifact.add_file(str(ROOT/name))
            run.log_artifact(artifact).wait()
        run.summary.update({'training/status':'complete','training/result':result})
        print('TRAINING_COMPLETE',json.dumps(result),flush=True)
    except BaseException as error:
        atomic(ROOT/'progress.json',dict(status='failed',error=repr(error),step=step,epoch=epoch,offset=offset,updated_at=time.time()))
        raise
    finally:run.finish()


if __name__=='__main__':
    def stop(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    main()
