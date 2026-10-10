"""Conservative scan/surface shift hypotheses, added after legacy augmentation.

No target statistics or predictions tune these ranges. Geometry samples the
measured parent context and shares its XY mapping with nearest-neighbor labels.
Depth perturbations taper to zero at the measured stack endpoints.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F

PRESET=dict(
    name='scroll-shift-v1',spatial_probability=.5,rotation_degrees=8,
    scale=[.9,1.1],shear=.05,smooth_xy_displacement_pixels=.8,
    depth_probability=.5,depth_shift_planes=1.,depth_tilt_planes=.4,depth_scale=[.9,1.1],
    gamma_probability=.35,gamma=[.85,1.18],bias_field_probability=.3,bias_field_log_sigma=.05,
    noise_probability=.35,noise_sigma_gray8=[0.,3.],xy_blur_probability=.25,xy_blur_sigma=[.4,.9],
    low_resolution_probability=.2,low_resolution_scale=[.8,1.],depth_blur_probability=.2,
    measured_context=128,minimum_context_halo=16,legacy_augmentations='Retained: gain/bias, D4, random measured crop and depth dropout',
    interpretation='Physical distribution-shift hypotheses; no target-fitted normalization and no guarantee of transfer')

PRESET_V2=dict(PRESET,name='scroll-shift-v2',measured_context=352,minimum_context_halo=48,additional=dict(
    surface_curvature_probability=.35,surface_curvature_planes=.6,surface_control_grid=3,
    local_contrast_probability=.35,local_contrast_gain=[.8,1.2],local_background_kernel=9,
    depth_response_probability=.25,depth_gain_bound=.06,depth_bias_gray8_bound=3.,depth_response_knots=4,
    anisotropic_blur_probability=.25,anisotropic_sigma_major=[.6,1.1],anisotropic_sigma_minor=[.3,.6],
    anisotropic_blur_mix=[.25,.65],anisotropic_orientation_degrees=[0,180]),
    interpretation='V1 retained, plus bounded surface curvature and CT response variations. Synthetic domain randomization; not a measured S1-to-1447 mapping.')

DEPTH_PRESET_NAME='scroll-shift-v1-zjitter-zflip'
DEPTH_PRESET=dict(PRESET,name=DEPTH_PRESET_NAME,depth_window=dict(
    measured_planes=20,input_planes=16,jitter_offsets=[-2,-1,0,1,2],sampling='Uniform per patch',
    z_reverse_probability=.5,label_and_supervision_mask='Unchanged by depth-window selection and reversal',
    rng_seed=2026090821),
    interpretation='Exact V1 transforms plus measured 16-of-20 depth-window jitter and independent per-patch 50% Z reversal.')


DEPTH_JITTER_NAME='scroll-shift-v1-zjitter'
DEPTH_PRESET_NAMES=(DEPTH_PRESET_NAME,DEPTH_JITTER_NAME)
DEPTH_JITTER_PRESET=dict(DEPTH_PRESET,name=DEPTH_JITTER_NAME,
    depth_window=dict(DEPTH_PRESET['depth_window'],z_reverse_probability=0.),
    interpretation='Exact V1 transforms plus measured 16-of-20 depth-window jitter; Z reversal disabled.')


def preset(name):
    if name=='scroll-shift-v1':return PRESET
    if name=='scroll-shift-v2':return PRESET_V2
    if name==DEPTH_PRESET_NAME:return DEPTH_PRESET
    if name==DEPTH_JITTER_NAME:return DEPTH_JITTER_PRESET
    raise ValueError('Unknown scroll augmentation: '+name)


def warp(raw,label,window,theta,displacement,depth_parameters,depth_field=None):
    """Explicit shared coordinates make label alignment and support testable."""
    n,_,depth,h,w=raw.shape;device=raw.device
    xy=torch.arange(window,device=device,dtype=torch.float32)-(window-1)/2
    yy,xx=torch.meshgrid(xy,xy,indexing='ij')
    source=torch.stack([xx,yy],dim=-1).expand(n,-1,-1,-1)
    source=torch.einsum('nij,nhwj->nhwi',theta,source)+displacement.permute(0,2,3,1)
    # Both cached context and output have even size: native pixel centers.
    gridxy=torch.stack([(source[...,0]+(w-1)/2+.5)*2/w-1,
                        (source[...,1]+(h-1)/2+.5)*2/h-1],dim=-1)
    z=torch.arange(depth,device=device,dtype=torch.float32)
    fade=torch.sin(math.pi*z/(depth-1))
    offset=(depth_parameters[:,0,None,None]+depth_parameters[:,1,None,None]*(xx/(window/2))+
            depth_parameters[:,2,None,None]*(yy/(window/2)))
    if depth_field is not None:offset=offset+depth_field[:,0]
    zz=z[None,:,None,None]+fade[None,:,None,None]*(offset[:,None]+
        depth_parameters[:,3,None,None,None]*(z[None,:,None,None]-(depth-1)/2))
    zz=zz.expand(n,depth,window,window)
    grid=torch.cat([gridxy[:,None].expand(-1,depth,-1,-1,-1),((zz+.5)*2/depth-1)[...,None]],dim=-1)
    # Callers supply a window-dependent measured halo for the XY warp.
    # Endpoint-anchored depth maps stay within the existing sixteen planes.
    x=F.grid_sample(raw,grid,mode='bilinear',padding_mode='border',align_corners=False)
    y=F.grid_sample(label,gridxy,mode='nearest',padding_mode='border',align_corners=False)
    return x,y


def augment(raw,label,window,rng,preset_name='scroll-shift-v1'):
    spec=preset(preset_name)
    n=raw.shape[0];device=raw.device
    def tensor(a):return torch.as_tensor(a,device=device,dtype=torch.float32)
    def selected(p):return tensor(rng.random(n)<p)[:,None,None,None,None]
    spatial=rng.random(n)<.5
    angle=np.deg2rad(rng.uniform(-8,8,n))*spatial
    scales=1+(rng.uniform(.9,1.1,(n,2))-1)*spatial[:,None]
    shear=rng.uniform(-.05,.05,n)*spatial
    theta=np.zeros((n,2,2),np.float32)
    theta[:,0,0]=np.cos(angle)*scales[:,0];theta[:,0,1]=-np.sin(angle)*scales[:,1]+shear
    theta[:,1,0]=np.sin(angle)*scales[:,0];theta[:,1,1]=np.cos(angle)*scales[:,1]
    displacement=F.interpolate(tensor(rng.uniform(-.8,.8,(n,2,3,3))*spatial[:,None,None,None]),
        size=(window,window),mode='bilinear',align_corners=False)
    depth=rng.random(n)<.5
    pars=np.stack([rng.uniform(-1,1,n),rng.uniform(-.4,.4,n),rng.uniform(-.4,.4,n),rng.uniform(-.1,.1,n)],axis=1)*depth[:,None]
    depth_field=None
    if preset_name=='scroll-shift-v2':
        extra=spec['additional'];chosen=rng.random(n)<extra['surface_curvature_probability']
        depth_field=F.interpolate(tensor(rng.uniform(-extra['surface_curvature_planes'],extra['surface_curvature_planes'],
            (n,1,3,3))*chosen[:,None,None,None]),size=(window,window),mode='bilinear',align_corners=False)
    # Curvature shares the original resampling pass: no second depth blur.
    x,y=warp(raw,label,window,tensor(theta),displacement,tensor(pars),depth_field)
    gamma=1+(tensor(rng.uniform(.85,1.18,n))[:,None,None,None,None]-1)*selected(.35)
    x=(x.clamp_min(0)/255).pow(gamma)*255
    field=F.interpolate(tensor(rng.normal(0,.05,(n,1,3,3))),size=(window,window),mode='bilinear',align_corners=False)[:,None]
    x=x*torch.exp(field*selected(.3))
    sigma=float(rng.uniform(.4,.9));a=torch.arange(-2,3,device=device,dtype=torch.float32)
    kernel=torch.exp(-a.square()/(2*sigma*sigma));kernel=kernel/kernel.sum()
    flat=x.reshape(-1,1,window,window)
    blurred=F.conv2d(F.pad(flat,(2,2,0,0),mode='replicate'),kernel.view(1,1,1,5))
    blurred=F.conv2d(F.pad(blurred,(0,0,2,2),mode='replicate'),kernel.view(1,1,5,1)).reshape_as(x)
    x=x+(blurred-x)*selected(.25)
    low=max(2,round(window*float(rng.uniform(.8,1.))))
    reduced=F.interpolate(x[:,0],size=(low,low),mode='bilinear',align_corners=False,antialias=True)
    reduced=F.interpolate(reduced,size=(window,window),mode='bilinear',align_corners=False)[:,None]
    x=x+(reduced-x)*selected(.2)
    smooth=F.avg_pool3d(x,(3,1,1),stride=1,padding=(1,0,0),count_include_pad=False)
    x=x+(smooth-x)*selected(.2)*float(rng.uniform(.1,.3))
    generator=torch.Generator(device=device).manual_seed(int(rng.integers(0,2**63-1)))
    noise=torch.randn(x.shape,device=device,generator=generator)
    correlated=torch.randn((n,1,1,window,window),device=device,generator=generator)
    amplitude=tensor(rng.uniform(0,3,n))[:,None,None,None,None]*selected(.35)
    x=x+(noise+correlated)*(.5**.5)*amplitude
    if preset_name=='scroll-shift-v2':x=appearance_v2(x,rng)
    return x.clamp_min_(0),y


def appearance_v2(x,rng):
    """CT-only variations; positive bounded gains and no added texture templates."""
    n,_,depth,h,w=x.shape;device=x.device;spec=PRESET_V2['additional']
    def tensor(a):return torch.as_tensor(a,device=device,dtype=torch.float32)
    def selected(p):return tensor(rng.random(n)<p)[:,None,None,None,None]
    # Vary local fine-structure contrast without scaling the background by it.
    k=spec['local_background_kernel'];pad=k//2;flat=x.reshape(-1,1,h,w)
    background=F.avg_pool2d(F.pad(flat,(pad,pad,0,0),mode='replicate'),(1,k),stride=1)
    background=F.avg_pool2d(F.pad(background,(0,0,pad,pad),mode='replicate'),(k,1),stride=1).reshape_as(x)
    gain=F.interpolate(tensor(rng.uniform(*spec['local_contrast_gain'],(n,1,3,3))),
        size=(h,w),mode='bilinear',align_corners=False)[:,None]
    x=x+(x-background)*(gain-1)*selected(spec['local_contrast_probability'])
    # Smooth response through depth; zero mean and explicit bounds avoid a
    # second uncontrolled global gain/bias on top of the retained legacy one.
    def depth_curve(bound):
        curve=F.interpolate(tensor(rng.uniform(-bound,bound,(n,1,spec['depth_response_knots']))),
            size=depth,mode='linear',align_corners=False)
        curve=curve-curve.mean(-1,keepdim=True)
        return (curve/(curve.abs().amax(-1,keepdim=True)/bound).clamp_min(1))[:,:,:,None,None]
    gain_z=depth_curve(spec['depth_gain_bound']);bias_z=depth_curve(spec['depth_bias_gray8_bound'])
    x=x+(x*gain_z+bias_z)*selected(spec['depth_response_probability'])
    # Same rotated anisotropic point-spread kernel in all sixteen planes.
    major=float(rng.uniform(*spec['anisotropic_sigma_major']));minor=float(rng.uniform(*spec['anisotropic_sigma_minor']))
    angle=float(rng.uniform(0,math.pi));axis=torch.arange(-3,4,device=device,dtype=torch.float32)
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    u=xx*math.cos(angle)+yy*math.sin(angle);v=-xx*math.sin(angle)+yy*math.cos(angle)
    kernel=torch.exp(-.5*((u/major).square()+(v/minor).square()));kernel=kernel/kernel.sum()
    blurred=F.conv2d(F.pad(x.reshape(-1,1,h,w),(3,3,3,3),mode='replicate'),kernel[None,None]).reshape_as(x)
    mix=tensor(rng.uniform(*spec['anisotropic_blur_mix'],n))[:,None,None,None,None]
    return x+(blurred-x)*mix*selected(spec['anisotropic_blur_probability'])
