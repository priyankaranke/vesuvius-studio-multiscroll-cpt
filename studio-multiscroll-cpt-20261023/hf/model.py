"""Deep normalized 3D U-Nets for native 64/96px, sixteen-plane inputs."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(32,channels),channels)


class ResidualBlock(nn.Module):
    def __init__(self,channels,dropout):
        super().__init__()
        self.branch=nn.Sequential(norm(channels),nn.SiLU(),
            nn.Conv3d(channels,channels,3,padding=1,bias=False),
            norm(channels),nn.SiLU(),nn.Dropout3d(dropout),
            nn.Conv3d(channels,channels,3,padding=1,bias=False))
    def forward(self,x):return x+self.branch(x)


class DeepResidualUNet(nn.Module):
    def __init__(self,deeper=False):
        super().__init__()
        widths=[48,96,192,384,640]
        counts=[3,4,6,8,10] if deeper else [2,3,4,5,6]
        drops=[.05,.10,.15,.20,.30] if deeper else [.02,.05,.10,.15,.20]
        decoder_counts=[4,4,3,3] if deeper else [3,3,2,2]
        self.stem=nn.Conv3d(1,widths[0],3,padding=1,bias=False)
        self.enc=nn.ModuleList([nn.Sequential(*[ResidualBlock(c,p) for _ in range(n)])
            for c,n,p in zip(widths,counts,drops)])
        self.down=nn.ModuleList([nn.Sequential(norm(widths[i]),nn.SiLU(),
            nn.Conv3d(widths[i],widths[i+1],3,stride=(1 if i==0 else 2,2,2),padding=1,bias=False)) for i in range(4)])
        self.up=nn.ModuleList([nn.Conv3d(widths[i+1],widths[i],1,bias=False) for i in range(3,-1,-1)])
        self.fuse=nn.ModuleList([nn.Sequential(norm(widths[i]*2),nn.SiLU(),
            nn.Conv3d(widths[i]*2,widths[i],1,bias=False)) for i in range(3,-1,-1)])
        self.dec=nn.ModuleList([nn.Sequential(*[ResidualBlock(widths[i],drops[i]) for _ in range(n)])
            for i,n in zip(range(3,-1,-1),decoder_counts)])
        self.head=nn.Sequential(norm(widths[0]),nn.SiLU(),nn.Conv3d(widths[0],1,1))
        self.depth_logits=nn.Parameter(torch.zeros(16))
        self.architecture_details=dict(family='Deep residual 3D U-Net',widths=widths,
            encoder_blocks=counts,decoder_blocks=decoder_counts,normalization='GroupNorm before every residual convolution and resampling/fusion stage',
            dropout3d=drops,depth_pool='Learned global softmax across the unchanged sixteen input planes',
            initialization='scratch',spatial_resampling='Four down/up stages; first downsample preserves depth')
    def forward(self,x):
        x=self.stem(x);skips=[]
        for i,stage in enumerate(self.enc):
            x=stage(x)
            if i<4:skips.append(x);x=self.down[i](x)
        for up,fuse,stage,skip in zip(self.up,self.fuse,self.dec,reversed(skips)):
            x=up(F.interpolate(x,size=skip.shape[-3:],mode='trilinear',align_corners=False))
            x=stage(fuse(torch.cat([x,skip],dim=1)))
        return (self.head(x)*self.depth_logits.softmax(0)[None,None,:,None,None]).sum(2)


class RegularizedMedNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        from mednext import MedNeXt
        self.net=MedNeXt(in_channels=1,n_channels=80,n_classes=1,
            exp_r=[3,4,8,8,8,8,8,4,3],kernel_size=3,
            block_counts=[3,4,8,8,8,8,8,4,3],do_res=True,do_res_up_down=True,
            norm_type='group',dim='3d',deep_supervision=False)
        # Keep upstream blocks intact; add channel-wise spatial dropout to
        # their residual branches by wrapping each compression convolution.
        for name,module in list(self.net.named_modules()):
            if hasattr(module,'conv3'):
                p=.20 if name.startswith('bottleneck') else .15 if '_3' in name else .10
                module.conv3=nn.Sequential(nn.Dropout3d(p),module.conv3)
        self.depth_logits=nn.Parameter(torch.zeros(16))
        self.architecture_details=dict(family='MedNeXt Large-80 3D U-Net',width=80,
            blocks=[3,4,8,8,8,8,8,4,3],normalization='Upstream channel-wise GroupNorm in every inverted bottleneck',
            dropout3d=[.10,.15,.20],depth_pool='Learned global softmax across sixteen planes',initialization='scratch')
    def forward(self,x):
        return (self.net(x)*self.depth_logits.softmax(0)[None,None,:,None,None]).sum(2)
