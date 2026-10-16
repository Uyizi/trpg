"""Three-kernel dynamic NVFP4 packer for the isolated SM120 experiment.

No CUDA operations run at import. No training/model integration. The reference
format and nearest-even encoding live in models.nvfp4_probe.
"""
import torch
import triton
import triton.language as tl
from models.nvfp4_probe import PackedNVFP4


@triton.jit
def _partial_amax(X, Partial, COUNT:tl.constexpr, BLOCK:tl.constexpr):
    ids=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+ids,ids<COUNT,0).to(tl.float32)
    tl.store(Partial+tl.program_id(0),tl.max(tl.abs(x),0))


@triton.jit
def _global_scale(Partial, Scale, COUNT:tl.constexpr, BLOCK:tl.constexpr):
    ids=tl.arange(0,BLOCK)
    value=tl.load(Partial+ids,ids<COUNT,0)
    scale=tl.maximum(tl.max(value,0)/(448.*6.),1.1754943508222875e-38)
    tl.store(Scale,scale)


@triton.jit
def _pack_blocks(X, Q, SF, GS, ROWS:tl.constexpr, K:tl.constexpr, COLS:tl.constexpr,
                 PAD_COLS:tl.constexpr, PAD_ROWS:tl.constexpr, BLOCKS:tl.constexpr):
    ids=tl.program_id(0)*BLOCKS+tl.arange(0,BLOCKS)
    rows=ids//PAD_COLS
    cols=ids%PAD_COLS
    valid=(rows<ROWS)&(cols<COLS)
    pairs=tl.arange(0,8)
    base=rows[:,None]*K+cols[:,None]*16+pairs[None,:]*2
    even=tl.load(X+base,valid[:,None],0).to(tl.float32)
    odd=tl.load(X+base+1,valid[:,None],0).to(tl.float32)
    maximum=tl.maximum(tl.max(tl.abs(even),1),tl.max(tl.abs(odd),1))
    gs=tl.load(GS)
    scale=tl.minimum(tl.maximum(tl.div_rn(maximum,(6.*gs)),0.001953125),448.).to(tl.float8e4nv)
    denom=scale.to(tl.float32)*gs
    ev=tl.div_rn(even,denom[:,None])
    od=tl.div_rn(odd,denom[:,None])
    ae=tl.abs(ev); ao=tl.abs(od)
    ce=((ae>.25).to(tl.int32)+(ae>=.75).to(tl.int32)+(ae>1.25).to(tl.int32)
        +(ae>=1.75).to(tl.int32)+(ae>2.5).to(tl.int32)+(ae>=3.5).to(tl.int32)+(ae>5.).to(tl.int32))
    co=((ao>.25).to(tl.int32)+(ao>=.75).to(tl.int32)+(ao>1.25).to(tl.int32)
        +(ao>=1.75).to(tl.int32)+(ao>2.5).to(tl.int32)+(ao>=3.5).to(tl.int32)+(ao>5.).to(tl.int32))
    ce=ce|((ev.to(tl.int32,bitcast=True)>>28)&8)
    co=co|((od.to(tl.int32,bitcast=True)>>28)&8)
    tl.store(Q+rows[:,None]*(K//2)+cols[:,None]*8+pairs[None,:],(ce|(co<<4)).to(tl.uint8),valid[:,None])
    swizzled=((rows//128)*(PAD_COLS//4)+cols//4)*512+(rows%32)*16+((rows%128)//32)*4+cols%4
    scale=tl.where(valid,scale.to(tl.float32),0.).to(tl.float8e4nv)
    tl.store(SF+swizzled,scale,rows<PAD_ROWS)


@torch.inference_mode()
def quantize_nvfp4_triton(x):
    """Dynamic global scale, E4M3 block16, E2M1 RNE; finite input is required."""
    if x.ndim!=2 or x.shape[1]%16 or min(x.shape)<=0:
        raise ValueError('NVFP4 expects a nonempty 2D matrix with K divisible by 16')
    if not x.is_cuda or not x.is_contiguous() or x.dtype not in (torch.float32,torch.bfloat16,torch.float16):
        raise ValueError('Triton NVFP4 requires a contiguous CUDA FP32/BF16/FP16 matrix')
    m,k=x.shape
    pc=((k//16+3)//4)*4; pr=((m+127)//128)*128
    partial=torch.empty((triton.cdiv(x.numel(),4096),),device=x.device,dtype=torch.float32)
    scale=torch.empty((),device=x.device,dtype=torch.float32)
    packed=torch.empty((m,k//2),device=x.device,dtype=torch.uint8)
    sf=torch.empty(pr*pc,device=x.device,dtype=torch.float8_e4m3fn)
    _partial_amax[(partial.numel(),)](x,partial,x.numel(),4096)
    _global_scale[(1,)](partial,scale,partial.numel(),triton.next_power_of_2(partial.numel()))
    _pack_blocks[(triton.cdiv(pr*pc,32),)](x,packed,sf,scale,m,k,k//16,pc,pr,32,enable_fp_fusion=False)
    return PackedNVFP4(packed,sf,scale,(m,k))
