# Original code from:                     
# https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/attention.py#L196

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction

from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange


############################################ UTILS
def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

############################################ MODELS 
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)



class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
    
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum(q, k, 'b i d, b j d -> b i j') * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum(attn, v, 'b i j, b j d -> b i d')
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class EncryptionModule(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels=3, n_heads=8, d_head=64, depth=1, dropout=0., 
                 context_dim=None, img_size=128, patch_size=8, template_init='rand', strength = 'fixed'):
        super().__init__()

        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        patch_dim = in_channels * patch_size ** 2

        init_dict = {
            'rand': torch.rand,
            'zero': torch.zeros,
            'ones': torch.ones,
            'uniform': lambda shape: torch.empty(shape).uniform_()
        }

        if template_init not in init_dict:
            raise ValueError('template_init must be one of: rand, zero, ones, uniform')
        self.template = nn.Parameter(init_dict[template_init]((1, 256,512)))
        
        if strength == 'fixed':
            self.signal_strength = 0.01
        elif strength == 'learned':
            self.signal_strength = nn.Parameter(torch.tensor(0.01))

        self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, inner_dim))

        self.proj_in = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, inner_dim),
            nn.LayerNorm(inner_dim),
        )

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
            for d in range(depth)])
        
        self.to_out = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                           h = img_size // patch_size, 
                           w = img_size // patch_size, 
                           p1 = patch_size, 
                           p2 = patch_size),
        )


    def forward(self, x):
        b, _, _, _ = x.shape
        
        x = self.proj_in(x)
        template = self.template.repeat(b, 1, 1)

        out = self.transformer_blocks[0](template, context=x)
        
        for block in self.transformer_blocks[1:]:
            out = block(out, context=x)

        return self.to_out(out) * self.signal_strength

    
    def get_template(self):
        return self.template
        
    def set_strength(self, strength):
        self.signal_strength = strength

    
class SignalRecovery(nn.Module):
    def __init__(self, in_channels=3, n_heads=8, d_head=64, depth=1, dropout=0.1, num_features=64 ,img_size=128, patch_size=8):
        super(SignalRecovery, self).__init__()
            
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        patch_dim = in_channels * patch_size ** 2

        self.proj_in = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, inner_dim),
            nn.LayerNorm(inner_dim),
        )

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout)
            for d in range(depth)])
    
        self.to_out = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h = img_size // patch_size, 
                        w = img_size // patch_size, 
                        p1 = patch_size, 
                        p2 = patch_size)
        )        

       
    def forward(self, imgs, template = None):
        x = imgs
        b, _, _, _ = x.shape
        x = self.proj_in(x)

        context = x if template is None else template.repeat(b, 1, 1)

        for block in self.transformer_blocks:
            x = block(x, context=context)

        signal = self.to_out(x)

        return signal
    