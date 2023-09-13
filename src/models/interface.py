from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attention import FlashAttention

import slowfast.utils.logging as logging
from slowfast.models.utils import get_3d_sincos_pos_embed

from .resnet import get_resnet
from .svt import get_svt, VisionTransformer as SVT
from .video_mae import get_sinusoid_encoding_table, pretrain_videomae_star_patch16_224, vit_star_patch16_224

logger = logging.get_logger(__name__)


def get_initialization_model(model):
    vmae_pt = partial(pretrain_videomae_star_patch16_224, star='base', decoder_depth=4)
    vmae_large_pt = partial(pretrain_videomae_star_patch16_224, star='large', decoder_depth=12)
    vmae_ft_k400 = partial(vit_star_patch16_224, star='base', num_classes=400)
    models_dict = dict(
        svt_b_pt_k400_20=get_svt,
        pbyol=get_resnet,
        vmae_b_pt_k400_1600=partial(vmae_pt, ckpt='initialization/vmae_base1600_pt_kinetics.pth'),
        vmae_b_ft_k400_1600=partial(vmae_ft_k400, ckpt='initialization/vmae_base1600_ft_kinetics.pth'),
        vmae_l_pt_k400_1600=partial(vmae_large_pt, ckpt='initialization/vmae_large1600_pt_kinetics.pth'),
    )
    if model is None:
        return models_dict
    convert_dict = dict(
        svt_b_pt_k400_20=convert_svt,
        pbyol=convert_resnet,
        vmae_b_pt_k400_1600=convert_vmae,
        vmae_b_ft_k400_1600=partial(convert_vmae, finetuned=True),
        vmae_l_pt_k400_1600=convert_vmae,
    )
    return convert_dict[model], models_dict[model]


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, num_frames=16, patch_size=16, temporal_patch_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) * (num_frames // temporal_patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(temporal_patch_size, patch_size, patch_size),
                              stride=(temporal_patch_size, patch_size, patch_size))

    def forward(self, x):
        # B, C, T, H, W = x.shape
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2), x.size()[-3:]  # B C T H W -> B THW C, THW


class VisionTransformer(nn.Module):
    def __init__(
            self, *, use_cls_token=True, use_learnable_pos_emb=True, factorized_pos_emb=True, pos_drop_rate=0.0,
            img_size=224, num_frames=16, patch_size=16, temporal_patch_size=2, in_chans=3, embed_dim=768, blocks=None,
            norm=None, norm_after_average=False, pos_3d=False, divided_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, num_frames=num_frames, patch_size=patch_size,
                                      temporal_patch_size=temporal_patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        self.factorized_pos_emb = factorized_pos_emb
        self.patch_dims = (num_frames // temporal_patch_size, img_size // patch_size, img_size // patch_size)
        if factorized_pos_emb:
            assert use_learnable_pos_emb
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_dims[0], embed_dim))
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim), requires_grad=not use_learnable_pos_emb)
            with torch.no_grad():
                if pos_3d:
                    pos_embed = get_3d_sincos_pos_embed(embed_dim, self.patch_dims[1], self.patch_dims[0])
                    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
                else:
                    pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
                self.pos_embed.data.copy_(pos_embed)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.divided_attention = divided_attention
        if blocks is None:
            self.blocks = nn.ModuleList([])
        else:
            self.blocks = blocks
        self.norm_after_average = norm_after_average
        if norm is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm

    def get_pos_embedding(self, t, h, w):
        if self.factorized_pos_emb:
            pos_embed = self.pos_embed_spatial.repeat(1, self.patch_dims[0], 1) + \
                        torch.repeat_interleave(self.pos_embed_temporal, self.patch_dims[1] * self.patch_dims[2], dim=1)
        else:
            pos_embed = self.pos_embed
        if self.patch_dims != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed.reshape(1, *self.patch_dims, -1).permute(0, 4, 1, 2, 3), size=(t, h, w), mode="trilinear")
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)
        return pos_embed

    def drop_tokens(self, x, num_tokens=-1, t=None):
        if num_tokens == -1:
            return x
        N, L, D = x.size()  # batch, length, dim
        if self.cls_token is not None:
            num_tokens -= 1
        if self.divided_attention:
            assert t is not None
            num_tokens_per_frame = num_tokens // t
            x = x.view(N, -1, t, D)  # batch, spatial, time, dim
            noise = torch.rand(N, x.size(1), device=x.device)  # noise in [0, 1]
            ids_keep = torch.argsort(noise, dim=1)[:, :num_tokens_per_frame]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, t, D))  # bntd
            x_masked = x_masked.view(N, -1, D)
        else:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_keep = torch.argsort(noise, dim=1)[:, :num_tokens]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def prepare_tokens(self, x, num_tokens=-1):
        x, thw = self.patch_embed(x)
        x = x + self.get_pos_embedding(*thw).type_as(x)
        if self.divided_attention:  # B(TS)C -> B(ST)C
            batch_size = x.size(0)
            dim = x.size(-1)
            x = x.reshape(batch_size, thw[0], -1, dim).permute(0, 2, 1, 3).reshape(batch_size, -1, dim)
        return self.drop_tokens(x, num_tokens, thw[0]), thw

    def forward_head(self, x, num_tokens, thw=None):
        if isinstance(x, dict):
            assert num_tokens != -1
            x = torch.cat(
                [self.prepare_tokens(torch.cat(v), num_tokens)[0] for k, v in x.items()
                 if k not in {'index', 'label', 'pos_info'}])  # N1xNSAxB+N2xNSAxB+..., M, C
        else:
            x, thw = self.prepare_tokens(x, num_tokens)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = self.pos_drop(x)
        if self.divided_attention:
            extra_args = [thw[0]]
        else:
            extra_args = []
        return x, extra_args

    def forward_tail(self, x):
        if not self.norm_after_average:
            x = self.norm(x)
        cls = x.mean(1) if self.cls_token is None else x[:, 0]
        if self.norm_after_average:
            cls = self.norm(cls)
        return cls

    def drop_half(self, x, extra_args=None):
        if self.divided_attention:
            if self.cls_token is not None:
                cls, rest = x[:, :1], x[:, 1:]
            else:
                rest = x
            b, st, d = rest.size()
            rest = rest.reshape(b, -1, extra_args[0], d)[:, :(st // extra_args[0]) // 2].reshape(b, -1, d)
            if self.cls_token is not None:
                x = torch.cat([cls, rest], dim=1)
            else:
                x = rest
        else:
            x = x[:, :x.size(1) // 2, :]  # B, N//2, C
        return x

    def simplified_forward(self, x, num_tokens=-1, token_drop=False, frozen_backbone=-1, given_t=None):
        if isinstance(x, dict) and self.divided_attention and given_t is None:
            current_t = None
            current_dict = {}
            result = []
            for k, v in x.items():
                if k in {'index', 'label', 'pos_info'}:
                    continue
                a, b = k.split('_')
                t = int(a)
                if current_t is None:
                    current_t = t
                    current_dict[k] = v
                elif t == current_t:
                    current_dict[k] = v
                else:
                    result.append(self.simplified_forward(current_dict, num_tokens,
                                                          token_drop, frozen_backbone, [current_t]))
                    current_dict = {k: v}
                    current_t = t
            result.append(self.simplified_forward(current_dict, num_tokens, token_drop, frozen_backbone, [current_t]))
            return torch.cat(result, dim=0)
        else:
            if frozen_backbone == -1:
                x, extra_args = self.forward_head(x, num_tokens, given_t)
            else:
                with torch.no_grad():
                    x, extra_args = self.forward_head(x, num_tokens, given_t)
            for i, b in enumerate(self.blocks):
                if i < frozen_backbone:
                    with torch.no_grad():
                        if i == 6 and token_drop:
                            x = self.drop_half(x, extra_args=None)
                        x = b(x, *extra_args)
                else:
                    if i == 6 and token_drop:
                        x = self.drop_half(x, extra_args=None)
                    x = b(x, *extra_args)
            if frozen_backbone == 13:
                with torch.no_grad():
                    cls = self.forward_tail(x)
            else:
                cls = self.forward_tail(x)
            return cls

    def forward(self, x, num_tokens=-1, token_drop=False, frozen_backbone=-1):
        return self.simplified_forward(x, num_tokens, token_drop, frozen_backbone)


class FlashAttentionWrapper(nn.Module):
    def __init__(self, attention_module, active, attention_dropout=0.0):
        super().__init__()
        self.flash_attention = FlashAttention(attention_dropout=attention_dropout)
        self.attention_module = attention_module
        self.active = active

    def flash_forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if self.active:
            return self.flash_forward(*args, **kwargs)
        return self.attention_module(*args, **kwargs)


class OmniFlashAttention(FlashAttentionWrapper):
    def __init__(self, omni_attention, active):
        super().__init__(omni_attention, active, omni_attention.attn_drop.p)

    def flash_forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.attention_module.qkv(x).reshape(B, N, 3, self.attention_module.num_heads,
                                                   C // self.attention_module.num_heads)
        x = self.flash_attention(qkv)[0].view(B, N, -1)  # BN3Hd => BNC
        return self.attention_module.proj_drop(self.attention_module.proj(x))


class VMAEFlashAttention(FlashAttentionWrapper):
    def __init__(self, vmae_attention, active):
        super().__init__(vmae_attention, active, vmae_attention.attn_drop.p)

    def flash_forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.attention_module.q_bias is not None:
            qkv_bias = torch.cat((self.attention_module.q_bias, torch.zeros_like(
                self.attention_module.v_bias, requires_grad=False), self.attention_module.v_bias))
        qkv = F.linear(input=x, weight=self.attention_module.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.attention_module.num_heads, -1)
        x = self.flash_attention(qkv)[0].view(B, N, -1)
        return self.attention_module.proj_drop(self.attention_module.proj(x))


def convert_resnet(resnet, finetuned=False, convert_to_flash=False, **kwargs):
    resnet.embed_dim = 2048
    return resnet, None


@torch.no_grad()
def convert_svt(svt: SVT, finetuned=False, convert_to_flash=False, **kwargs):
    assert not finetuned
    for b in svt.blocks:
        b.attn = OmniFlashAttention(b.attn, convert_to_flash)
        b.temporal_attn = OmniFlashAttention(b.temporal_attn, convert_to_flash)
    base = VisionTransformer(use_cls_token=True, use_learnable_pos_emb=True, factorized_pos_emb=True, pos_drop_rate=0.0,
                             temporal_patch_size=1, blocks=svt.blocks, norm=svt.norm, norm_after_average=False,
                             pos_3d=False, divided_attention=True, num_frames=8, **kwargs)
    base.patch_embed.proj.weight.copy_(svt.patch_embed.proj.weight.unsqueeze(2))
    base.patch_embed.proj.bias.copy_(svt.patch_embed.proj.bias)
    base.cls_token.copy_(svt.cls_token + svt.pos_embed[:, :1])
    base.pos_embed_spatial.copy_(svt.pos_embed[:, 1:])
    base.pos_embed_temporal.copy_(svt.time_embed)
    return base, svt.head


@torch.no_grad()
def convert_vmae(vmae, finetuned=False, convert_to_flash=False, **kwargs):
    if finetuned:
        head = vmae.head
        params = dict(pos_drop_rate=vmae.pos_drop.p, norm=vmae.fc_norm, norm_after_average=True)
    else:
        head = None
        vmae = vmae.encoder
        params = dict(pos_drop_rate=0.0, norm=vmae.norm, norm_after_average=False)
    for b in vmae.blocks:
        b.attn = VMAEFlashAttention(b.attn, convert_to_flash)
    base = VisionTransformer(
        use_cls_token=False, use_learnable_pos_emb=type(vmae.pos_embed) is nn.Parameter, factorized_pos_emb=False,
        embed_dim=vmae.embed_dim, blocks=vmae.blocks, pos_3d=False, **params, **kwargs)
    base.patch_embed.proj = vmae.patch_embed.proj
    base.pos_embed.copy_(vmae.pos_embed)
    return base, head
