import math
from enum import Enum
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer import Block


class ArgParseEnum(str, Enum):
    def __str__(self):
        return self.value


def get_head(input_dim, output_dim, is_prediction=False):
    if is_prediction:
        num_layers, hidden_dim = 2, 4096
    else:
        num_layers, hidden_dim = 3, 2048
    mlp = []
    for layer in range(num_layers):
        dim1 = input_dim if layer == 0 else hidden_dim
        dim2 = output_dim if layer == num_layers - 1 else hidden_dim
        mlp.append(nn.Linear(dim1, dim2, bias=False))
        if layer < num_layers - 1:
            mlp.append(nn.GELU())
    return nn.Sequential(*mlp)


def s_int(x):
    a = math.floor(x)
    return a + ((x - a) > random())


class LinearOutputMode(ArgParseEnum):
    logits = 'logits'
    avg_logits = 'avg_logits'
    avg_probs = 'avg_probs'


class MLP(nn.Module):
    def __init__(self, input_embed_dim, hidden_size, output_dim, use_h_avg_pool,
                 batch_norm, num_layers, num_classes=400):
        super().__init__()
        self.input_embed_dim = input_embed_dim
        self.use_h_avg_pool = use_h_avg_pool
        mlp = [nn.Linear(input_embed_dim, hidden_size), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            mlp.append(nn.Linear(hidden_size, hidden_size))
            mlp.append(nn.ReLU(inplace=True))
        mlp.append(nn.Linear(hidden_size, output_dim))
        self.mlp = nn.Sequential(*mlp)
        input_dim = input_embed_dim + output_dim
        if use_h_avg_pool:
            input_dim += hidden_size
        self.linear = nn.Conv1d(input_dim, num_classes, 1)  # BCL
        if batch_norm:
            self.linear = nn.Sequential(nn.BatchNorm1d(input_dim, affine=False), self.linear)

    def forward(self, o, output_mode=LinearOutputMode.logits, no_logits=False):
        h = self.mlp(o['x'])
        inputs = [o['x'], h]
        if self.use_h_avg_pool:
            inputs.append(h.mean(1, keepdim=True).expand(-1, h.size(1), -1))
        if no_logits:
            return torch.cat(inputs, dim=-1)
        logits = self.linear(torch.cat(inputs, dim=-1).permute(0, 2, 1))  # b, d, v => b, k, v
        if output_mode is LinearOutputMode.logits:
            return logits
        elif output_mode is LinearOutputMode.avg_logits:
            return logits.mean(-1)
        return logits.softmax(1).mean(-1)


class LinearOnTop(nn.Module):
    def __init__(self, input_embed_dim, hidden_size=0, append_h=False,
                 use_h_avg_pool=False, batch_norm=False, num_classes=400):
        super().__init__()
        self.input_embed_dim = input_embed_dim
        self.hidden_size = hidden_size
        self.append_h = append_h
        self.use_h_avg_pool = use_h_avg_pool
        input_dim = input_embed_dim
        if append_h:
            input_dim += hidden_size
        if use_h_avg_pool:
            input_dim += hidden_size
        if self.add_set_cls:
            input_dim += hidden_size
        if self.add_coc_cls:
            input_dim += hidden_size
        self.linear = nn.Conv1d(input_dim, num_classes, 1)  # BCL
        if batch_norm:
            self.linear = nn.Sequential(nn.BatchNorm1d(input_dim, affine=False), self.linear)

    def forward(self, o, output_mode=LinearOutputMode.logits, no_logits=False):
        inputs = []
        if self.input_embed_dim != 0:
            inputs.append(o['x'])
        if self.hidden_size != 0:
            h = o['h']
            v = h.size(1)
            if self.append_h:
                inputs.append(h)
            if self.use_h_avg_pool:
                inputs.append(h.mean(1, keepdim=True).expand(-1, v, -1))
            if o['set_cls'].ndim == 2:
                set_cls = o['set_cls'].unsqueeze(1).expand(-1, v, -1)
            else:
                set_cls = o['set_cls']
            inputs.append(set_cls)
        if no_logits:
            return torch.cat(inputs, dim=-1)
        logits = self.linear(torch.cat(inputs, dim=-1).permute(0, 2, 1))
        if output_mode is LinearOutputMode.logits:
            return logits
        elif output_mode is LinearOutputMode.avg_logits:
            return logits.mean(-1)
        return logits.softmax(1).mean(-1)


class Scale(nn.Module):
    def __init__(self, input_embed_dim, hidden_size=512, num_layers=2, set_dim=256, set_temperature=0.2,
                 mask_ratio=0.25, dense_dim=256, dense_temperature=0.2, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        assert 1.0 > self.mask_ratio >= 0.0
        self.mask_ratio = mask_ratio
        self.dense_dim = dense_dim
        self.input_projection = nn.Linear(input_embed_dim, hidden_size)
        self.pos_encoding_mlp = nn.Sequential(nn.Linear(7, 2 * hidden_size), nn.ReLU(True),
                                              nn.Linear(2 * hidden_size, hidden_size))
        self.set_dim = set_dim
        self.set_cls_token = None
        self.set_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.set_head = get_head(hidden_size, set_dim, True)
        self.set_temperature = set_temperature
        self.transformer = nn.ModuleList(
            [Block(hidden_size, num_heads=hidden_size // 32, qkv_bias=True, mlp_ratio=mlp_ratio,
                   drop=drop, attn_drop=attn_drop) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.dense_temperature = dense_temperature
        self.msk_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dense_target_head = get_head(input_embed_dim, dense_dim)  # gives the target
        self.dense_pred_head = get_head(hidden_size, dense_dim, True)

    @staticmethod
    def contrastive_loss(prediction, target, temp, dual=True):
        prediction = prediction.reshape(-1, prediction.size(-1))
        target = target.reshape(-1, target.size(-1))
        logits = torch.einsum('nd,md->nm', prediction, target) / temp
        con_loss = -torch.diag(torch.log_softmax(logits, dim=1))
        if dual:
            con_loss = (con_loss - torch.diag(torch.log_softmax(logits, dim=0))) * 0.5
        with torch.no_grad():
            batch_size = logits.size(0)
            arange = torch.arange(batch_size, device=logits.device)
            con_acc = (logits.argmax(1) == arange).long().sum().item() / batch_size * 100.0
            if dual:
                con_acc = (con_acc + (logits.argmax(0) == arange).long().sum().item() / batch_size * 100.0) / 2.0
        return con_loss.mean(), con_acc

    @staticmethod
    def add_token(sequence, token):
        return torch.cat((token.expand(sequence.size(0), -1, -1), sequence), dim=1)

    def forward(self, x, pos=None, unsupervised=True):
        o = {}
        if x.ndim == 4:
            B, two, V, D = x.size()  # B2VD
            if unsupervised:
                assert two == 2
        else:
            B, V, D = x.size()  # BVD
            two = 1
        projected_input = self.input_projection(x.view(-1, V, D))  # B2, V, D
        h = projected_input
        pos = self.pos_encoding_mlp(pos.reshape(-1, V, pos.size(-1)))  # B2, V, D
        if unsupervised:
            mask = torch.rand(h.size(0), h.size(1), dtype=h.dtype, device=h.device) < self.mask_ratio
            h = torch.where(mask.unsqueeze(-1), self.msk_token.expand_as(h), h)
        h = h + pos
        h = self.add_token(h, self.set_cls_token)  # B2, 1+V, D
        for block in self.transformer:
            h = block(h)
        h = self.norm(h)
        set_cls = h[:, 0]
        if unsupervised:
            set_cls = F.normalize(self.set_head(set_cls), dim=-1).view(B, two, -1)  # B2, D => B, 2, D
            o['set_loss'], o['set_acc'] = self.contrastive_loss(set_cls[:, 0], set_cls[:, 1], self.set_temperature)
        else:
            o['set_cls'] = set_cls
        h = h[:, 1:]
        if unsupervised:
            target = self.dense_target_head(x)
            target = F.normalize(target, dim=-1)
            target = target.view(-1, target.size(-1))
            prediction = F.normalize(self.dense_pred_head(h), dim=-1).view(-1, target.size(-1))  # B2, V, D
            logits = torch.einsum('nc,mc->nm', prediction, target) / self.dense_temperature
            mask = mask.view(-1)
            o['dense_loss'] = -torch.masked_select(torch.diag(torch.log_softmax(logits, dim=1)), mask).mean()
            with torch.no_grad():
                o['dense_acc'] = torch.masked_select(
                    logits.argmax(dim=1) == torch.arange(logits.size(0), device=logits.device),
                    mask).float().mean() * 100.0
        else:
            o['h'] = h
            o['x'] = x.view(-1, V, D)
        return o
