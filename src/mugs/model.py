# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
models and functions for building student and teacher networks for multi-granular losses.
"""
from functools import partial

import torch
import torch.nn as nn

from cpc import CPC
from mugs.vision_transformer import trunc_normal_, Block


class ContrastiveHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=256, pred_hidden_dim=4096,
                 nlayers=3, proj_bn=False, pred_bn=False, norm_before_pred=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.norm_before_pred = norm_before_pred
        self.projector = self._build_mlp(nlayers, in_dim, hidden_dim, out_dim, use_bn=proj_bn)
        self.apply(self._init_weights)
        self.predictor = None
        if pred_hidden_dim > 0:  # teacher no, student yes
            self.predictor = self._build_mlp(nlayers - 1, out_dim, pred_hidden_dim, out_dim, use_bn=pred_bn)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _build_mlp(num_layers, input_dim, hidden_dim, output_dim, use_bn=False):
        mlp = []
        for layer in range(num_layers):
            dim1 = input_dim if layer == 0 else hidden_dim
            dim2 = output_dim if layer == num_layers - 1 else hidden_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if layer < num_layers - 1:
                if use_bn:
                    mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.GELU())
        return nn.Sequential(*mlp)

    def forward(self, x, return_target=False):
        feat = self.projector(x)
        if return_target:
            return nn.functional.normalize(feat, dim=-1, p=2)
        if self.norm_before_pred:
            feat = nn.functional.normalize(feat, dim=-1, p=2)
        return nn.functional.normalize(self.predictor(feat), dim=-1, p=2)


class Group_Superivsion_Head(nn.Module):
    """
    a class to implement Local Group Superivsion Head which is the same as Instance Superivsion Head
    --in_dim: input dimension of projection head
    --hidden_dim: hidden dimension of projection head
    --out_dim: ouput dimension of projection and prediction heads
    --pred_hidden_dim: hidden dimension of prediction head
    --nlayers: layer number of projection head. prediction head has nlayers-1 layer
    --proj_bn: whether we use batch normalization in projection head
    --pred_bn: whether we use batch normalization in prediction head
    --norm_before_pred:  whether we use normalization before prediction head
    """

    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256,
                 nlayers=3, use_bn=False, norm_last_layer=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.projector = self._build_mlp(nlayers, in_dim, hidden_dim, bottleneck_dim, use_bn=use_bn)
        self.apply(ContrastiveHead._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _build_mlp(num_layers, in_dim, hidden_dim, output_dim, use_bn=False):
        """
        build a mlp
        """
        if num_layers == 1:
            mlp = nn.Linear(in_dim, output_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            mlp = nn.Sequential(*layers)
        return mlp

    def forward(self, x):
        return self.last_layer(nn.functional.normalize(self.projector(x), dim=-1, p=2))


class Block_mem(nn.Module):
    """
    a class to implement a memory block for local group supervision
    --dim: feature vector dimenstion in the memory
    --K: memory size
    --top_n: number for neighbors in local group supervision
    """

    def __init__(self, dim, K=2048, top_n=10):
        super().__init__()
        self.dim = dim
        self.K = K
        self.top_n = top_n
        # create the queue
        self.register_buffer("queue_q", torch.randn(K, dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, query, weak_aug_flags=None):
        """
        update memory queue
        """
        len_weak = 0
        query = concat_all_gather(query.contiguous())
        if weak_aug_flags is not None:
            weak_aug_flags = weak_aug_flags.cuda()
            weak_aug_flags = concat_all_gather(weak_aug_flags)
            idx_weak = torch.nonzero(weak_aug_flags)
            len_weak = len(idx_weak)
            if len_weak > 0:
                idx_weak = idx_weak.squeeze(-1)
                query = query[idx_weak]
            else:
                return len_weak

        all_size = query.shape[0]
        ptr = int(self.queue_ptr)
        remaining_size = ptr + all_size - self.K
        if remaining_size <= 0:
            self.queue_q[ptr: ptr + all_size, :] = query
            ptr = ptr + all_size
            self.queue_ptr[0] = (ptr + all_size) % self.K
        else:
            self.queue_q[ptr: self.K, :] = query[0: self.K - ptr, :]
            self.queue_q[0:remaining_size, :] = query[self.K - ptr:, :]
            self.queue_ptr[0] = remaining_size
        return len_weak

    @torch.no_grad()
    def _get_similarity_index(self, x):
        """
        compute the index of the top-n neighbors (key-value pair) in memory
        """
        x = nn.functional.normalize(x, dim=-1)
        queue_q = nn.functional.normalize(self.queue_q, dim=-1)

        cosine = x @ queue_q.T
        _, index = torch.topk(cosine, self.top_n, dim=-1)
        return index

    @torch.no_grad()
    def _get_similarity_samples(self, query, index=None):
        """
        compute top-n neighbors (key-value pair) in memory
        """
        if index is None:
            index = self._get_similarity_index(query)
        get_q = self.queue_q[index.view(-1)]
        B, tn = index.shape
        get_q = get_q.view(B, tn, self.dim)
        return get_q

    def forward(self, query):
        """
        forward to find the top-n neighbors (key-value pair) in memory
        """
        return self._get_similarity_samples(query)


class vit_mem(nn.Module):
    """
    a class to implement a memory for local group supervision
    --dim: feature vector dimenstion in the memory
    --K: memory size
    --top_n: number for neighbors in local group supervision
    """

    def __init__(self, dim, K=2048, top_n=10):
        super().__init__()
        self.block = Block_mem(dim, K, top_n)

    def _dequeue_and_enqueue(self, query, weak_aug_flags=None):
        """
        update memory queue
        """
        return self.block._dequeue_and_enqueue(query, weak_aug_flags)

    def forward(self, query):
        """
        forward to find the top-n neighbors (key-value pair) in memory
        """
        return self.block(query)


class SimplifiedMugsWrapper(nn.Module):
    def __init__(self, backbone, num_tokens, token_drop, is_resnet, cpc):
        super().__init__()
        self.backbone = backbone
        self.num_tokens = num_tokens
        self.token_drop = token_drop
        self.is_resnet = is_resnet
        self.cpc = cpc

    def forward(self, batch):
        with torch.no_grad():
            if not self.is_resnet and self.num_tokens != -1:  # efficient vit forward
                class_tokens = self.backbone(batch, num_tokens=self.num_tokens, token_drop=self.token_drop)
            else:
                result = []
                for k, v in batch.items():
                    if k in {'index', 'label', 'pos_info'}:
                        continue
                    if self.is_resnet:
                        result.append(self.backbone([torch.cat(v)]))
                    else:
                        result.append(self.backbone(v, num_tokens=self.num_tokens, token_drop=self.token_drop))
                class_tokens = torch.cat(result, dim=0)
        if self.cpc is None:
            return class_tokens
        cpc_loss, cpc_acc = self.cpc(class_tokens, batch['pos_info'])
        return cpc_loss, cpc_acc


class Mugs_Wrapper(nn.Module):
    """
    a class to implement a student or teacher wrapper for mugs
    --backbone: the backnone of student/teacher, e.g. ViT-small
    --instance_head: head, including projection/prediction heads, for instance supervision
    --local_group_head: head, including projection/prediction heads, for local group supervision
    --group_head: projection head for group supervision
    """

    def __init__(self, backbone, instance_head, local_group_head, group_head, mem,
                 return_target, num_tokens, num_relation_blocks, token_drop, cpc):
        super().__init__()
        self.backbone = backbone
        self.instance_head = instance_head
        self.local_group_head = local_group_head
        self.group_head = group_head
        self.mem = mem
        self.return_target = return_target
        self.num_tokens = num_tokens
        self.token_drop = token_drop
        self.num_relation_blocks = num_relation_blocks if mem is not None else 0
        if num_relation_blocks > 0 and mem is not None:
            self.relation_blocks = nn.ModuleList(
                [Block(dim=backbone.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                       drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                 for _ in range(num_relation_blocks)])
        self.cpc = cpc

    def _dequeue_and_enqueue(self, new_cls):
        if self.mem is not None:
            return self.mem._dequeue_and_enqueue(new_cls)

    def forward(self, batch, frozen_backbone=-1):
        if self.num_tokens != -1:  # N1xNSAxB+N2xNSAxB+..., C
            x = batch
        else:  # teacher (dense)
            keys = set()
            for k in batch.keys():
                if k not in {'index', 'label', 'pos_info'}:
                    a, b = k.split('_')
                    keys.add((int(a), int(b)))
            largest_crop = sorted(keys)[-1]
            x = torch.cat(batch[f'{largest_crop[0]}_{largest_crop[1]}'])  # N1xNSAxB, ...
        class_tokens = self.backbone(x, num_tokens=self.num_tokens, token_drop=self.token_drop,
                                     frozen_backbone=frozen_backbone)
        if self.num_relation_blocks != 0:
            rx = torch.cat((class_tokens.unsqueeze(1), self.mem(class_tokens)), dim=1)
            for blk in self.relation_blocks:
                rx = blk(rx)
            memory_class_tokens = self.backbone.norm(rx[:, 0])
        else:
            memory_class_tokens = None
        ## target [16, 256] for teacher,  [64, 256] for student
        instance_feat = self.instance_head(class_tokens, self.return_target) if self.instance_head else None
        ## target [16, 256] for teacher,  [64, 256] for student
        local_group_feat = self.local_group_head(memory_class_tokens,
                                                 self.return_target) if self.local_group_head else None
        # target [16, 65536] for teacher, [64, 65536] for student
        group_feat = self.group_head(class_tokens) if self.group_head is not None else None
        if self.cpc is not None:
            cpc_loss, cpc_acc = self.cpc(class_tokens, batch['pos_info'])
        else:
            cpc_loss, cpc_acc = None, None
        return instance_feat, local_group_feat, group_feat, class_tokens.detach(), cpc_loss, cpc_acc


def get_model_simplified(args, num_student_views, f, g, is_resnet, get_cpc):
    model, head = f(g(), convert_to_flash=args.flash_attention)
    embed_dim = model.embed_dim
    if get_cpc:
        cpc = CPC(embed_dim, args.pos_encoding, args.cpc_loss, num_student_views, hidden_size=512, num_layers=2,
                  projection_dim=256, temperature=0.2, mask_ratio=args.cpc_mask_ratio, nsa=args.num_spatial_augs)
    else:
        cpc = None
    return SimplifiedMugsWrapper(model, args.num_tokens, args.token_drop, is_resnet, cpc)


def get_model(args, num_student_views, is_teacher, f, g):
    model, head = f(g(), convert_to_flash=args.flash_attention)
    embed_dim = model.embed_dim
    instance_head, local_group_head, group_head, mem, cpc = None, None, None, None, None
    if args.loss_weights[0] > 0:
        instance_head = ContrastiveHead(
            in_dim=embed_dim, hidden_dim=2048, out_dim=args.instance_out_dim, pred_hidden_dim=0 if is_teacher else 4096,
            nlayers=3, proj_bn=args.use_bn_in_head, pred_bn=False, norm_before_pred=args.norm_before_pred)
    if args.loss_weights[1] > 0:
        local_group_head = ContrastiveHead(
            in_dim=embed_dim, hidden_dim=2048, out_dim=args.local_group_out_dim,
            pred_hidden_dim=0 if is_teacher else 4096, nlayers=3, proj_bn=args.use_bn_in_head,
            pred_bn=False, norm_before_pred=args.norm_before_pred)
        mem = vit_mem(embed_dim, K=args.local_group_queue_size, top_n=args.local_group_knn_top_n)
    if args.loss_weights[2] > 0:
        group_head = Group_Superivsion_Head(
            in_dim=embed_dim, out_dim=args.group_out_dim, hidden_dim=2048, bottleneck_dim=args.group_bottleneck_dim,
            nlayers=3, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer)
        use_head = False
        if args.initialization_model == 'svt_b_pt_k400_20':
            assert args.group_out_dim == 64 * 1024 and args.group_bottleneck_dim == 256 and \
                   not args.use_bn_in_head  # and args.norm_last_layer
            use_head = True
        if args.initialization_model.startswith('ibot_base400_imagenet'):
            assert args.group_out_dim == 8 * 1024 and args.group_bottleneck_dim == 256 \
                   and not args.use_bn_in_head  # and args.norm_last_layer
            use_head = True
        if use_head:
            state_dict = {k.replace('mlp', 'projector'): v
                          for k, v in head.state_dict().items() if not k.startswith('last_layer2')}
            group_head.load_state_dict(state_dict)
            group_head.train()
    if not is_teacher and args.loss_weights[3] > 0:
        cpc = CPC(embed_dim, args.pos_encoding, args.cpc_loss, num_student_views, hidden_size=512, num_layers=2,
                  projection_dim=256, temperature=0.2, mask_ratio=args.cpc_mask_ratio, nsa=args.num_spatial_augs)
    return Mugs_Wrapper(model, instance_head, local_group_head, group_head, mem, is_teacher,
                        -1 if is_teacher else args.num_tokens, args.num_relation_blocks,
                        not is_teacher and args.token_drop, cpc=cpc)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)
