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
functions for building multi-granular losses.
"""
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import slowfast.utils.logging as logging
from mugs.utils import concat_all_gather

logger = logging.get_logger(__name__)


class InfoNCELoss(nn.Module):
    """
    vanilla infoNCEloss.
    --ncrops: how many crops are used in student networks
    --dim: feature dimension in queue determinted by output dimention of student network
    --queue_size: queue size
    --temperature: temperature parameter for infoNCEloss
    """

    def __init__(self, num_teacher_views, num_student_views, dim=256,
                 queue_size=65536, temperature=0.2, nsa=1, avg=False):
        super().__init__()
        self.nsa = nsa
        self.avg = avg
        self.queue_size = queue_size
        self.temperature = temperature
        self.register_buffer('queue', torch.randn(dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.num_teacher_views = num_teacher_views
        self.num_student_views = num_student_views

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr: ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size
        else:
            keys_t = keys.T
            queue_remaining_size = self.queue_size - ptr
            self.queue[:, ptr:] = keys_t[:, :queue_remaining_size]
            self.queue[:, : batch_size - queue_remaining_size] = keys_t[:, queue_remaining_size:]
            ptr = batch_size - queue_remaining_size  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, student_output, teacher_output, epoch):
        # student_output is N1xNSAxB+N2xNSAxB+..., C and teacher_output is N1xNSAxB, C
        preds = student_output.chunk(self.num_student_views)
        targets = teacher_output.detach().chunk(self.num_teacher_views)
        small_crop_loss, large_crop_loss = 0, 0
        small_loss_terms, large_loss_terms = 0, 0
        small_crop_acc, large_crop_acc = 0, 0
        queue_feat = self.queue.clone().detach()
        labels = None
        nsa = self.nsa
        for t_idx, targ in enumerate(targets):
            for p_idx, pred in enumerate(preds):
                if (t_idx // nsa) == (p_idx // nsa):
                    continue
                l_pos = torch.einsum("nc,nc->n", [pred, targ]).unsqueeze(-1)  # positive logits: Nx1
                l_neg = torch.einsum("nc,ck->nk", [pred, queue_feat])  # negative logits: NxK
                logits = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)
                logits /= self.temperature  # apply temperature
                if labels is None or labels.size(0) != logits.size(0):  # labels: positive key indicators
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
                loss = self.CrossEntropyLoss(logits, labels)
                if p_idx < self.num_teacher_views:  ## large crop loss, namely loss on 224-sized images
                    large_crop_loss += loss
                    large_loss_terms += 1
                    with torch.no_grad():
                        large_crop_acc += (logits.argmax(1) == 0).float().mean()
                else:  ## small crop loss, namely loss on 96-sized images
                    small_crop_loss += loss
                    small_loss_terms += 1
                    with torch.no_grad():
                        small_crop_acc += (logits.argmax(1) == 0).float().mean()
            if not self.avg:
                self._dequeue_and_enqueue(targ)
        if self.avg:
            self._dequeue_and_enqueue(torch.stack(targets, dim=0).mean(0))  # list(BxC) (length = N1 x NSA) => BxC
        large_crop_loss /= large_loss_terms
        small_crop_loss /= small_loss_terms
        large_crop_acc /= large_loss_terms
        small_crop_acc /= small_loss_terms
        return 0.5 * (large_crop_loss + small_crop_loss), large_crop_acc.item() * 100.0, small_crop_acc.item() * 100.0


class ClusteringLoss(nn.Module):
    """
    Clustering loss which is very simialr to the one in DINO
    --out_dim: center dimension determinted by output dimention of student network
    --ncrops: how many crops are used in student networks
    --warmup_teacher_temp: Initial value for the teacher temperature
    --teacher_temp: Final value (after linear warmup) of the teacher temperature
    --warmup_teacher_temp_epochs: Number of warmup epochs for the teacher temperature
    --nepochs: total training epoch
    --student_temp: temperature parameter in student output
    --center_momentum:  EMA parameter for center update
    """

    def __init__(self, out_dim, num_teacher_views, num_student_views, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9, center=None, nsa=1):
        super().__init__()
        self.nsa = nsa
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_teacher_views = num_teacher_views
        self.num_student_views = num_student_views
        self.register_buffer("center", torch.zeros(1, out_dim))
        if center is not None:
            with torch.no_grad():
                self.center.copy_(center.data)
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training unstable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
             np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.num_student_views)
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.num_teacher_views)
        loss_large_crop, loss_small_crop = 0.0, 0.0
        loss_terms_large_crop, loss_terms_small_crop = 0, 0
        small_crop_acc, large_crop_acc = 0, 0
        nsa = self.nsa
        for iq, q in enumerate(teacher_out):
            with torch.no_grad():
                q_label = q.argmax(1)
            for v in range(len(student_out)):
                if (v // nsa) == (iq // nsa):  # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean()
                with torch.no_grad():
                    v_label = student_out[v].argmax(1)
                    crop_acc = (q_label == v_label).float().mean()
                if v < self.num_teacher_views:
                    loss_large_crop += loss
                    loss_terms_large_crop += 1
                    large_crop_acc += crop_acc
                else:
                    loss_small_crop += loss
                    loss_terms_small_crop += 1
                    small_crop_acc += crop_acc
        self.update_center(teacher_output)
        loss_large_crop /= loss_terms_large_crop
        loss_small_crop /= loss_terms_small_crop
        large_crop_acc /= loss_terms_large_crop
        small_crop_acc /= loss_terms_small_crop
        return 0.5 * (loss_large_crop + loss_small_crop), large_crop_acc.item() * 100.0, small_crop_acc.item() * 100.0

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=False)
        dist.all_reduce(batch_center)
        batch_center = batch_center / dist.get_world_size()
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)  # ema


def get_multi_granular_loss(args, num_teacher_views, num_student_views):
    nsa = args.num_spatial_augs
    all_weights = {"instance-sup.": args.loss_weights[0], "local-group-sup.": args.loss_weights[1],
                   "group-sup.": args.loss_weights[2], 'cpc-sup.': args.loss_weights[3]}
    if args.loss_weights[2] > 0 and args.initialization_model.startswith('ibot_base400'):
        center = torch.load('./initialization/ibot_base400_imagenet.pth', map_location='cpu')['ibot_loss']['center']
    else:
        center = None
    all_losses = {
        'instance-sup.': InfoNCELoss(
            num_teacher_views, num_student_views, dim=args.instance_out_dim, queue_size=args.instance_queue_size,
            temperature=args.instance_temp, nsa=nsa, avg=args.avg_momentum).cuda() if args.loss_weights[
                                                                                          0] > 0 else None,
        'local-group-sup.': InfoNCELoss(
            num_teacher_views, num_student_views, dim=args.local_group_out_dim, queue_size=args.local_group_queue_size,
            temperature=args.local_group_temp, nsa=nsa, avg=args.avg_momentum).cuda() if args.loss_weights[
                                                                                             1] > 0 else None,
        'group-sup.': ClusteringLoss(
            args.group_out_dim, num_teacher_views, num_student_views, args.group_warmup_teacher_temp,
            args.group_teacher_temp, args.group_warmup_teacher_temp_epochs, args.epochs, center=center, nsa=nsa,
            student_temp=args.group_student_temp, center_momentum=0.9).cuda() if args.loss_weights[2] > 0 else None}
    return all_losses, all_weights
