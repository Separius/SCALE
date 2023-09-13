import os
import pickle
import argparse
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import slowfast.utils.logging as logging
from slowfast.datasets.ucf_orig import UCF
from slowfast.datasets.lvu_orig import LVU
from slowfast.datasets.hmdb_orig import HMDB
from slowfast.datasets.ssv2_orig import SSV2
from slowfast.datasets.kinetics_orig import Kinetics
from slowfast.config.defaults_orig import load_config

import utils
from models.interface import get_initialization_model

logger = logging.get_logger(__name__)


class ModelWrapper(nn.Module):
    def __init__(self, backbone, is_resnet):
        super().__init__()
        self.backbone = backbone
        self.is_resnet = is_resnet

    def forward(self, batch):
        with torch.no_grad():
            result = []
            for k, v in batch.items():
                if k in {'index', 'label', 'pos_info'}:
                    continue
                if self.is_resnet:
                    result.append(self.backbone([torch.cat(v)]))
                else:
                    result.append(self.backbone(v, num_tokens=self.num_tokens, token_drop=self.token_drop))
            class_tokens = torch.cat(result, dim=0)
        return class_tokens


def get_model_simplified(args, f, g, is_resnet):
    model, head = f(g(), convert_to_flash=args.flash_attention)
    return ModelWrapper(model, is_resnet)


def parse_args(given_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./extracted_features/')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--augment', type=str, default='linear', choices=['byol', 'linear'])
    parser.add_argument('--dataset', type=str, default='kineticts',
                        choices=['kinetics', 'ucf', 'hmdb', 'ssv2', 'lvu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size_per_gpu', type=int, default=4, help='number of distinct videos(not clips) loaded')
    parser.add_argument('--chunks', type=int, default=1, help='we will split the minibatch into this many chunks')
    parser.add_argument('--num_workers', default=3, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--dist_url', default="env://", type=str, help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank for distrbuted training.')
    parser.add_argument('--fp16', type=utils.bool_flag, default=False)
    parser.add_argument('--resume', type=utils.bool_flag, default=False)
    return parser.parse_args(given_args)


def get_nan_indices(args, labels, pos_info, mem, force=False):
    postfix = 'val' if args.mode == 'val' else ('aug' if args.augment == 'byol' else 'lin')
    p = os.path.join(args.output_dir, f'{args.dataset}_{postfix}.pkl')
    if not force:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)['nan_indices']
    s = set()
    db_len = labels.shape[0]
    arange = np.arange(db_len)
    s = s.union(set(arange[labels == -1].tolist()))
    db = pos_info.reshape((db_len, -1)).mean(axis=1)
    s = s.union(set(arange[db != db].tolist()))
    for x in mem.values():
        db = x.reshape((db_len, -1)).mean(axis=1)
        s = s.union(set(arange[db != db].tolist()))
    s = list(s)
    with open(p, 'wb') as f:
        pickle.dump({'db_len': db_len, 'nan_indices': s, 'num_views': pos_info.shape[1]}, f)
    return s


def get_dataset(args):
    cfg = load_config(f'./SlowFast/configs/{args.augment}_k400_16x4.yaml')
    if args.dataset == 'kinetics':
        dataset = Kinetics(cfg, args.mode)
    elif args.dataset == 'ucf':
        dataset = UCF(cfg, args.mode)
    elif args.dataset == 'lvu':
        dataset = LVU(cfg, args.mode)
    elif args.dataset == 'hmdb':
        dataset = HMDB(cfg, args.mode)
    elif args.dataset == 'ssv2':
        dataset = SSV2(cfg, args.mode)
    else:
        raise ValueError
    nsv = dataset._num_clips if args.mode == 'val' else dataset.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL
    return dataset, len(dataset), nsv


def get_mems(is_main_process, dataset_len, num_student_views, models, args):
    mem_mode = 'w+' if is_main_process and not args.resume else 'r+'
    mem = {}
    extra_str = 'val' if args.mode == 'val' else ('aug' if args.augment == 'byol' else 'lin')
    ds = 'k400' if args.dataset == 'kinetics' else args.dataset
    labels = np.memmap(os.path.join(args.output_dir, f'{ds}_lbl_{extra_str}.dat'), dtype=np.int16,
                       mode=mem_mode, offset=0, shape=(dataset_len,))
    if not args.resume:
        labels.fill(-1)
    dtype = np.float16 if args.fp16 else np.float32
    pos_info = np.memmap(os.path.join(args.output_dir, f'{ds}_pos_{extra_str}.dat'),
                         dtype=dtype, mode=mem_mode, offset=0, shape=(dataset_len, num_student_views, 7))
    if not args.resume:
        pos_info.fill(np.nan)
    for initialization_model, model in models.items():
        dim = 2048 if initialization_model == 'pbyol' else model.embed_dim
        middle = {'pbyol': 'byl', 'svt_b_pt_k400_20': 'svt', 'vmae_b_ft_k400_1600': 'vbf', 'vmae_b_ft_ssv2_2400': 'vsf',
                  'vmae_b_pt_k400_1600': 'vbp', 'vmae_l_pt_k400_1600': 'vlp', 'vmae_b_pt_ssv2_2400': 'vbs'}[
            initialization_model]
        mem[initialization_model] = np.memmap(filename=os.path.join(args.output_dir, f'{ds}_{middle}_{extra_str}.dat'),
                                              dtype=np.float16 if args.fp16 else np.float32, mode=mem_mode, offset=0,
                                              shape=(dataset_len, num_student_views, dim))
        if not args.resume:
            mem[initialization_model].fill(np.nan)
    return labels, pos_info, mem


def h(tensor, fp16):
    if fp16:
        return tensor.half()
    return tensor


def get_models(args):
    models = {}
    for initialization_model in get_initialization_model(None).keys():
        student = get_model_simplified(args, *get_initialization_model(initialization_model),
                                       initialization_model == 'pbyol').backbone.cuda()
        if args.fp16:
            student = student.half()
        for p in student.parameters():
            p.requires_grad = False
        models[initialization_model] = student.eval()
    return models


@torch.no_grad()
def extract():
    args = parse_args()
    if args.mode == 'val' and args.augment == 'byol':
        raise ValueError
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    args.flash_attention = args.fp16
    args.token_drop = False
    args.pin_memory = True
    models = get_models(args)
    dataset, dataset_len, num_student_views = get_dataset(args)
    if args.rank == 0:
        print(f'Num Models={len(models)}, DB_LEN={dataset_len}')
        labels, pos_info, mem = get_mems(True, dataset_len, num_student_views, models, args)
    dist.barrier()
    if args.rank != 0:
        labels, pos_info, mem = get_mems(False, dataset_len, num_student_views, models, args)
    if args.resume:
        if args.rank == 0:
            nan_indices = get_nan_indices(args, labels, pos_info, mem, force=True)
            print(f'Nan indices={len(nan_indices)}')
        dist.barrier()
        if args.rank != 0:
            nan_indices = get_nan_indices(args, labels, pos_info, mem)
        if len(nan_indices) == 0:
            return
        dataset.set_indices(nan_indices)
    data_loader = DataLoader(
        dataset, sampler=DistributedSampler(dataset, shuffle=False), batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    metric_logger = utils.MetricLogger(delimiter="  ")
    for batch_data in metric_logger.log_every(data_loader, 1,
                                              header=f'extracting {args.dataset}_{args.mode}_{args.augment} features'):
        x, label, index, pos = batch_data  # (B,views,3,frames,h,w) / (B,) / (B,) / (B,views,7)
        x = torch.flatten(h(x.cuda(non_blocking=True), args.fp16), 0, 1)  # Bxviews, 3, frames, h, w
        for initialization_model, model in models.items():
            if args.chunks == 1:
                o = model([x] if initialization_model == 'pbyol' else x)
            else:
                o = torch.cat(
                    [model([xx] if initialization_model == 'pbyol' else xx).cpu() for xx in x.chunk(args.chunks)],
                    dim=0)
            mem[initialization_model][index, :, :] = o.view(-1, num_student_views, o.size(-1)).cpu().numpy()  # B, V, C
        labels[index] = label
        pos_info[index, :, :] = h(pos, args.fp16)
    for v in mem.values():
        v.flush()
    labels.flush()
    pos_info.flush()
    dist.barrier()
    if args.rank == 0:
        post_nan_indices = get_nan_indices(args, labels, pos_info, mem, force=True)
        print(f'Len(nan): {len(post_nan_indices)}')


if __name__ == "__main__":
    extract()
