import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import wandb
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim._multi_tensor import AdamW, SGD
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import utils
from scale import Scale


def parse_args(given_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--extracted_features', type=str, default='./extracted_features')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dense_temperature', type=float, default=0.2)
    parser.add_argument('--set_temperature', type=float, default=0.2)
    parser.add_argument('--set_loss_weight', type=float, default=1.0)
    parser.add_argument('--dense_loss_weight', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_percentage', type=float, default=0.05)
    parser.add_argument('--linear_db', type=utils.bool_flag, default=True)  # lin or aug
    parser.add_argument('--dense_contrastive_dim', type=int, default=128, help='0 to disable')
    parser.add_argument('--set_contrastive_dim', type=int, default=128, help='set to 0 to disable')
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mlp_ratio', type=float, default=2.0)
    parser.add_argument('--num_views', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512, help='total batch size')
    parser.add_argument('--initialization_model', type=str, default='byl',
                        choices=['byl', 'svt', 'vbf', 'vbp', 'vlp'])
    parser.add_argument('--dataset', type=str, default='kinetics',
                        choices=['kinetics', 'ucf', 'hmdb', 'ssv2', 'k400'])  # k400 == kinetics
    return parser.parse_args(given_args)


def get_parameter_groups(model):
    not_regularized = []
    regularized = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim == 1 or name.endswith('.bias') or name.endswith('_token'):
                not_regularized.append(p)
            else:
                regularized.append(p)
    return [{'params': not_regularized, 'weight_decay': 0.0}, {'params': regularized}]


def warm_up_cosine_lr_scheduler(optimizer, T_max, warm_up_iterations=0, eta_min=0.0):
    if warm_up_iterations <= 0:
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    def warm_up_with_cosine_lr(epoch):
        return eta_min + (epoch / warm_up_iterations) if epoch <= warm_up_iterations else 0.5 * (
                np.cos((epoch - warm_up_iterations) / (T_max - warm_up_iterations) * np.pi) + 1)

    return LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)


@torch.no_grad()
def split_in_half(x, y, num_views, two=2, label=None):
    N, L, D = x.shape  # batch, length, dim
    if num_views == -1:
        num_views = L // two
    len_keep = two * num_views
    assert len_keep <= L
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)).view(N, two, num_views, -1)
    y = torch.gather(y, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, y.size(-1))).view(N, two, num_views, -1)
    if two == 1:
        return x.squeeze(1), y.squeeze(1), label
    return x, y, label


def get_scale(args, embedding_dim):
    return Scale(
        embedding_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, set_dim=args.set_contrastive_dim,
        set_temperature=args.set_temperature, mask_ratio=args.mask_ratio, attn_drop=args.dropout,
        dense_dim=args.dense_contrastive_dim, dense_temperature=args.dense_temperature, drop=args.dropout,
        mlp_ratio=args.mlp_ratio
    ).train().cuda()


def subsample(labels, percentage):
    if percentage < 0.0:
        neg = True
        percentage = -percentage
    else:
        neg = False
    labels_counts = defaultdict(int)
    for label in labels:
        labels_counts[label] += 1
    selected_indices = []
    target_labels_counts = {k: max(int(v * percentage), 1 if v != 0 else 0) for k, v in labels_counts.items()}
    len_labels = len(labels)
    for i, label in enumerate(labels[::-1] if neg else labels):
        if target_labels_counts[label] > 0:
            target_labels_counts[label] -= 1
            selected_indices.append((len_labels - i - 1) if neg else i)
    return selected_indices


def get_db(db_name, middle, postfix, dataset_len, num_student_views, exfea_loc, fp=np.float16):
    labels = np.memmap(f'{exfea_loc}/{db_name}_lbl_{postfix}.dat',
                       dtype=np.int16, mode='r', offset=0, shape=(dataset_len,))
    pos_info = np.memmap(f'{exfea_loc}/{db_name}_pos_{postfix}.dat', dtype=fp,
                         mode='r', offset=0, shape=(dataset_len, num_student_views, 7))
    dim = 2048 if middle == 'byl' else (1024 if middle == 'vlp' else 768)
    db = np.memmap(f'{exfea_loc}/{db_name}_{middle}_{postfix}.dat',
                   dtype=np.float16, mode='r', offset=0, shape=(dataset_len, num_student_views, dim))
    return labels, pos_info, db


def get_dl(args, give_label=False, train=True, percentage=1.0):
    postfix = ('lin' if args.linear_db else 'aug') if train else 'val'
    with open(f'{args.extracted_features}/{args.dataset}_{postfix}.pkl', 'rb') as f:
        d = pickle.load(f)
        assert len(d['nan_indices']) == 0
        db_len = d['db_len']
        num_views = d['num_views']
    db_name = 'k400' if args.dataset == 'kinetics' else args.dataset
    label, pos, db = get_db(db_name, args.initialization_model, postfix, db_len, num_views, args.extracted_features)
    if give_label:
        if percentage != 1.0:
            selected_indices = subsample(label, percentage)
            db = db[selected_indices]
            pos = pos[selected_indices]
            label = label[selected_indices]
        label = torch.from_numpy(label).long().cuda()
        extra_params = [label]
    else:
        if percentage != 1.0:
            raise NotImplementedError
        extra_params = []
    db = torch.from_numpy(db).cuda()
    pos = torch.from_numpy(pos).float().cuda()
    dataset = TensorDataset(db, pos, *extra_params)
    dataloader = DataLoader(dataset, shuffle=train, batch_size=args.batch_size, drop_last=train)
    return dataloader, db.size(-1)


def get_optimizer(args, model, ipe, adam=True):
    if adam:
        optimizer = AdamW(get_parameter_groups(model), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = SGD(get_parameter_groups(model), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    t_max = ipe * args.epochs
    scheduler = warm_up_cosine_lr_scheduler(optimizer, T_max=t_max, warm_up_iterations=args.warmup_percentage * t_max)
    return optimizer, scheduler


def run():
    args = parse_args()
    if args.dataset == 'k400':
        args.dataset = 'kinetics'
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    output_dir = os.path.join('./experiments/', args.exp_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if args.set_contrastive_dim == 0:
        args.set_loss_weight = 0.0
    wandb.init(project='SCALE', config=args, name=args.exp_name)
    args.call_script = ' '.join(sys.argv)
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    dataloader, input_embed_dim = get_dl(args)
    network = get_scale(args, input_embed_dim)
    loss_weight_sum = args.set_loss_weight + args.dense_loss_weight
    ipe = len(dataloader)
    optimizer, scheduler = get_optimizer(args, network, ipe)
    for epoch in range(args.epochs):
        metric_logger = utils.MetricLogger(delimiter='  ')
        j = 0
        for batch in metric_logger.log_every(dataloader, 100):
            optimizer.zero_grad(set_to_none=True)
            x, y, _ = split_in_half(*batch, args.num_views)
            out = network(x.float(), y)
            loss = (out.get('set_loss', 0.0) * args.set_loss_weight +
                    out.get('dense_loss', 0.0) * args.dense_loss_weight) / loss_weight_sum
            loss.backward()
            out['loss'] = loss
            optimizer.step()
            scheduler.step()
            metric_logger.update(**out)
            wandb.log(out, step=ipe * epoch + j)
            j += 1
        save_dict = {'student': network.state_dict(), 'epoch': epoch + 1, 'optimizer': optimizer.state_dict()}
        utils.save_on_master(save_dict, os.path.join(output_dir, f'./checkpoint_{args.exp_name}.pth'))
        if (epoch + 1) % 50 == 0:
            utils.save_on_master(save_dict, os.path.join(output_dir, f'./checkpoint_{args.exp_name}_ep{epoch}.pth'))
    wandb.finish()


if __name__ == '__main__':
    run()
