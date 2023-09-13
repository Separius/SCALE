import os
import json
import pickle
import argparse
from glob import glob
from argparse import Namespace

import torch
import jsonlines
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

from slowfast.datasets.ucf_orig import UCF
from slowfast.datasets.hmdb_orig import HMDB
from slowfast.config.defaults_orig import load_config

import utils
from scale import LinearOutputMode, LinearOnTop, MLP
from train import get_scale, split_in_half, get_db, subsample, get_optimizer


class Linear(nn.Module):
    def __init__(self, scale, linear_on_top, freeze):
        super().__init__()
        self.scale = scale
        self.linear_on_top = linear_on_top
        self.freeze = freeze

    def get_scale_output(self, x, pos, disjoint_views=False):
        if self.scale is not None and disjoint_views:
            N, V, D = x.size()
            x = x.view(N * V, 1, D)
            pos = pos.view(N * V, 1, -1)
            o = self.scale(x, pos, unsupervised=False)
            for key in ['x', 'h', 'set_cls']:
                if key in o:
                    o[key] = o[key].view(N, V, -1)
            return o
        return self.scale(x, pos, unsupervised=False) if self.scale is not None else dict(x=x)

    def forward(self, x, pos=None, output_mode=LinearOutputMode.logits, disjoint_views=False, no_logits=False):
        if self.freeze:
            with torch.no_grad():
                o = self.get_scale_output(x, pos, disjoint_views)
        else:
            o = self.get_scale_output(x, pos, disjoint_views)
        return self.linear_on_top(o, output_mode, no_logits=no_logits)


def parse_args(given_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--percentage', type=float, default=1.0, help='train % (only for Kinetics and SSV2)')
    parser.add_argument('--linear_db', type=utils.bool_flag, default=True)  # lin or aug (db to supervisely train on)
    parser.add_argument('--dataset', type=str, default='ssv2',
                        choices=['kinetics', 'ucf', 'hmdb', 'ssv2'])
    parser.add_argument('--scale', type=utils.bool_flag, default=True)
    parser.add_argument('--load', type=int, default=None, help='None => not load, -1 => last, other => epoch to load')
    parser.add_argument('--freeze', type=utils.bool_flag, default=False)
    parser.add_argument('--mlp', type=utils.bool_flag, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--db_split', type=int, default=1, help='only used for hmdb and ucf', choices=[1, 2, 3])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup_percentage', type=float, default=0.05)
    parser.add_argument('--batch_norm', type=utils.bool_flag, default=False)
    parser.add_argument('--train_output_mode', type=LinearOutputMode, choices=list(LinearOutputMode), default='logits')
    parser.add_argument('--num_views', type=int, default=8)
    parser.add_argument('--adam', type=utils.bool_flag, default=True)
    return parser.parse_args(given_args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, metric_logger, dataloader, optimizer, scheduler, args):
    model = model.train()
    if args.freeze:
        model.scale.eval()
    for x, pos, label in metric_logger.log_every(dataloader, 100):
        x, pos, label = split_in_half(x, pos, args.num_views, two=1, label=label)
        optimizer.zero_grad(set_to_none=True)
        out = model(x.float(), pos.float(), args.train_output_mode)
        if args.train_output_mode is LinearOutputMode.logits:
            label = label.unsqueeze(1).expand(-1, out.size(-1))
            loss = F.cross_entropy(out, label)  # B, C, V and B, V
        elif args.train_output_mode is LinearOutputMode.avg_logits:
            loss = F.cross_entropy(out, label)
        else:
            loss = F.nll_loss(out, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss = loss.item()
        with torch.no_grad():
            train_acc = (out.argmax(1) == label).float().mean().item() * 100.0
            metric_logger.update(loss=loss, acc=train_acc)


@torch.no_grad()
def eval_model(model, val_dataloader, best_acc, epoch, metric_logger, best_epoch,
               acc_at_best, best_mode, disjoint_views=False):
    model = model.eval()
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    count = 0
    for x, pos, label in val_dataloader:
        x, pos, label = split_in_half(x, pos, -1, two=1, label=label)
        for output_mode in [LinearOutputMode.avg_logits, LinearOutputMode.avg_probs, LinearOutputMode.logits]:
            out = model(x.float(), pos.float(), output_mode, disjoint_views=disjoint_views)
            if output_mode is LinearOutputMode.avg_logits:
                bs = out.size(0)
                count += bs
            if output_mode is LinearOutputMode.logits:
                label = label.unsqueeze(1).expand(-1, out.size(-1))
            acc = (out.argmax(1) == label).float().mean() * bs
            if output_mode is LinearOutputMode.avg_logits:
                acc1 += acc
            elif output_mode is LinearOutputMode.avg_probs:
                acc2 += acc
            else:
                acc3 += acc
    acc1 = acc1.item() * 100.0 / count
    acc2 = acc2.item() * 100.0 / count
    acc3 = acc3.item() * 100.0 / count
    current_best_acc = max(acc1, acc2, acc3)
    if current_best_acc > best_acc:
        best_acc = current_best_acc
        best_epoch = epoch + 1
        acc_at_best = metric_logger.acc.avg
        if acc1 == current_best_acc:
            best_mode = LinearOutputMode.avg_logits
        elif acc2 == current_best_acc:
            best_mode = LinearOutputMode.avg_probs
        else:
            best_mode = LinearOutputMode.logits
    return best_acc, best_epoch, acc_at_best, best_mode, acc1, acc2, acc3


def get_dl(params, args, train=True):  # params is from the checkpoint and args is for this evaluation
    postfix = ('lin' if args.linear_db else 'aug') if train else 'val'
    with open(f'extracted_features_final/{args.dataset}_{postfix}.pkl', 'rb') as f:
        d = pickle.load(f)
        assert len(d['nan_indices']) == 0
        db_len = d['db_len']
        num_views = d['num_views']
    db_name = 'k400' if args.dataset == 'kinetics' else args.dataset
    label, pos, db = get_db(db_name, params.initialization_model, postfix, db_len, num_views,
                            './extracted_features_final')
    selected_indices = None
    if db_name in {'hmdb', 'ucf'}:
        percentage = 1.0
        ps = set()
        if db_name == 'ucf':
            if train:
                with open(f'./ucf_splits/trainlist0{args.db_split}.txt') as f:
                    for line in f:
                        ps.add(line.split()[0])
            else:
                with open(f'./ucf_splits/testlist0{args.db_split}.txt') as f:
                    for line in f:
                        ps.add(line.strip())
            dataset = UCF(load_config(f'./SlowFast/configs/linear_k400_16x4.yaml'), 'train')
            selected_indices = [i for i, p in enumerate(dataset._path_to_videos) if p in ps]
        else:
            for ff in glob(f'./hmdb_splits/*_test_split{args.db_split}.txt'):
                with open(ff) as f:
                    for line in f:
                        p, m = line.split()
                        if int(m) == (1 if train else 2):
                            ps.add(p)
            dataset = HMDB(load_config(f'./SlowFast/configs/linear_k400_16x4.yaml'), 'train')
            selected_indices = [i for i, p in enumerate(dataset._path_to_videos) if p.split('/')[1] in ps]
    elif db_name in {'k400', 'ssv2'}:
        percentage = args.percentage
    else:
        raise ValueError
    if percentage != 1.0:
        selected_indices = subsample(label, percentage)
    if selected_indices is not None:
        db = db[selected_indices]
        pos = pos[selected_indices]
        label = label[selected_indices]
    label = torch.from_numpy(label).long().cuda()
    db = torch.from_numpy(db).cuda()
    pos = torch.from_numpy(pos).float().cuda()
    dataloader = DataLoader(TensorDataset(db, pos, label), shuffle=train, batch_size=args.batch_size, drop_last=train)
    if train:
        return dataloader, db.size(-1)
    return dataloader


def knn_classifier(train_features, train_labels, test_features, test_labels, num_classes, k=20, T=0.07):
    top1, total = 0.0, 0
    train_features = train_features.t()
    num_test_images = test_labels.shape[0]
    imgs_per_chunk = 128
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)
    return top1 * 100.0 / total


@torch.no_grad()
def knn(train_loader, val_loader, model, disjoint_views=False, one_random_view=True):
    def extract_features(loader):
        all_outs = []
        all_labels = []
        for x, pos, label in loader:
            x, pos, label = split_in_half(x, pos, -1, two=1, label=label)
            out = model(x.float(), pos.float(), None, disjoint_views=disjoint_views, no_logits=True)  # B, V, d
            if one_random_view:
                out = out.mean(1)
            else:
                label = label.unsqueeze(1).expand(-1, out.size(1))  # B, V
                out = out.view(-1, out.size(-1))
                label = label.reshape(-1)
            all_outs.append(out.cpu())
            all_labels.append(label.cpu())
        all_outs = torch.cat(all_outs, dim=0)
        all_outs = all_outs - all_outs.mean(dim=0, keepdim=True)
        return nn.functional.normalize(all_outs, dim=1, p=2), torch.cat(all_labels, dim=0)

    model.eval()
    train_features, train_labels = extract_features(train_loader)
    val_features, val_labels = extract_features(val_loader)
    return knn_classifier(train_features, train_labels, val_features, val_labels, val_labels.max().item() + 1)


def run():
    args = parse_args()
    output_dir = os.path.join('./experiments/', args.exp_name)
    with open(os.path.join(output_dir, 'args.json'), 'r') as f:
        params = Namespace(**json.load(f))
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    dataloader, input_embed_dim = get_dl(params, args, train=True)
    val_dataloader = get_dl(params, args, train=False)
    params.dropout = args.dropout
    scale = get_scale(params, input_embed_dim) if args.scale else None
    if scale is not None:
        if args.load is not None:
            if args.load == -1:
                p = os.path.join(output_dir, f'./checkpoint_{args.exp_name}.pth')
            else:
                p = os.path.join(output_dir, f'./checkpoint_{args.exp_name}_ep{args.load}.pth')
            scale.load_state_dict(torch.load(p)['student'])
        if args.freeze:
            for param in scale.parameters():
                param.requires_grad = False
        if hasattr(scale, 'msk_token'):
            del scale.msk_token
        if scale.set_dim > 0:
            del scale.set_head
        if hasattr(scale, 'dense_target_head'):
            del scale.dense_target_head
        if hasattr(scale, 'dense_pred_head'):
            del scale.dense_pred_head
    num_classes = {'kinetics': 400, 'ucf': 101, 'hmdb': 51, 'ssv2': 174}[args.dataset]
    if args.mlp:
        assert scale is None
        linear = MLP(input_embed_dim, hidden_size=2048, output_dim=512, use_h_avg_pool=False,
                     batch_norm=args.batch_norm, num_layers=2, num_classes=num_classes)
    else:
        linear = LinearOnTop(input_embed_dim, hidden_size=params.hidden_size if scale is not None else 0,
                             append_h=False, num_classes=num_classes, use_h_avg_pool=False, batch_norm=args.batch_norm)
    model = Linear(scale, linear, args.freeze).cuda()
    knn_accuracy = knn(dataloader, val_dataloader, model, disjoint_views=False, one_random_view=True)
    print(f'KNN Accuracy: {knn_accuracy:.2f}')
    num_params = count_parameters(model)
    ipe = len(dataloader)
    optimizer, scheduler = get_optimizer(params, model, ipe, args.adam)
    best_acc, best_mode, best_epoch, acc_at_best, final_acc, best_disjoint_acc, best_disjoint_mode = \
        -1.0, 'none', -1, -1, -1, -1.0, 'none'
    for epoch in range(args.epochs):
        metric_logger = utils.MetricLogger(delimiter="  ")
        train_model(model, metric_logger, dataloader, optimizer, scheduler, args)
        best_acc, best_epoch, acc_at_best, best_mode, acc1, acc2, acc3 = eval_model(
            model, val_dataloader, best_acc, epoch, metric_logger, best_epoch,
            acc_at_best, best_mode, False)
        final_acc = metric_logger.acc.avg
    with jsonlines.open('results.jsonl' if args.tag is None else f'results_{args.tag}.jsonl', mode='a') as writer:
        writer.write({'args': vars(args), 'params': vars(params),
                      'results': {'best_acc': best_acc, 'best_mode': best_mode, 'best_epoch': best_epoch,
                                  'best_disjoint_acc': best_disjoint_acc, 'best_disjoint_mode': best_disjoint_mode,
                                  'num_params': num_params, 'train_acc_at_best': acc_at_best,
                                  'train_acc_at_end': final_acc, 'knn_accuracy': knn_accuracy}})
    output = f'BestAcc={best_acc:.2f} with Mode={best_mode}@{best_epoch}, knn_accuracy={knn_accuracy:.2f}, ' \
             f'BestDisjointAcc={best_disjoint_acc:.2f} with Mode={best_disjoint_mode}, ' \
             f'NumParams={num_params}, TrainAcc@Best={acc_at_best:.2f}, TrainAcc@End={final_acc:.2f}'
    print(output)


if __name__ == '__main__':
    run()
