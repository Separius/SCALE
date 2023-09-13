#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os

from slowfast.datasets.dataset import VideoDataset


class Kinetics(VideoDataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=1):
        self.root = '/data/cvg/datasets/Videos/kinetics-dataset/k400/'
        super().__init__(cfg, mode, num_retries)

    def _construct_loader(self):
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        labels = []
        with open(os.path.join(self.root, f'{self.mode}_meta.csv'), 'r') as f:
            for clip_idx, line in enumerate(f):
                path, _, _, label = line.strip().split(',')
                for idx in range(self._num_clips):
                    self._path_to_videos.append(path.strip())
                    labels.append(label.strip().strip('"'))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert len(self._path_to_videos) > 0, "Failed to load Kinetics split {}".format(self.mode)
        labels_dict = {k: i for i, k in enumerate(sorted(set(labels)))}
        self._labels = [labels_dict[label] for label in labels]

    def get_video_path(self, index):
        return os.path.join(self.root, f'{self.mode}_resized320/{self._path_to_videos[index]}.mp4')


def sample():
    from slowfast.config.defaults_orig import load_config
    from slowfast.datasets.kinetics_orig import Kinetics

    cfg = load_config('./SlowFast/configs/byol_k400_16x4.yaml')
    o = Kinetics(cfg, 'train')[0]  # Tensor(num_decode, 3, frames, h, w), label, index, Tensor(num_decode, 7)
    o = Kinetics(cfg, 'val')[0]  # Tensor(_num_clips, 3, frames, h, w), label, index, Tensor(_num_clips, 7)
