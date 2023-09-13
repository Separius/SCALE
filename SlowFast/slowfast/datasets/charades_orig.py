#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
from itertools import chain as chain

import torch
import numpy as np
import torch.utils.data

import slowfast.utils.logging as logging
import slowfast.datasets.utils_orig as utils
from slowfast.datasets.dataset import VideoDataset

logger = logging.get_logger(__name__)


class Charades(VideoDataset):
    """
    Charades video loader. Construct the Charades video loader, then sample
    clips from the videos. For training and validation, a single clip is randomly
    sampled from every video with random cropping, scaling, and flipping. For
    testing, multiple clips are uniformaly sampled from every video with uniform
    cropping. For uniform cropping, we take the left, center, and right crop if
    the width is larger than height, or take top, center, and bottom crop if the
    height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=1):
        """
        Load Charades data (frame paths, labels, etc. ) to a given Dataset object.
        The dataset could be downloaded from Chrades official website
        (https://allenai.org/plato/charades/).
        Please see datasets/DATASET.md for more information about the data format.
        Args:
            dataset (Dataset): a Dataset object to load Charades data to.
            mode (string): 'train', 'val', or 'test'.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        self.root = '/data/cvg/datasets/Videos/charades/'  # num classes is 157
        super().__init__(cfg, mode, num_retries)

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(self.root, f'{self.mode}.csv')
        assert os.path.exists(path_to_file), f'{path_to_file} dir not found'
        (self._path_to_videos, self._labels) = utils.load_image_lists(
            path_to_file, os.path.join(self.root, 'Charades_v1_rgb'), return_list=True)
        if self.mode == 'val':
            # Form video-level labels from frame level annotations.
            self._labels = utils.convert_to_video_level_labels(self._labels)
        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [range(self._num_clips) for _ in range(len(self._labels))]
            )
        )

        logger.info(
            "Charades dataloader constructed (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def get_seq_frames(self, index):
        """
        Given the video index, return the list of indexs of sampled frames.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of sampled frames from the video.
        """
        temporal_sample_index = (
            -1 if self.mode == 'train' else self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS)
        num_frames = self.cfg.DATA.NUM_FRAMES
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        video_length = len(self._path_to_videos[index])
        assert video_length == len(self._labels[index])
        clip_length = (num_frames - 1) * sampling_rate + 1
        if temporal_sample_index == -1:
            if clip_length > video_length:
                start = random.randint(video_length - clip_length, 0)
            else:
                start = random.randint(0, video_length - clip_length)
        else:
            gap = float(max(video_length - clip_length, 0)) / (
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1
            )
            start = int(round(gap * temporal_sample_index))
        seq = [
            max(min(start + i * sampling_rate, video_length - 1), 0)
            for i in range(num_frames)
        ]
        return seq

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]
        xs = []
        labels = []
        poses = []
        for idx in range(self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL if self.mode == 'train' else self._num_clips):
            x, label, _, pos = self.get_item(index if self.mode == 'train' else (index * self._num_clips + idx))
            xs.append(x)
            labels.append(label)
            poses.append(pos)
        return torch.stack(xs, dim=0), torch.stack(labels, dim=0) if self.mode == 'train' else labels[-1], index, \
               torch.from_numpy(self.normalize_pos_info(torch.stack(poses, dim=0).numpy()))

    def get_item(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
            time index (zero): The time index is currently not supported.
            {} extra data, currently not supported
        """
        if self.mode == 'train':
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        seq = self.get_seq_frames(index)
        frames = torch.as_tensor(  # T, H, W, 3
            utils.retry_load_images([self._path_to_videos[index][frame] for frame in seq], self._num_retries))
        label = utils.aggregate_labels([self._labels[index][i] for i in range(seq[0], seq[-1] + 1)])
        label = torch.as_tensor(utils.as_binary_vector(label, 157)).bool()  # (157,)

        # Perform color normalization.
        frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames, flipped, h0, w0, h1, w1 = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE
        )
        pos_info = np.zeros((7,), dtype=np.float32)
        pos_info[0] = flipped
        pos_info[1] = h0
        pos_info[2] = w0
        pos_info[3] = h1
        pos_info[4] = w1
        pos_info[5] = seq[0]
        pos_info[6] = seq[-1]
        return frames, label, index, torch.from_numpy(pos_info)
