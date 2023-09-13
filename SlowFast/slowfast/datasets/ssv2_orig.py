#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import json

import slowfast.utils.logging as logging
from slowfast.datasets.dataset import VideoDataset

logger = logging.get_logger(__name__)


class SSV2(VideoDataset):
    def __init__(self, cfg, mode, num_retries=1):
        self.root = '/data/cvg/datasets/Videos/something2/'
        cfg.TEST.NUM_ENSEMBLE_VIEWS = 2
        cfg.TEST.NUM_SPATIAL_CROPS = 3
        # cfg.DATA.TRAIN_CROP_NUM_TEMPORAL = 8
        cfg.DATA.SAMPLING_RATE = 2
        cfg.DATA.RANDOM_FLIP = False
        super().__init__(cfg, mode, num_retries)

    def _construct_loader(self):
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        labels = []
        with open(os.path.join(self.root, f'labels/{self.mode}.json'), 'r') as f:
            for clip_idx, line in enumerate(json.load(f)):
                path, label = line['id'], line['template']
                for idx in range(self._num_clips):
                    self._path_to_videos.append(path)
                    labels.append(label)
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (len(self._path_to_videos) > 0), "Failed to load Kinetics split {}".format(self.mode)
        labels_dict = {k: i for i, k in enumerate(sorted(set(labels)))}
        self._labels = [labels_dict[label] for label in labels]
        logger.info("Constructing kinetics dataloader (size: {})".format(len(self._path_to_videos)))

    def get_video_path(self, index):
        return os.path.join(self.root, f'ssv2_resized_30fps/{self._path_to_videos[index]}.mp4')
