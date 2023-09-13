#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from glob import glob

from natsort import natsorted

import slowfast.utils.logging as logging
from slowfast.datasets.dataset import VideoDataset

logger = logging.get_logger(__name__)


class HMDB(VideoDataset):
    def __init__(self, cfg, mode, num_retries=1):
        self.root = '/data/cvg/datasets/Videos/hmdb51/'
        super().__init__(cfg, mode, num_retries)

    def _construct_loader(self):
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        labels = []
        for clip_idx, line in enumerate(natsorted(glob(self.root + '/*/*.avi'))):
            line = line.rsplit('/', 2)
            label = line[1]
            path = '/'.join(line[1:])
            for idx in range(self._num_clips):
                self._path_to_videos.append(path.strip())
                labels.append(label.strip().strip('"'))
                self._spatial_temporal_idx.append(idx)
                self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (len(self._path_to_videos) > 0), "Failed to load Kinetics split {}".format(self.mode)
        labels_dict = {k: i for i, k in enumerate(natsorted(set(labels)))}
        self._labels = [labels_dict[label] for label in labels]
        logger.info("Constructing hmdb dataloader (size: {})".format(len(self._path_to_videos)))

    def get_video_path(self, index):
        return os.path.join(self.root, self._path_to_videos[index])
