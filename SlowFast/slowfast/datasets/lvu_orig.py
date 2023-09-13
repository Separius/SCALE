#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from glob import glob

from natsort import natsorted

import slowfast.utils.logging as logging
from slowfast.datasets.dataset import VideoDataset

logger = logging.get_logger(__name__)


class LVU(VideoDataset):
    def __init__(self, cfg, mode, num_retries=1):
        self.root = '/home/cvg/data/datasets/Videos/lvu/'
        super().__init__(cfg, mode, num_retries)

    def _construct_loader(self):
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        labels = []
        clip_idx = 0
        with open(self.root+f'lvu_1.0/genre/{self.mode}.csv', 'r') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                else:
                    line = line.strip().split(' ')
                    path, label = self.root+f'videos/{line[2]}.mp4', int(line[0])
                    if os.path.exists(path):
                        for idx in range(self._num_clips):
                            self._path_to_videos.append(line[2])
                            labels.append(label)
                            self._spatial_temporal_idx.append(idx)
                            self._video_meta[clip_idx * self._num_clips + idx] = {}
                        clip_idx += 1
        assert (len(self._path_to_videos) > 0), "Failed to load LVU split {}".format(self.mode)
        self._labels = labels
        logger.info("Constructing LVU dataset (size: {})".format(len(self._path_to_videos)))

    def get_video_path(self, index):
        return self.root+f'videos/{self._path_to_videos[index]}.mp4'
