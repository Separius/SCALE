#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import numpy as np
import torch.utils.data

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr
from slowfast.datasets.decoder import decode
import slowfast.datasets.transform as transform
import slowfast.datasets.utils as utils
from slowfast.datasets.video_container import get_video_container

logger = logging.get_logger(__name__)


# SSL_COLOR_HUE:: pBYOL is 0.15, DINO and friends are 0.1
# SSL_COLOR_BRI_CON_SAT: pBYOL is 0.6 x 3, MoCoV2 is 0.4 x 3,  DINO and friends are 0.4, 0.4, 0.2
# MAE(sc) is .5, 1; pBYOL(sc) is .2, .766; DINO(sc) is .4, 1; DINO(mc) is .05, .25, 1; iBOT(mc) is .32; MSN(mc) is .3
# TRAIN_JITTER_SCALES_RELATIVE: [[96, [0.05, 0.32]], [224, [0.32, 1.0]]]
class Kinetics(torch.utils.data.Dataset):
    def __init__(self, train_crop_size=(224, 224), train_jitter_scales_relative=None, mode='train', num_retries=8,
                 num_test_views=10, num_spatial_views=3, test_crop_size=224, multi_thread_decode=False,
                 data_location='/data/cvg/datasets/Videos/kinetics-dataset/k400/', num_frames=16, sampling_rate=4,
                 ssl_color_hue=0.1, ssl_color_bri_con_sat=(0.4, 0.4, 0.2)):
        assert mode in ["train", "val", "test"], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.p_convert_gray = 0.2
        self._video_meta = {}
        self.data_location = data_location
        self._num_retries = num_retries
        self.num_test_views = num_test_views
        self.num_spatial_views = num_spatial_views
        self.train_jitter_scales = (256, 320)
        self.train_crop_size = train_crop_size
        self.test_crop_size = test_crop_size
        self.multi_thread_decode = multi_thread_decode
        self.ssl_color_hue = ssl_color_hue
        self.ssl_color_bri_con_sat = ssl_color_bri_con_sat
        if not isinstance(num_frames, list):
            num_frames = [num_frames] * len(train_crop_size)
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = num_test_views * num_spatial_views
        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        self.aug = self.mode == "train"
        if train_jitter_scales_relative is None:
            train_jitter_scales_relative = [[96, [0.05, 0.32]], [224, [0.32, 1.0]]]
        self.res_to_scale = {k: v for k, v in train_jitter_scales_relative}

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(self.data_location, "{}_meta.csv".format(self.mode))
        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0
        with pathmgr.open(path_to_file, "r") as f:
            for clip_idx, line in enumerate(f):
                path, _, label = line.strip().split(',')
                path = path.strip()
                label = label.strip().strip('"')
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(self.data_location, f'{self.mode}_resized320/{path}.mp4'))
                    self._labels.append(label)
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert len(self._path_to_videos) > 0, "Failed to load Kinetics from {}".format(path_to_file)
        self.labels_dict = {k: i for i, k in enumerate(sorted(set(self._labels)))}
        self._labels = [self.labels_dict[label] for label in self._labels]
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {} ".format(len(self._path_to_videos), path_to_file))

    def __getitem__(self, index):
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.train_jitter_scales[0]
            max_scale = self.train_jitter_scales[1]
            crop_size = self.train_crop_size
        else:  # test
            temporal_sample_index = self._spatial_temporal_idx[index] // self.num_spatial_views
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                    self._spatial_temporal_idx[index] % self.num_spatial_views) if self.num_spatial_views > 1 else 1
            min_scale, max_scale, crop_size = (
                [self.test_crop_size] * 3 if self.num_spatial_views > 1
                else [self.train_jitter_scales[0]] * 2 + [self.test_crop_size]
            )
            crop_size = [crop_size]
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        num_decode = len(self.train_crop_size) if self.mode in ["train"] else 1
        min_scale, max_scale = [min_scale], [max_scale]
        if len(min_scale) < num_decode:
            min_scale += [self.train_jitter_scales[0]] * (num_decode - len(min_scale))
            max_scale += [self.train_jitter_scales[1]] * (num_decode - len(max_scale))
            assert self.mode in ["train", "val"]
        for i_try in range(self._num_retries):
            try:
                video_container = get_video_container(
                    self._path_to_videos[index], self.multi_thread_decode, 'torchvision')
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(self._path_to_videos[index], e))
                if self.mode not in ["test"]:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue  # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try))
                if self.mode not in ["test"] and i_try > self._num_retries // 8:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            num_frames = self.num_frames if self.mode == 'train' else [max(self.num_frames)]
            sampling_rate = [self.sampling_rate]
            if len(num_frames) < num_decode:
                num_frames.extend([num_frames[-1] for _ in range(num_decode - len(num_frames))])
            if len(sampling_rate) < num_decode:
                # base case where keys have same frame-rate as query
                sampling_rate.extend([sampling_rate[-1] for _ in range(num_decode - len(sampling_rate))])
            num_frames = num_frames[:num_decode]
            sampling_rate = sampling_rate[:num_decode]

            if self.mode in ["train"]:
                assert len(min_scale) == len(max_scale) == len(crop_size) == num_decode

            target_fps = 30
            # Decode video. Meta info is used to perform selective decoding.
            max_spatial_scale = min_scale[0] if all(x == min_scale[0] for x in min_scale) else 0
            frames_decoded, time_idx_decoded, tdiff = decode(
                video_container, sampling_rate, num_frames, temporal_sample_index, self.num_test_views,
                video_meta=self._video_meta[index] if len(self._video_meta) < 5e6 else {}, target_fps=target_fps,
                backend='torchvision', use_offset=False, max_spatial_scale=max_spatial_scale, time_diff_prob=0.0)

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames_decoded is None or None in frames_decoded:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(index, self._path_to_videos[index], i_try))
                if self.mode not in ["test"] and (i_try % (self._num_retries // 8)) == 0:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            num_aug = 1
            num_out = num_aug * num_decode
            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            label = self._labels[index]
            if self.mode == 'train':
                pos_info = np.zeros((num_out, 7), dtype=np.float32)

            for i in range(num_decode):
                for _ in range(num_aug):
                    idx += 1
                    f_out[idx] = frames_decoded[i].clone().float() / 255.0
                    time_idx_out[idx] = time_idx_decoded[i, :]
                    if self.mode in ["train"]:
                        f_out[idx] = transform.color_jitter_video_ssl(
                            f_out[idx], bri_con_sat=self.ssl_color_bri_con_sat,
                            hue=self.ssl_color_hue, p_convert_gray=self.p_convert_gray, moco_v2_aug=True)
                    # Perform color normalization.
                    f_out[idx] = utils.tensor_normalize(f_out[idx], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    # T H W C -> C T H W.
                    f_out[idx] = f_out[idx].permute(3, 0, 1, 2)
                    o = utils.spatial_sampling(
                        f_out[idx], spatial_idx=spatial_sample_index, min_scale=min_scale[i], max_scale=max_scale[i],
                        crop_size=crop_size[i], random_horizontal_flip=self.mode != 'train',
                        inverse_uniform_sampling=False, aspect_ratio=[0.75, 1.3333] if self.mode == 'train' else None,
                        scale=self.res_to_scale[crop_size[i]] if self.mode == 'train' else None,
                        motion_shift=False, detailed=self.mode == 'train')
                    if self.mode == 'train':
                        f_out[idx], pos_info[idx, 0], pos_info[idx, 1], pos_info[idx, 2], pos_info[idx, 3], pos_info[
                            idx, 4] = o
                        pos_info[idx, 5], pos_info[idx, 6] = time_idx_decoded[i, 0], time_idx_decoded[i, 1]
                    else:
                        f_out[idx] = o
            if self.mode == 'train':
                return f_out, pos_info, label, index  # List[Tensor] (len=N), array(Nx7), scalar, scalar
            return f_out[0], label, index  # Tensor, scalar, scalar
        else:
            logger.warning(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
