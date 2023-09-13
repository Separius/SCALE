import torch
import numpy as np
from torch.utils.data import Dataset

import slowfast.datasets.utils_orig as utils
import slowfast.datasets.decoder_orig as decoder
import slowfast.datasets.transform_orig as transform
import slowfast.datasets.video_container_orig as container


class VideoDataset(Dataset):
    def __init__(self, cfg, mode, num_retries=1):
        assert mode in ["train", "val"], "Split '{}' not supported".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = cfg.DATA.COLOR_RND_GRAYSCALE
        self._video_meta = {}
        self._num_retries = num_retries
        if self.mode == "train":
            self._num_clips = 1
        elif self.mode == "val":
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        self.indices = None
        self._construct_loader()

    def get_video_path(self, index):
        raise NotImplementedError

    def _construct_loader(self):
        raise NotImplementedError

    def set_indices(self, indices):
        self.indices = indices

    @staticmethod
    def normalize_pos_info(pos_info):  # flipped, h0, w0, h1, w1, t0, t1
        min_h = np.min(pos_info[:, 1])
        min_w = np.min(pos_info[:, 2])
        max_h = np.max(pos_info[:, 3])
        max_w = np.max(pos_info[:, 4])
        min_t = np.min(pos_info[:, 5])
        max_t = np.max(pos_info[:, 6])
        pos_info[:, 1] -= min_h
        pos_info[:, 1] /= max_h - min_h
        pos_info[:, 3] -= min_h
        pos_info[:, 3] /= max_h - min_h
        pos_info[:, 2] -= min_w
        pos_info[:, 2] /= max_w - min_w
        pos_info[:, 4] -= min_w
        pos_info[:, 4] /= max_w - min_w
        pos_info[:, 5] -= min_t
        pos_info[:, 5] /= max_t - min_t
        pos_info[:, 6] -= min_t
        pos_info[:, 6] /= max_t - min_t
        return pos_info

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]
        if self.mode == 'train':
            x, label, index, pos = self.get_item(index)
        else:
            x = []
            p = []
            for idx in range(self._num_clips):
                o = self.get_item(index * self._num_clips + idx)  # Tensor(3, frames, h, w), label, index, Tensor(1, 7)
                x.append(o[0])
                p.append(o[3])
            x = torch.stack(x, dim=0)
            label = o[1]
            pos = torch.cat(p, dim=0)
        return x, label, index, torch.from_numpy(self.normalize_pos_info(pos.numpy()))

    def get_item(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode == 'train':
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            temporal_sample_index = self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = ((self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                                    if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1)
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                     + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        num_decode = self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL if self.mode == "train" else 1
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]
        if len(min_scale) < num_decode:
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (num_decode - len(min_scale))
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (num_decode - len(max_scale))
            crop_size += [self.cfg.DATA.TRAIN_CROP_SIZE] * (num_decode - len(crop_size))
            assert self.mode == "train"
        # Try to decode and sample a clip from a video.
        for i_try in range(self._num_retries):
            path = self.get_video_path(index)
            video_container = container.get_video_container(
                path, self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE, self.cfg.DATA.DECODING_BACKEND)
            if video_container is None:
                raise ValueError(f'video index={index}')
            num_frames = [self.cfg.DATA.NUM_FRAMES]
            sampling_rate = [self.cfg.DATA.SAMPLING_RATE]
            if len(num_frames) < num_decode:
                num_frames.extend([num_frames[-1] for _ in range(num_decode - len(num_frames))])
                # base case where keys have same frame-rate as query
                sampling_rate.extend([sampling_rate[-1] for _ in range(num_decode - len(sampling_rate))])
            elif len(num_frames) > num_decode:
                num_frames = num_frames[:num_decode]
                sampling_rate = sampling_rate[:num_decode]
            if self.mode == "train":
                assert (len(min_scale) == len(max_scale) == len(crop_size) == num_decode)
            target_fps = self.cfg.DATA.TARGET_FPS
            # Decode video. Meta info is used to perform selective decoding.
            frames, time_idx, tdiff = decoder.decode(
                video_container,
                sampling_rate,
                num_frames,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=target_fps,
                backend=self.cfg.DATA.DECODING_BACKEND,
                use_offset=False,
                max_spatial_scale=min_scale[0] if all(x == min_scale[0] for x in min_scale) else 0,
                time_diff_prob=0.0,
                temporally_rnd_clips=True,
                min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
            )
            frames_decoded = frames
            time_idx_decoded = time_idx

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames_decoded is None or None in frames_decoded:
                raise ValueError(
                    "Failed to decode video idx {} from {}; trial {}".format(index, path, i_try)
                )

            num_out = num_decode
            f_out = [None] * num_out
            idx = -1
            label = self._labels[index]
            pos_info = np.zeros((num_out, 7), dtype=np.float32)

            for i in range(num_decode):
                idx += 1
                f_out[idx] = frames_decoded[i].clone()
                f_out[idx] = f_out[idx].float() / 255.0
                if self.mode == "train" and self.cfg.DATA.SSL_COLOR_JITTER:
                    f_out[idx] = transform.color_jitter_video_ssl(
                        f_out[idx],
                        bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                        hue=self.cfg.DATA.SSL_COLOR_HUE,
                        p_convert_gray=self.p_convert_gray,
                        moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                        gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                        gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                    )
                # Perform color normalization.
                f_out[idx] = utils.tensor_normalize(f_out[idx], self.cfg.DATA.MEAN, self.cfg.DATA.STD)
                # T H W C -> C T H W.
                f_out[idx] = f_out[idx].permute(3, 0, 1, 2)
                scl, asp = self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE, self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE
                relative_scales = None if (self.mode == 'val' or len(scl) == 0) else scl
                relative_aspect = None if (self.mode == 'val' or len(asp) == 0) else asp
                f_out[idx], flipped, h0, w0, h1, w1 = utils.spatial_sampling(
                    f_out[idx],
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale[i],
                    max_scale=max_scale[i],
                    crop_size=crop_size[i],
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    aspect_ratio=relative_aspect,
                    scale=relative_scales,
                    motion_shift=False
                )
                pos_info[idx, 0] = flipped
                pos_info[idx, 1] = h0
                pos_info[idx, 2] = w0
                pos_info[idx, 3] = h1
                pos_info[idx, 4] = w1
                pos_info[idx, 5] = time_idx_decoded[i, 0]  # t0
                pos_info[idx, 6] = time_idx_decoded[i, 1]  # t1
            frames = f_out[0] if num_out == 1 else torch.stack(f_out, dim=0)  # num_decode, 3, frames, h, w
            return frames, label, index, torch.from_numpy(pos_info)
        else:
            raise ValueError(f'index={index}')

    def __len__(self):
        if self.indices is None:
            return len(self._path_to_videos) // self._num_clips
        return len(self.indices)
