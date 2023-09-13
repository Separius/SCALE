#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import av
import torch
from decord import VideoReader
from decord import cpu, gpu


def get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav", max_scale=-1):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        container = av.open(path_to_vid)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        return container
    elif backend.startswith('decord'):
        if backend == 'decord_cpu':
            ctx = cpu(0)
        else:
            ctx = gpu(torch.cuda.current_device())
        if max_scale != -1:
            height = max_scale
            width = 0
        else:
            height = width = -1
        with open(path_to_vid, 'rb') as fp:
            container = VideoReader(fp, ctx=ctx, width=width, height=height)
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))
