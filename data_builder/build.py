# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler


def build_loader(config):
    # config.defrost()
    dataset_train, config.nb_classes = build_dataset(is_train=True, config=config)
    # config.freeze()
    print(f"local rank {config.rank} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.rank} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        drop_last=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        drop_last=False
    )
    
    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.mixup > 0 or config.cutmix> 0. or config.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.mixup, cutmix_alpha=config.cutmix, cutmix_minmax=config.cutmix_minmax,
            prob=config.mixup_prob, switch_prob=config.mixup_switch_prob, mode=config.mixup_mode,
            label_smoothing=config.smoothing, num_classes=config.nb_classes)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.data_set == 'IMNET':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.data_path, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.input_size,
            is_training=True,
            color_jitter=config.color_jitter if config.color_jitter > 0 else None,
            auto_augment=config.aa if config.aa != 'none' else None,
            re_prob=config.reprob,
            re_mode=config.remode,
            re_count=config.recount,
            interpolation=config.train_interpolation,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * config.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(config.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
