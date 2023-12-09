from typing import List, NamedTuple, Dict, Union

from dataclasses import dataclass

import torchvision.transforms as transforms
import torchvision.transforms as transforms
import torch.nn as nn

from torch_mate.utils import get_class

BuiltTransform = Union[transforms.Compose, None]
StateTransform = Union[BuiltTransform, callable]


def build_transform(augmentations: List[Dict]):
    if len(augmentations) == 0:
        return None

    return transforms.Compose(
        [get_class(transforms, aug["name"])(**(aug.get("cfg", {}))) for aug in augmentations]  
    )


def create_state_transforms(task_stage_cfg: Dict, common_pre_transforms: BuiltTransform, common_post_transforms: BuiltTransform) -> StateTransform:
    stage_transforms = []

    if common_pre_transforms:
        stage_transforms.append(common_pre_transforms)

    if task_stage_cfg.get("transforms", None):
        stage_transforms.append(build_transform(task_stage_cfg["transforms"]))

    if common_post_transforms:
        stage_transforms.append(common_post_transforms)
    
    if len(stage_transforms) == 0:
        return None

    if len(stage_transforms) == 1:
        return stage_transforms[0]

    return transforms.Compose(stage_transforms)


def build_data_loader_kwargs(task_stage_cfg: Dict, data_loaders_cfg: Dict, stage: str) -> Dict:
    kwargs = data_loaders_cfg.get("default", {})

    if stage in data_loaders_cfg:
        for (key, value) in data_loaders_cfg[stage].items():
            kwargs[key] = value

    # Only allow batch size and shuffle to pass through for now
    ALLOWED_KWARGS = ['batch_size', 'shuffle']

    for key in ALLOWED_KWARGS:
        if key in task_stage_cfg:
            kwargs[key] = task_stage_cfg[key]
    
    return kwargs

