from typing import List, NamedTuple, Dict, Union

from dataclasses import dataclass

import torchvision.transforms as transforms
import torchvision.transforms as transforms
import torch.nn as nn

from dotmap import DotMap

from torch_mate.utils import get_class

PossibleTransform = Union[transforms.Compose, nn.Identity, None]

@dataclass
class DataAugmentation(NamedTuple):
    name: str
    cfg: Dict


def build_transform(augmentations: List[DataAugmentation]):
    return transforms.Compose(
        [get_class(transforms, aug.name)(**(aug.cfg if aug.cfg else {})) for aug in augmentations]  
    )


def create_state_transforms(task_stage_cfg: DotMap, common_pre_transforms: PossibleTransform, common_post_transforms: PossibleTransform):
    stage_transforms = []

    if common_pre_transforms:
        stage_transforms.append(common_pre_transforms)

    if task_stage_cfg and task_stage_cfg.transforms:
        stage_transforms.append(build_transform(task_stage_cfg.transforms))

    if common_post_transforms:
        stage_transforms.append(common_post_transforms)
    
    if len(stage_transforms) == 0:
        return None

    if len(stage_transforms) == 1:
        return stage_transforms[0]

    return transforms.Compose(stage_transforms)


def build_data_loader_kwargs(task_stage_cfg: DotMap, data_loaders_cfg: DotMap, stage: str):
    data_loaders_cfg_dict = data_loaders_cfg.toDict()

    kwargs = data_loaders_cfg_dict['default'] if 'default' in data_loaders_cfg else {}

    if stage in data_loaders_cfg_dict:
        for (key, value) in data_loaders_cfg_dict[stage].items():
            kwargs[key] = value

    if task_stage_cfg:
        task_stage_cfg_dict = task_stage_cfg.toDict()

        # Only allow batch size and shuffle to pass through for now
        ALLOWED_KWARGS = ['batch_size', 'shuffle']

        for key in ALLOWED_KWARGS:
            if key in task_stage_cfg_dict:
                kwargs[key] = task_stage_cfg_dict[key]
    
    return kwargs
