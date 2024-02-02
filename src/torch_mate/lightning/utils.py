from typing import List, Dict, Union

import torchvision.transforms as transforms
import torchvision.transforms as transforms

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


# Only allow batch size and shuffle to pass through for now
ALLOWED_KWARGS = ['batch_size', 'shuffle']


def build_data_loader_kwargs(task_stage_cfg: Dict, data_loaders_cfg: Dict, stage: str) -> Dict:
    # Need to copy, else data from a stage will leak into the default dict,
    # and this data will leak into other stages as the kwargs are built.
    kwargs = data_loaders_cfg.get("default", {}).copy()

    if stage in data_loaders_cfg:
        for (key, value) in data_loaders_cfg[stage].items():
            kwargs[key] = value

    for key in ALLOWED_KWARGS:
        if key in kwargs:
            raise ValueError(f"Cannot override {key} from hparams.data_loaders configuration")
        
        if key in task_stage_cfg:
            kwargs[key] = task_stage_cfg[key]
    
    return kwargs
