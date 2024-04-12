from typing import List, Dict, Union
import copy

import torchvision.transforms as transforms

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from torch_mate.utils import get_class_and_init


BuiltTransform = Union[transforms.Compose, None]
StageTransform = Union[BuiltTransform, callable]


def build_transform(augmentations: List[Dict]):
    if len(augmentations) == 0:
        return None

    return transforms.Compose(
        [get_class_and_init(transforms, aug) for aug in augmentations]  
    )


def create_stage_transforms(task_stage_cfg: Dict, common_pre_transforms: BuiltTransform, common_post_transforms: BuiltTransform) -> StageTransform:
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


def build_data_loader_kwargs(data_loaders_cfg: Dict, stage: str) -> Dict:
    # Need to copy, else data from a stage will leak into the default dict,
    # and this data will leak into other stages as the kwargs are built.
    kwargs = data_loaders_cfg.get("default", {}).copy()

    if stage in data_loaders_cfg:
        for (key, value) in data_loaders_cfg[stage].items():
            kwargs[key] = value

    return kwargs


def build_trainer_kwargs(cfg: Dict) -> Dict:
    """Lightweight function to build the kwargs for a PyTorch Lightning Trainer from a configuration dictionary.

    Args:
        cfg (Dict): configuration dictionary

    Returns:
        Dict: kwargs for a PyTorch Lightning Trainer
    """

    if "training" in cfg:
        cfg_dict = copy.deepcopy(cfg["training"])

        # Add support for early stopping
        if 'early_stopping' in cfg_dict:
            cfg_dict['callbacks'] = [EarlyStopping(**cfg_dict['early_stopping'])]
            del cfg_dict['early_stopping']

        return cfg_dict

    return {}
