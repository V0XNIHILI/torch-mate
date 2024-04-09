from typing import List, Dict, Union, Optional
from copy import deepcopy

import torchvision.transforms as transforms
import torchvision.transforms as transforms

from lightning import Trainer

from torch_mate.utils import get_class
from torch_mate.lightning import build_trainer_kwargs


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


def build_data_loader_kwargs(data_loaders_cfg: Dict, stage: str) -> Dict:
    # Need to copy, else data from a stage will leak into the default dict,
    # and this data will leak into other stages as the kwargs are built.
    kwargs = data_loaders_cfg.get("default", {}).copy()

    if stage in data_loaders_cfg:
        for (key, value) in data_loaders_cfg[stage].items():
            kwargs[key] = value

    return kwargs


def get_stack(cfg: Dict, trainer_kwargs: Optional[Dict], omit_dataset_module_cfg: bool = False):
    trainer_cfg = build_trainer_kwargs(cfg)

    if trainer_kwargs is not None:
        trainer_cfg.update(trainer_kwargs)

    trainer = Trainer(
       **trainer_cfg
    )

    if "name" in cfg["dataset"] and "module" in cfg["dataset"]:
        raise ValueError("Cannot have both 'name' and 'module' in the dataset config when using get_stack(). Please only use 'module'.")

    data_class = get_class(torch_mate.lightning, cfg["dataset"]["module"]["name"])

    if "cfg" in cfg["dataset"]["module"]:
        data = data_class(cfg, **cfg["dataset"]["module"]["cfg"])
    else:
        data = data_class(cfg, )

    cfg = deepcopy(cfg)

    if omit_dataset_module_cfg:
        cfg["dataset"]["module"].pop("cfg", None)

    if "name" in cfg["learner"] and "module" in cfg["learner"]:
        raise ValueError("Cannot have both 'name' and 'module' in the learner config when using get_stack(). Please only use 'module'.")

    model_class = get_class(torch_mate.lightning, cfg["learner"]["module"]["name"])

    if "cfg" in cfg["learner"]["module"]:
        model = model_class(cfg, **cfg["learner"]["module"]["cfg"])
    else:
        model = model_class(cfg)

    return trainer, model, data
