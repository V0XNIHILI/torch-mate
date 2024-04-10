from typing import Dict, Optional
from copy import deepcopy

from lightning import Trainer

import torch_mate
from torch_mate.utils import get_class
from torch_mate.lightning.utils import build_trainer_kwargs


def configure_trainer(cfg: Dict, **kwargs):
    trainer_cfg = build_trainer_kwargs(cfg)

    if kwargs != {}:
        trainer_cfg.update(kwargs)

    return Trainer(**trainer_cfg)


def configure_data(cfg: Dict, del_dataset_module_kwargs: bool = True, **kwargs):
    data_class = get_class(torch_mate.lightning.datasets, cfg["dataset"]["name"])

    all_kwargs = {}

    if "kwargs" in cfg["dataset"]:
        all_kwargs.update(cfg["dataset"]["kwargs"])

    all_kwargs.update(kwargs)

    if all_kwargs != {}:
        if del_dataset_module_kwargs and "kwargs" in cfg["dataset"]:
            cfg["dataset"].pop("kwargs", None)

        data = data_class(cfg, **all_kwargs)
    else:
        data = data_class(cfg)

    return data


def configure_model(cfg: Dict, **kwargs):
    model_class = get_class(torch_mate.lightning.lm, cfg["learner"]["name"])

    all_kwargs = {}

    if "kwargs" in cfg["learner"]:
        all_kwargs.update(cfg["learner"]["kwargs"])

    all_kwargs.update(kwargs)

    if all_kwargs != {}:
        model = model_class(cfg, **all_kwargs)
    else:
        model = model_class(cfg)

    return model


def configure_model_data(cfg: Dict, model_kwargs: Optional[Dict], data_kwargs: Optional[Dict], del_dataset_module_kwargs: bool = True):
    cfg = deepcopy(cfg)

    data = configure_data(cfg, del_dataset_module_kwargs, **data_kwargs)
    model = configure_model(cfg, **model_kwargs)

    return model, data


def configure_stack(cfg: Dict, trainer_kwargs: Optional[Dict] = None, model_kwargs: Optional[Dict] = None, data_kwargs: Optional[Dict] = None, del_dataset_module_kwargs: bool = True):
    cfg = deepcopy(cfg)

    trainer = configure_trainer(cfg, **(trainer_kwargs if trainer_kwargs else {}))
    # Instantiate data module before model module as the the dataset
    # module kwargs might be deleted from the config dictionary
    data = configure_data(cfg, del_dataset_module_kwargs, **(data_kwargs if data_kwargs else {}))
    model = configure_model(cfg, **(model_kwargs if model_kwargs else {}))

    return trainer, model, data
