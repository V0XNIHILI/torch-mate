from typing import Dict

import torch.nn as nn

from torch_mate.utils import get_class_and_init


def get_modules(base_module, config: Dict):
    if type(config) is tuple:
        raise ValueError("A tuple of modules is not supported. Please use a list or a dict of modules.")

    if type(config) is list:
        return nn.ModuleList([get_class_and_init(base_module, config_entry) for config_entry in config])
    
    # "name" is a reserved key for the model name in case of a single model
    if type(config) is dict and "name" not in config:
        keys = config.keys()
        values = config.values()

        return nn.ModuleDict({key: get_class_and_init(base_module, value) for key, value in zip(keys, values)})

    return get_class_and_init(base_module, config)
