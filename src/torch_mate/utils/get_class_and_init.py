from typing import Dict
from copy import deepcopy

import inspect

from torch_mate.utils import get_class


def get_class_and_init(base_module, name_and_config: Dict, *args):
    name = name_and_config["name"]
    # Need to deepcopy, else the original configuration dictionary
    # will contain constructed classes instead of their configuration.
    kwargs = deepcopy(name_and_config.get("cfg", None))

    retrieved_class = name if inspect.isclass(name) else get_class(base_module, name)

    if kwargs is not None:
        for key, value in kwargs.items():
            if isinstance(value, dict) and "name" in value:
                kwargs[key] = get_class_and_init(base_module, value)
            elif isinstance(value, list):
                kwargs[key] = [(get_class_and_init(base_module, val)  if isinstance(val, dict) and "name" in val else val) for val in value]

        if args:
            return retrieved_class(*args, **kwargs)

        return retrieved_class(**kwargs)
    
    if args:
        return retrieved_class(*args)

    return retrieved_class()
