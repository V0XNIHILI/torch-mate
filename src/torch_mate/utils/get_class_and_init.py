from typing import Dict

import inspect

from torch_mate.utils import get_class

def get_class_and_init(base_module, name_and_config: Dict, *args):
    name = name_and_config["name"]
    kwargs = name_and_config.get("cfg", None)

    retrieved_class = name if inspect.isclass(name) else get_class(base_module, name)

    if kwargs is not None:
        if args:
            return retrieved_class(*args, **kwargs)

        return retrieved_class(**kwargs)
    
    if args:
        return retrieved_class(*args)

    return retrieved_class()
