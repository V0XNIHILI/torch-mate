from typing import Dict

from torch_mate.utils import get_class

def get_class_and_init(base_module, name_and_config: Dict):
    name = name_and_config["name"]
    kwargs = name_and_config.get("cfg", None)

    retrieved_class = get_class(base_module, name)

    if kwargs is not None:
        return retrieved_class(**kwargs)
    
    return retrieved_class()
