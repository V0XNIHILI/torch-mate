from typing import List, NamedTuple, Dict

from dataclasses import dataclass

from torch_mate.utils import get_class

import torchvision.transforms as transforms


@dataclass
class DataAugmentation(NamedTuple):
    name: str
    cfg: Dict


def build_transform(augmentations: List[DataAugmentation]):
    return transforms.Compose(
        [get_class(transforms, aug.name)(**(aug.cfg if aug.cfg else {})) for aug in augmentations]  
    )
