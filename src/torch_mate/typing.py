import torch
from typing import Tuple, Union

TupleOfTensors = Tuple[torch.Tensor, ...]
NestedTupleOfTensors = Union['NestedTupleOfTensors', TupleOfTensors]
