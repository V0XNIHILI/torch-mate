import torch
from typing import Tuple, Union, Optional, Callable

TupleOfTensors = Tuple[torch.Tensor, ...]
NestedTupleOfTensors = Union['NestedTupleOfTensors', TupleOfTensors]
OptionalBatchTransform = Optional[Callable[[torch.Tensor], torch.Tensor]]
