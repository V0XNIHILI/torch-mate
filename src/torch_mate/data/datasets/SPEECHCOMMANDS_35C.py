from torch import Tensor
import torch
from torchaudio.datasets import SPEECHCOMMANDS

from typing import Union, Optional, Callable, Any, Tuple
from pathlib import Path

class SPEECHCOMMANDS_35C(SPEECHCOMMANDS):
    def __init__(self,
                root: Union[str, Path],
                download: bool = False,
                subset: Optional[str] = None,
                return_int_targets: bool = True,
                transform: Union[Callable, None] = None,
                target_transform: Optional[Callable[Union[Tuple[int, str, str, int], int], Any]] = None) -> None:
        """Google Speech Commands V2 dataset with 35 classes. For the 12-class variant, use:
        https://github.com/KinWaiCheuk/AudioLoader/blob/master/AudioLoader/speech/speechcommands.py

        Args:
            root (Union[str, Path]): Root directory where datasets exist or will be saved.
            download (bool, optional): Whether to download the dataset to disk. Defaults to False.
            subset (Optional[str], optional): Which subset of the dataset to use. Can be either "training"/None, "validation" or "testing". Defaults to None.
            return_int_targets (bool, optional): Whether to return integer targets. Defaults to True.
            transform (Union[Callable, None], optional): Transform to apply to data. Defaults to None.
            target_transform (Optional[Callable[[int, str, str, int], Any]], optional): Transform to apply to targets. Defaults to None.
        """

        super().__init__(root=root, url="speech_commands_v0.02", download=download, subset=subset)

        self.return_int_targets = return_int_targets

        self.transform = transform
        self.target_transform = target_transform

        if self.return_int_targets:
            all_metadata = [self.dataset.get_metadata(n) for n in range(len(self))]
            self.labels = sorted(list(set([m[2] for m in all_metadata])))

    def _label_to_index(self, word: str) -> int:
        return torch.tensor(self.labels.index(word))

    def __getitem__(self, n: int):
        item = super().__getitem__(n)

        waveform, sample_rate, label, speaker_id, utterance_number = item

        if self.transform:
            waveform = self.transform(waveform)

        if self.return_int_targets:
            label = torch.tensor(self.labels.index(word))
        else:
            label = (sample_rate, label, speaker_id, utterance_number)

        if self.target_transform:
            label = self.target_transform(label)

        return waveform, label
