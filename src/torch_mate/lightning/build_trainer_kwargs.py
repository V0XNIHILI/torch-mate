from typing import Dict

import copy

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def build_trainer_kwargs(cfg: Dict) -> Dict:
    """Lightweight function to build the kwargs for a PyTorch Lightning Trainer from a configuration dictionary.

    Args:
        cfg (Dict): configuration dictionary

    Returns:
        Dict: kwargs for a PyTorch Lightning Trainer
    """

    if "training" in cfg:
        cfg_dict = copy.deepcopy(cfg["training"])

        # Add support for early stopping
        if 'early_stopping' in cfg_dict:
            cfg_dict['callbacks'] = [EarlyStopping(**cfg_dict['early_stopping'])]
            del cfg_dict['early_stopping']

        return cfg_dict

    return {}
