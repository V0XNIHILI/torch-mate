from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from dotmap import DotMap

def build_trainer_kwargs(cfg: DotMap):
    if cfg.training:
        cfg_dict = cfg.training.toDict()

        # Add support for early stopping
        if 'early_stopping' in cfg_dict:
            cfg_dict['callbacks'] = [EarlyStopping(**cfg_dict['early_stopping'])]
            del cfg_dict['early_stopping']

        return cfg_dict

    return {}
