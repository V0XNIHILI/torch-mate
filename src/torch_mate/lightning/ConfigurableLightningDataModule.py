from typing import Union

import lightning as L

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

from dotmap import DotMap

from torch_mate.lightning.build_transform import build_transform

PossibleTransform = Union[transforms.Compose, nn.Identity, None]

def create_state_transforms(task_stage_cfg: DotMap, common_pre_transforms: PossibleTransform, common_post_transforms: PossibleTransform):
    stage_transforms = []

    if common_pre_transforms:
        stage_transforms.append(common_pre_transforms)

    if task_stage_cfg and task_stage_cfg.transforms:
        stage_transforms.append(build_transform(task_stage_cfg.transforms))

    if common_post_transforms:
        stage_transforms.append(common_post_transforms)
    
    if len(stage_transforms) == 0:
        return None

    if len(stage_transforms) == 1:
        return stage_transforms[0]

    return transforms.Compose(stage_transforms)



def build_data_loader_kwargs(task_stage_cfg: DotMap, data_loaders_cfg: DotMap, stage: str):
    data_loaders_cfg_dict = data_loaders_cfg.toDict()

    kwargs = data_loaders_cfg_dict['default'] if 'default' in data_loaders_cfg else {}

    if stage in data_loaders_cfg_dict:
        for (key, value) in data_loaders_cfg_dict[stage].items():
            kwargs[key] = value

    if task_stage_cfg:
        task_stage_cfg_dict = task_stage_cfg.toDict()

        # Only allow batch size and shuffle to pass through for now
        ALLOWED_KWARGS = ['batch_size', 'shuffle']

        for key in ALLOWED_KWARGS:
            if key in task_stage_cfg_dict:
                kwargs[key] = task_stage_cfg_dict[key]
    
    return kwargs


class ConfigurableLightningDataModule(L.LightningDataModule):
    def __init__(self, cfg: DotMap):
        super().__init__()

        self.cfg = cfg

        self.train_dataloader_kwargs = build_data_loader_kwargs(cfg.task.train, cfg.data_loaders, 'train')
        self.val_dataloader_kwargs = build_data_loader_kwargs(cfg.task.val, cfg.data_loaders, 'val')
        self.test_dataloader_kwargs = build_data_loader_kwargs(cfg.task.test, cfg.data_loaders, 'test')

        common_pre_transforms = build_transform(cfg.task.transforms.pre) if cfg.task.transforms and cfg.task.transforms.pre else None
        common_post_transforms = build_transform(cfg.task.transforms.post) if cfg.task.transforms and cfg.task.transforms.post else None

        self.train_transforms = create_state_transforms(cfg.task.train, common_pre_transforms, common_post_transforms)
        self.val_transforms = create_state_transforms(cfg.task.val, common_pre_transforms, common_post_transforms)
        self.test_transforms = create_state_transforms(cfg.task.test, common_pre_transforms, common_post_transforms)

        if cfg.task.transforms and cfg.task.transforms.batch:
            if cfg.task.transforms.batch.pre:
                self.pre_transfer_batch_transform = build_transform(cfg.task.transforms.batch.pre)
            
            if cfg.task.transforms.batch.post:
                self.post_transfer_batch_transform = build_transform(cfg.task.transforms.batch.post)

    def get_dataset(self, phase: str):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.get_dataset('train'), **self.train_dataloader_kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.get_dataset('val'), **self.val_dataloader_kwargs)
    
    def test_dataloader(self):
        return DataLoader(self.get_dataset('test'), **self.test_dataloader_kwargs)
    
    def predict_dataloader(self):
        return DataLoader(self.get_dataset('predict'), **self.test_dataloader_kwargs)
    
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if hasattr(self, "post_transfer_batch_transform"):
            batch = self.post_transfer_batch_transform(batch)
        
        return batch
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        if hasattr(self, "pre_transfer_batch_transform"):
            batch = self.pre_transfer_batch_transform(batch)
        
        return batch
    