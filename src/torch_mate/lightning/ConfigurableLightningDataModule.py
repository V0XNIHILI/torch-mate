from typing import Dict

import lightning as L

from torch.utils.data import DataLoader

from torch_mate.lightning.utils import build_data_loader_kwargs, create_state_transforms, build_transform

class ConfigurableLightningDataModule(L.LightningDataModule):
    def __init__(self, cfg: Dict):
        """Lightweight wrapper around PyTorch Lightning LightningDataModule that adds support for configuration via a dictionary.

        Overall, compared to the PyTorch Lightning LightningModule, the following two attributes are added:
        - `self.reshape_batch_during_transfer(self, batch, dataloader_idx, moment)`: a function that reshapes the batch during transfer to allow for standardized batch transforms
        - `self.get_dataset(self, phase)`: a function that returns the dataset for a given phase

        Based on these, the following methods are automatically implemented:
        - `self.train_dataloader(self)`: calls `DataLoader(self.get_dataset('train'), **self.train_dataloader_kwargs)`
        - `self.val_dataloader(self)`: calls `DataLoader(self.get_dataset('val'), **self.val_dataloader_kwargs)`
        - `self.test_dataloader(self)`: calls `DataLoader(self.get_dataset('test'), **self.test_dataloader_kwargs)`
        - `self.predict_dataloader(self)`: calls `DataLoader(self.get_dataset('predict'), **self.test_dataloader_kwargs)`
        - `self.on_before_batch_transfer(self, batch, dataloader_idx)`: calls `self.reshape_batch_during_transfer(batch, dataloader_idx, "before")` followed by `self.post_transfer_batch_transform(batch)`
        - `self.on_after_batch_transfer(self, batch, dataloader_idx)`: calls `self.reshape_batch_during_transfer(batch, dataloader_idx, "after")` followed by `self.pre_transfer_batch_transform(batch)`

        Args:
            cfg (Dict): configuration dictionary
        """

        super().__init__()

        self.cfg = cfg

        self.save_hyperparameters(cfg.toDict())

        self.train_dataloader_kwargs = build_data_loader_kwargs(cfg.task.train, cfg.data_loaders, 'train')
        self.val_dataloader_kwargs = build_data_loader_kwargs(cfg.task.val, cfg.data_loaders, 'val')
        self.test_dataloader_kwargs = build_data_loader_kwargs(cfg.task.test, cfg.data_loaders, 'test')
        # TODO: add support for predict dataloader kwargs

        common_pre_transforms = build_transform(cfg.task.transforms.pre) if cfg.task.transforms and cfg.task.transforms.pre else None
        common_post_transforms = build_transform(cfg.task.transforms.post) if cfg.task.transforms and cfg.task.transforms.post else None

        self.train_transforms = create_state_transforms(cfg.task.train, common_pre_transforms, common_post_transforms)
        self.val_transforms = create_state_transforms(cfg.task.val, common_pre_transforms, common_post_transforms)
        self.test_transforms = create_state_transforms(cfg.task.test, common_pre_transforms, common_post_transforms)
        # TODO: add support for predict transforms

        self.train_target_transforms = build_transform(cfg.task.train.target_transforms) if cfg.task.train.target_transforms else None
        self.val_target_transforms = build_transform(cfg.task.val.target_transforms) if cfg.task.val.target_transforms else None
        self.test_target_transforms = build_transform(cfg.task.test.target_transforms) if cfg.task.test.target_transforms else None

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
    
    def reshape_batch_during_transfer(self, batch, dataloader_idx: int, moment: str):
        return batch
    
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if hasattr(self, "post_transfer_batch_transform"):
            batch = self.reshape_batch_during_transfer(batch, dataloader_idx, "before")
            batch = self.post_transfer_batch_transform()
        
        return batch
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        if hasattr(self, "pre_transfer_batch_transform"):
            batch = self.reshape_batch_during_transfer(batch, dataloader_idx, "after")
            batch = self.pre_transfer_batch_transform(batch)
        
        return batch
    