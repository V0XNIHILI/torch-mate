from typing import Dict

import lightning as L

from torch.utils.data import DataLoader

from torch_mate.lightning.utils import build_data_loader_kwargs, create_stage_transforms, build_transform
from torch_mate.data.utils import Transformed, PreLoaded

STAGES = ['train', 'val', 'test', 'predict']
MOMENTS = ["pre", "post"]


class ConfigurableLightningDataModule(L.LightningDataModule):

    def __init__(self, cfg: Dict):
        """Lightweight wrapper around PyTorch Lightning LightningDataModule that adds support for configuration via a dictionary.

        Overall, compared to the PyTorch Lightning LightningModule, the following two attributes are added:
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

        self.save_hyperparameters(self.configure_configuration(cfg))

        self.common_pre_transforms, self.common_post_transforms = [self.get_common_transform(m) for m in MOMENTS]
        self.common_pre_target_transforms, self.common_post_target_transforms = [self.get_common_target_transform(m) for m in MOMENTS]
        self.pre_transfer_batch_transform, self.post_transfer_batch_transform = [self.get_batch_transform(m) for m in MOMENTS]

    def configure_configuration(self, cfg: Dict):
        return cfg

    def get_common_transform(self, moment: str):
        return build_transform(self.hparams.dataset.get("transforms", {}).get(moment, []))
    
    def get_common_target_transform(self, moment: str):
        return build_transform(self.hparams.dataset.get("target_transforms", {}).get(moment, []))
    
    def get_common_batch_transform(self, moment: str):
        return build_transform(self.hparams.dataset.get("batch_transforms", {}).get(moment, []))

    def get_transform(self, stage: str):
        if "transforms" in self.hparams.dataset:
            return create_stage_transforms(
                self.hparams.dataset["transforms"].get(stage, {}),
                self.common_pre_transforms, 
                self.common_post_transforms
            )
        
        return None
    
    def get_target_transform(self, stage: str):
        if "target_transforms" in self.hparams.dataset:
            return create_stage_transforms(
                self.hparams.dataset["target_transforms"].get(stage, {}),
                self.common_pre_target_transforms, 
                self.common_post_target_transforms
            )
        
        return None
    
    def get_batch_transform(self, moment: str):
        if "batch_transforms" in self.hparams.dataset:
            return build_transform(
                self.hparams.dataset["batch_transforms"].get(moment, [])
            )
        
        return None
    
    def get_dataloader_kwargs(self, stage: str):
        data_loaders_cfg = self.hparams.get("data_loaders", {})
        
        return build_data_loader_kwargs(
            data_loaders_cfg,
            stage
        )

    def get_dataset(self, phase: str):
        raise NotImplementedError
    
    def get_dataset_for_dataloader(self, phase: str):
        dataset = self.get_dataset(phase)

        # API for this entry in the configuration dict is up for change imo.
        if "pre_load" in self.hparams.dataset.get("extra", {}):
            if phase in self.hparams.dataset["extra"]["pre_load"]:
                dataset = PreLoaded(dataset)

        transform = self.get_transform(phase)
        target_transform = self.get_target_transform(phase)

        if transform is None and target_transform is None:
            return dataset
        
        return Transformed(dataset, self.get_transform(phase), self.get_target_transform(phase))

    def train_dataloader(self):
        return DataLoader(self.get_dataset_for_dataloader('train'), **self.get_dataloader_kwargs('train'))
    
    def val_dataloader(self):
        return DataLoader(self.get_dataset_for_dataloader('val'), **self.get_dataloader_kwargs('val'))
    
    def test_dataloader(self):
        return DataLoader(self.get_dataset_for_dataloader('test'), **self.get_dataloader_kwargs('test'))
    
    def predict_dataloader(self):
        return DataLoader(self.get_dataset_for_dataloader('predict'), **self.get_dataloader_kwargs('predict'))
    
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if self.pre_transfer_batch_transform is not None:
            return self.pre_transfer_batch_transform(batch)
        
        return batch
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        if self.post_transfer_batch_transform is not None:
            return self.post_transfer_batch_transform(batch)
        
        return batch
