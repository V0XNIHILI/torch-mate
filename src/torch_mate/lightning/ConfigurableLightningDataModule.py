from typing import Dict

import lightning as L

from torch.utils.data import DataLoader

from torch_mate.lightning.utils import build_data_loader_kwargs, create_state_transforms, build_transform, BuiltTransform, StateTransform
from torch_mate.data.utils import Transformed, PreLoaded

STAGES = ['train', 'val', 'test', 'predict']
MOMENTS = ["pre", "post"]


class ConfigurableLightningDataModule(L.LightningDataModule):
    train_dataloader_kwargs: Dict
    val_dataloader_kwargs: Dict
    test_dataloader_kwargs: Dict
    predict_dataloader_kwargs: Dict

    train_transforms: StateTransform
    val_transforms: StateTransform
    test_transforms: StateTransform
    predict_transforms: StateTransform

    train_target_transforms: BuiltTransform
    val_target_transforms: BuiltTransform
    test_target_transforms: BuiltTransform
    predict_target_transforms: BuiltTransform

    pre_transfer_batch_transform: BuiltTransform
    post_transfer_batch_transform: BuiltTransform

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

        self.save_hyperparameters(cfg)

        common_pre_transforms, common_post_transforms = [build_transform(self.hparams.dataset.get("transforms", {}).get(m, [])) for m in MOMENTS]

        data_loaders_cfg = self.hparams.get("data_loaders", {})
        
        if "transforms" in self.hparams.dataset:
            # TODO: add support for predict dataloader kwargs
            for stage in STAGES:
                task_stage_cfg = self.hparams.dataset["transforms"].get(stage, {})

                setattr(self, f"{stage}_dataloader_kwargs", build_data_loader_kwargs(
                    data_loaders_cfg,
                    stage)
                )

                setattr(self, f"{stage}_transforms", create_state_transforms(
                    task_stage_cfg,
                    common_pre_transforms, 
                    common_post_transforms
                ))

                setattr(self, f"{stage}_target_transforms", build_transform(
                    task_stage_cfg.get("target_transforms", [])
                ))

        common_pre_batch_transforms, common_post_batch_transforms = [build_transform(self.hparams.dataset.get("batch_transforms", {}).get(m, [])) for m in MOMENTS]

        if "batch_transforms" in self.hparams.dataset:
            for m in MOMENTS:
                setattr(self, f"{stage}_transfer_batch_transform", create_state_transforms(
                    self.hparams.dataset["batch_transforms"].get(m, {}),
                    common_pre_batch_transforms, 
                    common_post_batch_transforms
                ))
        else:
            self.pre_transfer_batch_transform = None
            self.post_transfer_batch_transform = None

    def get_dataset(self, phase: str):
        raise NotImplementedError
    
    def get_dataset_for_dataloader(self, phase: str):
        dataset = self.get_dataset(phase)

        # API for this entry in the configuration dict is up for change imo.
        if "pre_load" in self.hparams.dataset.get("extra", {}):
            if phase in self.hparams.dataset["extra"]["pre_load"]:
                dataset = PreLoaded(dataset)

        if getattr(self, f"{phase}_transforms") is None or getattr(self, f"{phase}_target_transforms") is None:
            return dataset
        
        return Transformed(dataset, getattr(self, f"{phase}_transforms"), getattr(self, f"{phase}_target_transforms"))

    def train_dataloader(self):
        return DataLoader(self.get_dataset_for_dataloader('train'), **self.train_dataloader_kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.get_dataset_for_dataloader('val'), **self.val_dataloader_kwargs)
    
    def test_dataloader(self):
        return DataLoader(self.get_dataset_for_dataloader('test'), **self.test_dataloader_kwargs)
    
    def predict_dataloader(self):
        return DataLoader(self.get_dataset_for_dataloader('predict'), **self.predict_dataloader_kwargs)
    
    def reshape_batch_during_transfer(self, batch, dataloader_idx: int, moment: str):
        # Probably remove this function or make it direclty parametrizable via (-1, 1, x) etc.
        return batch
    
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if self.pre_transfer_batch_transform is not None:
            batch = self.reshape_batch_during_transfer(batch, dataloader_idx, "before")
            a = self.pre_transfer_batch_transform(batch[0])
            return a, batch[1]
        
        return batch
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        if self.post_transfer_batch_transform is not None:
            self.post_transfer_batch_transform.transforms[0].to(batch[0].device)
            batch = self.reshape_batch_during_transfer(batch, dataloader_idx, "after")
            a = self.post_transfer_batch_transform(batch[0])
            # Also add a transform here
            # need to add transform that you dont care about logging them
            return a, batch[1]
        
        return batch
