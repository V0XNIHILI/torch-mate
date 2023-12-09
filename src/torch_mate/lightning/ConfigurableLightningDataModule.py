from typing import Dict

import lightning as L

from torch.utils.data import DataLoader

from torch_mate.lightning.utils import build_data_loader_kwargs, create_state_transforms, build_transform, BuiltTransform, StateTransform

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

        common_pre_transforms, common_post_transforms = [build_transform(self.hparams.task.get("transforms", {}).get(m, [])) for m in MOMENTS]

        data_loaders_cfg = self.hparams.get("data_loaders", {})
        
        # TODO: add support for predict dataloader kwargs
        for stage in STAGES:
            task_stage_cfg = self.hparams.task.get(stage, {})

            setattr(self, f"{stage}_dataloader_kwargs", build_data_loader_kwargs(
                task_stage_cfg,
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

        if self.hparams.task.get("transforms", {}).get("batch", None):
            for m in MOMENTS:
                setattr(self, f"{m}_transfer_batch_transform", build_transform(
                    self.hparams.task["transforms"]["batch"].get(m, [])
                ))
        else:
            self.pre_transfer_batch_transform = None
            self.post_transfer_batch_transform = None

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
        if self.post_transfer_batch_transform is not None:
            batch = self.reshape_batch_during_transfer(batch, dataloader_idx, "before")
            batch = self.post_transfer_batch_transform(batch)
        
        return batch
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        if self.pre_transfer_batch_transform is not None:
            batch = self.reshape_batch_during_transfer(batch, dataloader_idx, "after")
            batch = self.pre_transfer_batch_transform(batch)
        
        return batch
    