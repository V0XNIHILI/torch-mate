# `torch_mate.lightning`

## Basic idea

My goal here is to achieve zero-code, pure from-configuration-only training of PyTorch models using PyTorch Lightning. This is achieved by using a configuration dictionary that specifies the model, the dataset, the data loaders, etc. The configuration is then used to build all required objects.

## Example usage

### Imports

```python
import yaml

from dotmap import DotMap

from lightning.pytorch.loggers import WandbLogger

from torch_mate.lightning import configure_stack
```

### Define the configuration

Note that I use `DotMap` here to define the configuration, but you can use any other dictionary-like object.

```python
cfg = DotMap()

# Specify the learner and its configuration
# Without having any dots in the name, the import will be done from
# the `torch_mate.lightning.lm` module
cfg.learner.name = "SupervisedLearner"
cfg.learner.cfg = DotMap({
    # Indicate whether classification accuracy should be computed
    "classification": True,
    # Optionally specify for which ks the top-k accuracy should be computed
    # "topk": [1, 5],
})

# Select the criterion and its configuration
cfg.criterion.name = 'CrossEntropyLoss'
# Optionally specify the configuration for the criterion:
# cfg.criterion.cfg = DotMap({
#     'reduction': 'mean'
# })

# Optionally specify the learning rate scheduler and its configuration
cfg.lr_scheduler = DotMap({
    "name": "StepLR",
    "cfg": DotMap({
        "step_size": 2,
        "verbose": True,
    }),
})

# Specify the model and its configuration
cfg.model.name = 'torch_mate.models.LeNet5BNMaxPool'
cfg.model.cfg = DotMap({
    'num_classes': 10
})

# Specify the optimizer and its configuration
cfg.optimizer.name = 'Adam'
cfg.optimizer.cfg = DotMap({"lr": 0.007})

# Specify the training configuration (passed directly to the PyTorch
# Lightning Trainer). The `early_stopping` configuration is optional
# and will be used to configure the early stopping callback.
cfg.training = DotMap({
    'max_epochs': 100,
    'early_stopping': DotMap({
        'monitor': 'val/loss',
        'patience': 10,
        'mode': 'min'
    }),
})

# Set the seed for reproducibility
cfg.seed = 4223747124

# Specify the dataset and its configuration.
# Without having any dots in the name, the import will be done from
# the `torch_mate.lightning.datasets` module
cfg.dataset.name = 'MNISTData'
cfg.dataset.kwargs = DotMap({
    "root": './data',
})

# Specify the transforms and their configuration
# Note that you can specify .pre (common pre-transform), .train
# .val/.test/.predict (specific transforms for each split) and
# .post (common post-transform). The complete transforms will
# then be built automatically. The same goes for target_transforms
# via: cfg.dataset.target_transforms
cfg.dataset.transforms.pre = [
    DotMap({'name': 'ToTensor'}),
    DotMap({'name': 'Resize', 'cfg': {'size': (28, 28)}}),
]

# Optionally, specify a pre-device and post-device transfer
# batch transform via: cfg.dataset.batch_transforms.pre and
# cfg.dataset.batch_transforms.post in the same manner
# as for the other transforms.

# Specify the data loaders and their configuration (where default
# is the fallback configuration for all data loaders)
cfg.data_loaders = DotMap({
    'default': DotMap({
        'num_workers': 4,
        'prefetch_factor': 16,
        'persistent_workers': True,
        'batch_size': 256,
    }),
    'train': DotMap({
        'batch_size': 512
    })
})

cfg = cfg.toDict()

print(yaml.dump(cfg, default_flow_style=False))
```

### Get the model, data and trainer

```python
trainer, model, data = configure_stack(cfg,
    # Specify all keyworded arguments that are not part of the 
    # `cfg.training` dictionary for the PyTorch Lightning Trainer
    {
        "enable_progress_bar": True,
        "accelerator": "mps",
        "devices": 1,
        "logger": WandbLogger(project="test_wandb_lightning")
    }
)
```

### Train the model

```python
trainer.fit(model, data)
```