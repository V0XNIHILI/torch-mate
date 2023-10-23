# `torch-mate`: flow

## Overview

### Basic idea

```python
from dotmap import DotMap

from torch_mate.flow import main

# Create config
cfg = DotMap()

# Required config fields
cfg.criterion.name = 'CrossEntropyLoss'
cfg.optimizer.name = 'SGD'
cfg.optimizer.cfg = DotMap({"lr": 0.18 , 'momentum': 0.9, "weight_decay": 0.0005})

# Optional
cfg.lr_scheduler.name = 'MultiStepLR'
cfg.lr_scheduler.cfg = DotMap({ "milestones":[100,200,300], "gamma": 0.1})

# Optional
cfg.early_stopping.cfg = DotMap({"patience": 10, "delta": 0.0})

cfg.task.train.n_epochs = 300

# Regular training

model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = ...

main(cfg, model, train_loader, val_loader, device, log=print)
```

### Evaluate on test set after training

```python
main(cfg, model, train_loader, val_loader, test_loader, device,
     test_data_loader=test_data_loader,
     log=print)
```

### Save model!!!!

```python
# Save model every 10 epochs
main(..., save_every=10, save_dir='path/to/save/dir')

# Save model only at the end of training
main(..., save_every=-1, save_dir='path/to/save/dir')

# Never save model
main(..., save_every=0, save_dir='path/to/save/dir')
```

### Custom training function

```python
def custom_train(model, loss, train_data_loader, optimizer, device, batch_transform, extra_loss):
    return {'train/loss': -7.0, 'train/accuracy': 0.0, 'train/time': 0.48}

main(..., custom_train=custom_train)
```

### Custom evaluation function !!!

```python
def custom_evaluate(model, loss, device, batch_transform):
    return {'val/loss': -7.0, 'val/accuracy': 0.0, 'val/time': 0.48}

main(..., val_data_loader=None, custom_evaluate=custom_evaluate)
```

### Various training options

* **Compile model (for PyTorch 2.0)**
    ```python
    main(..., compile=True)
    ```
* **Save metrics to WandB**
    ```python
    with wandb.init(project="project_name", entity="entity_name", config=cfg):
        main(..., log=wandb.log, save_dir=wandb.run.name)
    ```
* **Test only every 10 epochs**
    ```python
    main(..., test_every=10)
    ```
* **Transform every batch**
    ```python
    main(..., batch_transform=lambda x: x + 3.0)
    ```
* **Different logging step key**
    ```python
    main(..., step_key='iteration')
    ```
