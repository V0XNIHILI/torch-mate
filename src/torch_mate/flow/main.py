import os
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from dotmap import DotMap

from early_stopping import EarlyStopping

from torch.utils.data import DataLoader
from torch_mate.utils import calc_accuracy, iterate_to_device
from torch_mate.contexts import evaluating, training

from tqdm import tqdm

OptionalBatchTransform = Optional[Callable[[torch.Tensor], torch.Tensor]]
OptionalExtLoss = Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]


def process_batch(model: nn.Module,
                  criterion: nn.Module,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  batch_transform: OptionalBatchTransform = None,
                  extra_loss: OptionalExtLoss = None):
    if batch_transform:
        X = batch_transform(X)

    output = model(X)
    accuracy = calc_accuracy(output, y)

    error = criterion(output, y) + (extra_loss(X, output) if extra_loss else 0.0)

    preds = torch.argmax(output, dim=1)

    return (error, accuracy), (preds, y)


def train(model: nn.Module,
          criterion: nn.Module,
          train_data_loader: DataLoader,
          opt: torch.optim.Optimizer,
          device: torch.device,
          batch_transform: OptionalBatchTransform = None,
          extra_loss: OptionalExtLoss = None):
    train_error = 0.0
    train_accuracy = 0.0

    with training(model):
        for (X, y) in iterate_to_device(train_data_loader, device, True):
            # Following:
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            opt.zero_grad(set_to_none=True)

            (error, accuracy), _ = process_batch(model, criterion, X, y,
                                                 batch_transform, extra_loss)

            error.backward()
            opt.step()

            train_error += error.detach()
            train_accuracy += accuracy

    return train_error.item() / len(train_data_loader), train_accuracy / len(
        train_data_loader)


def evaluate(model: nn.Module,
             criterion: nn.Module,
             eval_data_loader: DataLoader,
             device: torch.device,
             batch_transform: OptionalBatchTransform = None,
             extra_loss: OptionalExtLoss = None):
    eval_error = 0.0
    eval_accuracies = []

    # Using evaluating(model) or evaluating(maml) here does not make a difference
    with evaluating(model), torch.no_grad():
        for (X, y) in iterate_to_device(eval_data_loader, device, True):
            (error, accuracy), _ = process_batch(model, criterion, X, y,
                                                       batch_transform,
                                                       extra_loss)

            eval_error += error.detach()
            eval_accuracies.append(accuracy)

    eval_accuracies = np.array(eval_accuracies)

    return eval_error.item() / len(eval_data_loader), (np.mean(eval_accuracies), 1.96 * np.std(eval_accuracies) / np.sqrt(len(eval_accuracies)))


def main(
    cfg: DotMap,
    model: nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: Optional[DataLoader],
    device: torch.device,
    test_data_loader: Optional[DataLoader] = None,
    batch_transform: OptionalBatchTransform = None,
    extra_loss: OptionalExtLoss =None,
    test_every: int = 2,
    save_every: int = 50,
    save_dir: Optional[str] = None,
    compile_model: bool = False,
    log: Union[Callable[[Dict], None], None] = None,
    custom_evaluate: Union[Callable[[nn.Module, nn.Module, torch.device, OptionalBatchTransform], Dict], None] = None,
    custom_train: Union[Callable[[nn.Module, nn.Module, DataLoader, torch.optim.Optimizer, torch.device, OptionalBatchTransform, OptionalExtLoss], Dict], None] = None,
    log_conf_interval: bool = False,
    step_key: str = "epoch",
):
    """Train a PyTorch model via one function call.

    Args:
        cfg (DotMap): Configuration for the training process, containing information for the criterion, optimizer and optionally the lr_scheduler and early_stopping.
        model (nn.Module): Model to train.
        train_data_loader (DataLoader): Training data loader.
        val_data_loader (Optional[DataLoader]): Validation data loader.
        device (torch.device): Device to move model and training batches to.
        test_data_loader (Optional[DataLoader], optional): Testing data loader. If specified, the model will be evaluated on the test set after training. Defaults to None.
        batch_transform (OptionalBatchTransform, optional): Transform to apply to whole batch of data during training/validation/testing. Defaults to None.
        extra_loss (OptionalExtLoss, optional): Extra loss function which gets added to the normal loss. This function receives the batch transformed data and the model outputs for that batch via `extra_loss(X, output)`. Defaults to None.
        test_every (int, optional): How often to measure validation performance. Defaults to 2.
        save_every (int, optional): How often to save the model. If set to 0, the model will never be saved, if set to -1, the model will only be saved in the last step. Defaults to 50.
        save_dir (Optional[str], optional): Where to save the model. Prepended inside of this function by `nets/`. Defaults to None.
        compile_model (bool, optional): Whether or not to compile the model (for PyTorch 2.0). Defaults to False.
        log (Union[Callable[[Dict], None], None], optional): Logging callback function through which all model performance will be communicated. Can be used for monitoring or metric tracking. Defaults to None.
        custom_evaluate (Union[Callable[[nn.Module, nn.Module, torch.device, OptionalBatchTransform], Dict], None], optional): _description_. Defaults to None.
        custom_train (Union[Callable[[nn.Module, nn.Module, DataLoader, torch.optim.Optimizer, torch.device, OptionalBatchTransform, OptionalExtLoss], Dict], None], optional): _description_. Defaults to None.
        log_conf_interval (bool, optional): Whether or not to log the 95% confidence interval of the test and validation accuracy in the last epoch. Defaults to False.
        step_key (str, optional): Logging step key, which is added to every dict that is passed to `log`. Defaults to "epoch".

    Raises:
        ValueError: If XOR of val_data_loader and custom_evaluate is False
    """
    assert save_dir is not None if save_every != 0 else True, "save_dir must be provided if save_every is not 0."

    if val_data_loader is None and custom_evaluate is None:
        raise ValueError(
            "No validation data loader provided, but no custom evaluation function either."
        )
    
    if custom_evaluate is not None and val_data_loader is not None:
        raise ValueError(
            "Both a validation data loader and a custom evaluation function were provided."
        )
    
    model_save_dir_path = f"nets/{save_dir}"
    os.makedirs(model_save_dir_path)

    model.to(device)

    if compile_model:
        model = torch.compile(model)

    opt = getattr(optim, cfg.optimizer.name)(model.parameters(),
                                             **cfg.optimizer.cfg.toDict())
    scheduler = getattr(optim.lr_scheduler, cfg.lr_scheduler.name)(
        opt, **cfg.lr_scheduler.cfg.toDict()) if cfg.lr_scheduler else None

    loss = getattr(nn, cfg.criterion.name)()

    stop_early = EarlyStopping(*cfg.early_stopping.cfg) if cfg.early_stopping else None

    if log is None:
        log = lambda _: None

    for epoch in tqdm(range(cfg.task.train.n_epochs)):
        train_data = {
            step_key: epoch,
        }

        if not custom_train:
            train_error, train_accuracy = train(model, loss, train_data_loader,
                                        opt, device, batch_transform,
                                        extra_loss)

            train_data.update({
                "train/loss": train_error,
                "train/accuracy": train_accuracy
            })
        else:
            train_data.update(custom_train(model, loss, train_data_loader, opt, device, batch_transform, extra_loss))

        log(train_data)

        if cfg.lr_scheduler:
            scheduler.step()

        is_last_epoch: bool = epoch == cfg.task.train.n_epochs - 1

        break_this_step = False

        if epoch % test_every == 0 or is_last_epoch:
            evaluation_data = {step_key: epoch}

            if val_data_loader is not None:
                val_error, (val_accuracy_avg, val_accuracy_conf_interval) = evaluate(model, loss, val_data_loader,
                                             device, batch_transform, extra_loss)

                evaluation_data.update({
                    "val/loss": val_error,
                    "val/accuracy": val_accuracy_avg
                })

                if log_conf_interval and is_last_epoch:
                    evaluation_data["val/accuracy_conf_interval"] = val_accuracy_conf_interval
            elif custom_evaluate is not None:
                evaluation_data.update(custom_evaluate(model, loss, device, batch_transform))

            log(evaluation_data)

            if stop_early:
                break_this_step = stop_early(evaluation_data.values()[1])

        if (save_every > 0 and (epoch % save_every == 0 or is_last_epoch or break_this_step)) or (save_every == -1 and (is_last_epoch or break_this_step)):
            # Also support DataParallel
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            torch.save(state_dict,
                       f"{model_save_dir_path}/{step_key}_{epoch}.pth")
            
        if break_this_step:
            break

    if test_data_loader:
        test_error, (test_accuracy_avg, test_accuracy_conf_interval) = evaluate(model, loss,
                                                  test_data_loader, device,
                                                  batch_transform, extra_loss)

        test_data = {
            step_key: cfg.task.train.n_epochs - 1,
            "test/loss": test_error,
            "test/accuracy": test_accuracy_avg
        }

        if log_conf_interval:
            test_data["test/accuracy_conf_interval"] = test_accuracy_conf_interval

        log(test_data)
