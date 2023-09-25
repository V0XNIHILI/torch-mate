import os
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dotmap import DotMap

from torch_mate.utils.calc_accuracy import calc_accuracy
from torch_mate.contexts import evaluating, training
from torch_mate.utils.iterate_to_device import iterate_to_device

from tqdm import tqdm

OptionalBatchTransform = Union[Callable[[torch.Tensor], torch.Tensor], None]
OptionalExtLoss = Union[Callable[[], torch.Tensor], None]


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

    error = criterion(output, y) + (extra_loss() if extra_loss else 0.0)

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
             extra_loss: OptionalExtLoss = None,
             store_predictions: bool = False):
    eval_error = 0.0
    eval_accuracy = 0.0

    preds = []
    labels = []

    # Using evaluating(model) or evaluating(maml) here does not make a difference
    with evaluating(model), torch.no_grad():
        for (X, y) in iterate_to_device(eval_data_loader, device, True):
            (error, accuracy), (pred,
                                label) = process_batch(model, criterion, X, y,
                                                       batch_transform,
                                                       extra_loss)

            eval_error += error.detach()
            eval_accuracy += accuracy

            if store_predictions:
                preds.append(pred)
                labels.append(label)

    if store_predictions:
        preds = torch.cat(preds)
        labels = torch.cat(labels)
    else:
        preds = None
        labels = None

    return (eval_error.item() / len(eval_data_loader),
            eval_accuracy / len(eval_data_loader)), (preds, labels)


def main(
    cfg: DotMap,
    model: nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: Optional[DataLoader],
    device: torch.device,
    test_data_loader: Optional[DataLoader] = None,
    batch_transform: OptionalBatchTransform = None,
    extra_loss=None,
    test_every=2,
    save_every=50,
    run_name: Union[str, None] = None,
    compile_model: bool = True,
    log: Union[Callable[[Dict], None], None] = None,
    custom_evaluate: Union[Callable[[nn.Module, nn.Module, torch.device],
                                    Dict], None] = None,
):
    assert run_name is not None if save_every is not None else True, "run_name must be provided if save_every is not None."

    if val_data_loader is None and custom_evaluate is None:
        raise ValueError(
            "No validation data loader provided, but no custom evaluation function either."
        )
    
    if custom_evaluate is not None and val_data_loader is not None:
        raise ValueError(
            "Both a validation data loader and a custom evaluation function were provided."
        )
    
    model_save_dir_path = f"nets/{run_name}"
    os.makedirs(model_save_dir_path)

    model.to(device)

    if compile_model:
        model = torch.compile(model)

    opt = getattr(optim, cfg.optimizer.name)(model.parameters(),
                                             **cfg.optimizer.cfg.toDict())
    scheduler = getattr(optim.lr_scheduler, cfg.lr_scheduler.name)(
        opt, **cfg.lr_scheduler.cfg.toDict()) if cfg.lr_scheduler else None

    loss = getattr(nn, cfg.criterion.name)()

    if log is None:
        log = lambda _: None

    for epoch in tqdm(range(cfg.task.train.n_epochs)):
        train_error, train_accuracy = train(model, loss, train_data_loader,
                                            opt, device, batch_transform,
                                            extra_loss)

        log({
            "epoch": epoch,
            "train/loss": train_error,
            "train/accuracy": train_accuracy
        })

        if cfg.lr_scheduler:
            scheduler.step()

        is_last_epoch: bool = epoch == cfg.task.train.n_epochs - 1

        if epoch % test_every == 0 or is_last_epoch:
            evaluation_data = {"epoch": epoch}

            if val_data_loader is not None:
                (val_error,
                 val_accuracy), _ = evaluate(model, loss, val_data_loader,
                                             device, batch_transform, extra_loss)

                evaluation_data.update({
                    "val/loss": val_error,
                    "val/accuracy": val_accuracy
                })
            elif custom_evaluate is not None:
                evaluation_data.update(custom_evaluate(model, loss, device))

            log(evaluation_data)

        if epoch % save_every == 0 or is_last_epoch:
            # Also support DataParallel
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            torch.save(state_dict,
                       f"{model_save_dir_path}/epoch_{epoch}.pth")

    if test_data_loader:
        (test_error, test_accuracy), _ = evaluate(model, loss,
                                                  test_data_loader, device,
                                                  batch_transform, extra_loss)

        log({
            "epoch": cfg.task.train.n_epochs - 1,
            "test/loss": test_error,
            "test/accuracy": test_accuracy
        })
