from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from sklearn.linear_model import LogisticRegression
from torch_mate.utils import calc_accuracy

from torch_mate.flow.main import OptionalBatchTransform

MetaSample = Tuple[Tuple[torch.Tensor, torch.Tensor],
                                            Tuple[torch.Tensor, torch.Tensor]]
MetaBatch = List[MetaSample]


UNKNOWN_METRIC_MESSAGE = 'Must be one of [euclidean, euclidean-squared, manhattan, dot, cosine, logistic-regression]'


def few_shot_nearest_neighbor(embedder: nn.Module,
                              loss: nn.Module,
                              metric: str,
                              average_support_embeddings: bool,
                              sample: MetaSample,
                              batch_transform: OptionalBatchTransform = None):
    # It is assumed that the train labels are structured like [0] * k_shot + [1] * k_shot, ...
    # and the evaluation labels are structured like [0] * k_query_shot + [1] * k_query_shot, ...
    ((train_data, test_data), (train_labels, evaluation_labels)) = sample

    data = torch.cat([train_data, test_data], dim=0)

    if batch_transform:
        data = batch_transform(data)

    embeddings = embedder(data)
    support_embeddings = embeddings[:train_data.size(0)]
    query_embeddings = embeddings[train_data.size(0):]

    k_shot = len(train_labels) // len(torch.unique(train_labels))
    total_queries = query_embeddings.size(0)

    if average_support_embeddings:
        # Average every k-shot embeddings to get a single embedding for each class
        grouped_embeddings = torch.reshape(support_embeddings, (-1, k_shot, support_embeddings.size(1)))
        support_embeddings = torch.mean(grouped_embeddings, dim=1)

    if metric.startswith('euclidean'):
        similarities = -torch.cdist(query_embeddings, support_embeddings)

        if metric.endswith('-squared'):
            similarities = -similarities ** 2
        elif not metric.endswith('euclidean'):
            raise ValueError(f'Unknown metric: {metric}. {UNKNOWN_METRIC_MESSAGE}')
    elif metric == 'logistic-regression':
        clf = LogisticRegression(random_state=0).fit(support_embeddings.cpu().numpy(), train_labels.cpu().numpy())
        similarities = torch.tensor(clf.predict_proba(query_embeddings.cpu().numpy()), device=evaluation_labels.device)
    else:    
        similarities = torch.empty(
            (total_queries, support_embeddings.size(0)),
            device=evaluation_labels.device)

        for i in range(total_queries):
            if metric == 'manhattan':
                similarities[i] = -F.pairwise_distance(
                    query_embeddings[i], support_embeddings, p=1)
            elif metric == 'dot':
                similarities[i] = torch.matmul(support_embeddings,
                                                    query_embeddings[i])
            elif metric == 'cosine':
                similarities[i] = F.cosine_similarity(
                    query_embeddings[i], support_embeddings)
            else:
                raise ValueError(f'Unknown metric: {metric}. {UNKNOWN_METRIC_MESSAGE}')

    if average_support_embeddings or metric == 'logistic-regression':
        shots_per_class = 1
    else:
        shots_per_class = k_shot

    # Take the maximum similarity per query sample between all support samples of a class. This means that in case there
    # are multiple support samples (k_shot > 1) for a certain class, per support class we find the support sample that is most similar
    # to the query sample.
    # Evaluation data is shaped like [(class1, shot1, data), (class1, shot2, data), (class2, shot1, data), ...)]
    similarities = torch.reshape(
        torch.max(torch.reshape(similarities, (-1, shots_per_class)),
                  dim=1).values, (-1, similarities.shape[1] // shots_per_class))

    evaluation_error = loss(similarities, evaluation_labels)

    evaluation_accuracy = calc_accuracy(similarities, evaluation_labels)

    return evaluation_error, evaluation_accuracy


def metric_learning_process_batch(embedder: nn.Module, batch: MetaBatch,
                                metric: str, average_support_embeddings: bool,
                                batch_transform: OptionalBatchTransform,
                                loss: nn.Module):
    ((X_train, X_test),
        (y_train, y_test)) = batch

    meta_error = 0.0
    meta_accuracies = []

    meta_batch_size = len(y_test)

    for task_idx in range(meta_batch_size):
        error, accuracy = few_shot_nearest_neighbor(
            embedder, loss, metric, average_support_embeddings,
            ((X_train[task_idx], X_test[task_idx]),
                (y_train[task_idx], y_test[task_idx])), batch_transform)

        meta_error += error.item()
        meta_accuracies.append(accuracy.item())

    meta_accuracies = np.array(meta_accuracies)

    return meta_error / meta_batch_size, (np.mean(meta_accuracies), 1.96 * np.std(meta_accuracies) / np.sqrt(meta_batch_size))

from torch_mate.lightning import ConfigurableLightningModule

def generic_step(module: ConfigurableLightningModule, batch, batch_idx, phase: str):
    loss, (accuracy, confidence_interval) = metric_learning_process_batch(
        module, batch, module.hparams.few_shot["cfg"]["metric"],
        module.hparams.few_shot["cfg"]["metric.average_support_embeddings"],
        None, module.criterion)
    
    module.log_dict({
        f"meta_{phase}/loss": loss,
        f"meta_{phase}/accuracy": accuracy,
        f"meta_{phase}/accuracy/95p_ci": confidence_interval
    })
    
    return loss

