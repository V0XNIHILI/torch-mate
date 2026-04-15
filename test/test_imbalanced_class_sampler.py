import torch

from torch_mate.data.samplers.imbalanced_class_sampler import ImbalancedClassSampler

from utils import LabelsAreData


def test_imbalanced_class_sampler_assigns_inverse_frequency_weights():
    data = LabelsAreData([0, 0, 1, 1, 1, 2], length=6)

    sampler = ImbalancedClassSampler(data, length=12)

    expected_weights = torch.tensor([
        1 / 2,
        1 / 2,
        1 / 3,
        1 / 3,
        1 / 3,
        1.0,
    ], dtype=sampler.weights.dtype)

    assert torch.allclose(sampler.weights, expected_weights)
