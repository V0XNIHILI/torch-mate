# TorchMate (`torch-mate`)

## Installation

```bash
git clone git@github.com:V0XNIHILI/torch-mate.git
cd torch-mate
# Make sure you have pip 23 or higher
pip install -e .
```

## Usage

TorchMate consists of a few different modules:

- [`torch_mate/lightning`](src/torch_mate/lightning/README.md): A PyTorch Lightning wrapper framework that allows for easy configuration of the model, data and trainer from a single dictionary, while maintaining the flexibility of PyTorch Lightning.
- [`torch_mate/data`](src/torch_mate/data/README.md): A collection of datasets, samplers, transforms and utilities for special use cases (i.e. few-shot learning, pre-loading datasets, siamese approaches).
- [`torch_mate/models`](src/torch_mate/models/README.md): A collection of models that can be used to quickly prototype and test different architectures.
- [`torch_mate/utils`](src/torch_mate/utils/README.md): A collection of utilities that can be used to simplify common tasks when working with CUDA devices and PyTorch.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
