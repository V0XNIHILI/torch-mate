# TorchMate (`torch-mate`)

## Installation

### As editable

```bash
git clone git@github.com:V0XNIHILI/torch-mate.git
cd torch-mate
# Make sure you have pip 23 or higher
pip install -e .
```

### Regular way

```bash
pip install git+https://github.com/V0XNIHILI/torch-mate.git
```

## Usage

TorchMate consists of a few different modules:

- [`torch_mate/data`](src/torch_mate/data/README.md): A small collection of datasets, samplers, transforms and utilities for special use cases (i.e. few-shot learning, pre-loading datasets, siamese approaches).
- [`torch_mate/models`](src/torch_mate/models/README.md): A collection of models that can be used to quickly prototype and test different architectures.
- [`torch_mate/utils`](src/torch_mate/utils/README.md): A collection of utilities that can be used to simplify common tasks when working with CUDA devices and PyTorch.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
