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

- [`torch_mate/data`](src/torch_mate/data/README.md): A collection of datasets, samplers, transforms and utilities for special use cases (i.e. few-shot learning, pre-loading datasets, siamese approaches).
- [`torch_mate/models`](src/torch_mate/models/README.md): A collection of models that can be used to quickly prototype and test different architectures.
- [`torch_mate/utils`](src/torch_mate/utils/README.md): A collection of utilities that can be used to simplify common tasks when working with CUDA devices and PyTorch.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--


meta-learning
unsupervised learning
pruning
quantization
knowledge distillation

Always one training method but maybe multiple validation methods


THINK ABOUT LOGGING via {"loss": loss} (returning from training_step for example)
how to make specialized pytorch lightning data modules
how to make forward pass easily changable
how to indicate which algorithm is used in which step
log number of parameters
look at all TODOs (# TODO)
Should generic train step and test step be configurable via hparams?

Different layers for different passes
Different algo in forward 

Solutions
Special forward

P>M>F 
F = copy model for each task and finetune it