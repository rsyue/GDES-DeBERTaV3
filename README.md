# GDES-DeBERTaV3

ELECTRA-style training with **Gradient-Disentangled Embedding Sharing (GDES)** in native PyTorch using HuggingFace Transformers.

## Overview

GDES-DeBERTaV3 implements the replaced token detection (RTD) pretraining objective from [ELECTRA](https://arxiv.org/abs/2003.10555) with the gradient-disentangled embedding sharing technique introduced in [DeBERTaV3](https://arxiv.org/abs/2111.09543). Rather than relying on a separate generator and discriminator network, GDES shares embeddings between the two while disentangling their gradient flows — enabling more parameter-efficient training without the instability of naive weight tying.

The training loop performs two forward passes per step:

1. **Generator pass** — predict masked tokens via MLM
2. **Discriminator pass** — classify each token as original or replaced, with embedding gradients frozen to disentangle the two objectives

## Installation

```bash
git clone https://github.com/rsyue/rtd-gdes
cd rtd-gdes
```

For most users, a standard install is all that's needed:

```bash
pip install .
```

### Platform-specific installation

For NVIDIA Jetson and AMD GFX 1151 targets, use `setup_env.py` instead. It reads the `TARGET` environment variable to select the correct package index and architecture flags, then calls `pip install .` with the right options on your behalf. Custom indexes are queried first; PyPI is used as a fallback.

```bash
# NVIDIA Jetson Orin Nano
TARGET=jetson python setup_env.py

# AMD GFX 1151 (ROCm)
TARGET=amd-gfx1151 python setup_env.py

# Auto-detect Jetson hardware (no TARGET needed)
python setup_env.py
```

Append `--dev` to any command to also install development dependencies. CLI flags (`--jetson`, `--amd-gfx1151`, `--no-jetson`) are also supported and take precedence over `TARGET` if both are set.

#### Environment variables

| Variable | Values | Description |
|---|---|---|
| `TARGET` | `jetson`, `amd-gfx1151` | Selects the platform install mode. Read by `setup_env.py` at install time. |
| `TORCH_CUDA_ARCH_LIST` | `8.7` | Set automatically for Jetson builds. Restricts CUDA kernel compilation to sm_87 (Ampere GA10B). |
| `PYTORCH_ROCM_ARCH` | `gfx1151` | Set automatically for AMD builds. Restricts HIP/ROCm kernel compilation to GFX 1151. |

These variables are injected into the pip subprocess only and do not persist in your shell environment.

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended, ROCm builds supported)
- Transformers
- Datasets
- Safetensors
- scikit-learn
- tqdm

## Quick Start

```bash
python -m rtd_gdes.train \
  --model microsoft/deberta-v3-base \
  --lambda_disc 0.5 \
  --batch_size 8 \
  --epochs 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --gamma 0.9 \
  --bf16
```

## Usage

### CLI Arguments

| Argument | Flag | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | `str` | `microsoft/deberta-v3-base` | Pretrained model to train with RTD + GDES |
| `--lambda_disc` | `-ld` | `float` | `0.5` | Lambda coefficient scaling the discriminator loss |
| `--batch_size` | `-bs` | `int` | `8` | Batch size for training and evaluation |
| `--epochs` | `-ep` | `int` | `5` | Number of training epochs |
| `--learning_rate` | `-lr` | `float` | `2e-5` | Learning rate for AdamW |
| `--weight_decay` | `-wd` | `float` | `0.01` | Weight decay for AdamW |
| `--gamma` | `-g` | `float` | `0.9` | Gamma for exponential LR scheduler |
| `--dataset` | | `str` | `imdb` | HuggingFace dataset name |
| `--fp16` | | `flag` | `False` | Enable FP16 mixed precision |
| `--bf16` | | `flag` | `False` | Enable BF16 mixed precision |
| `--compile` | `-c` | `flag` | `False` | Run `torch.compile` with `max-autotune` mode |

### Training Details

The script trains on the [IMDB unsupervised split](https://huggingface.co/datasets/imdb) by default, with a configurable 90/10 train/eval split. The dataset can be changed via `--dataset`. The combined loss is computed as:

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + \lambda \cdot \mathcal{L}_{\text{disc}}$$

where $\mathcal{L}_{\text{gen}}$ is the standard MLM cross-entropy loss and $\mathcal{L}_{\text{disc}}$ is binary cross-entropy over token-level replaced/original predictions.

Evaluation reports discriminator loss, accuracy, and F1 score on the held-out set.

### Saved Outputs

After training, the model and tokenizer are saved to a directory named after the model (e.g., `deberta_v3_base_gdes/`).

## Development

Install with dev dependencies:

```bash
python setup_env.py --dev
```

Run the test suite:

```bash
pytest tests/ -v --cov=rtd_gdes
```

## Project Structure

```
rtd-gdes/
├── src/
│   └── rtd_gdes/
│       ├── config.py          # TrainConfig dataclass — all hyperparameter defaults
│       ├── train.py           # Entry point and CLI
│       └── gdes/
│           ├── data.py        # Dataset loading and DataLoader construction
│           ├── model.py       # DebertaV3GDES — generator + discriminator
│           ├── trainer.py     # train_one_epoch and evaluate loops
│           └── utils.py       # Shared exceptions
├── tests/
│   ├── test_gdes.py           # Model, trainer, and config unit + integration tests
│   └── test_setup_env.py      # Platform detection and install logic tests
├── setup_env.py               # Hardware-aware install helper
└── pyproject.toml
```

## Roadmap

- [ ] Distributed training (DDP / FSDP)
- [ ] Publish as PyPI package
- [ ] Support additional model architectures beyond DeBERTaV3

## Citation

If you use this code, please cite the original papers:

```bibtex
@article{he2021debertav3,
  title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing},
  author={He, Pengcheng and Liu, Jianfeng and Gao, Jianfeng and Chen, Weizhu},
  journal={arXiv preprint arXiv:2111.09543},
  year={2021}
}

@article{clark2020electra,
  title={ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators},
  author={Clark, Kevin and Luong, Minh-Thang and Le, Quoc V. and Manning, Christopher D.},
  journal={arXiv preprint arXiv:2003.10555},
  year={2020}
}
```

## License

MIT