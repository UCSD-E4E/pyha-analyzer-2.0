# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pyha-analyzer-2.0** is a machine learning framework for audio classification (primarily bird species identification) built on top of HuggingFace Transformers. It provides an abstraction layer for training audio classification models using Mel-spectrogram-based approaches.

## Setup & Installation

`uv` is the package manager. Python 3.11 is required (enforced by `.python-version`).

```bash
uv sync --extra cpu      # CPU-only
uv sync --extra cu126    # CUDA 12.6
uv sync --extra cu128    # CUDA 12.8
uv sync --extra wandb    # Add WandB logging
```

## Running Experiments

There is no formal test suite or linting configuration. Experiments are run directly:

```bash
python experiment_birdset_replication.py
python experiment_peru_fewshot.py
python experiment_background_noise.py
python demo-notebooks/General/train.py
```

## Architecture

### Data Flow

```
Extractor → AudioDataset → Preprocessor (mel-spectrogram + augmentation) → PyhaTrainer → Metrics
```

1. **Extractors** (`pyha_analyzer/extractors/`) load datasets via HuggingFace `load_dataset()` and return an `AudioDataset`
2. **Preprocessors** (`pyha_analyzer/preprocessors/`) convert raw audio → mel-spectrograms and apply augmentations via `set_transform()`
3. **PyhaTrainer** (`pyha_analyzer/trainer.py`) wraps HuggingFace `Trainer` with multilabel classification support
4. **Metrics** (`pyha_analyzer/metrics/`) compute cMAP and AUROC for multilabel evaluation

### Core Modules

- **`pyha_analyzer/trainer.py`** — `PyhaTrainer` and `PyhaTrainingArguments`; wraps HF Trainer. Default training config: batch size 64 train / 32 eval, 4 dataloader workers.
- **`pyha_analyzer/dataset.py`** — `AudioDataset` extending HuggingFace `DatasetDict`. Required splits: `train`, `valid`, `test`. Required columns: `audio`, `audio_in`, `labels`, `filepath`.
- **`pyha_analyzer/models/base_model.py`** — `BaseModel` abstract class. All models must implement `forward()` decorated with `@has_required_inputs()`, accept `audio`/`audio_in`/`labels`, and return `{"loss": ..., "logits": ...}`.
- **`pyha_analyzer/preprocessors/spectogram_preprocessors.py`** — `MelSpectrogramPreprocessors`; defaults: 256 mel bins, 2048 FFT, 256 hop length, 5-second windows.
- **`pyha_analyzer/constants.py`** — Column name constants and default configs shared across modules.

### Models

- `efficientnet.py` — EfficientNet-B1 (primary production model)
- `demo_CNN.py` — ResNet-based demo models

### Extractors (Data Sources)

- `birdset.py` — BirdSet dataset (main focus; takes a region code like `"HSN"`)
- `peru132.py` — Peru 132 species dataset
- `coralreef.py`, `multi_coral.py`, `GNNcoral.py` — Coral reef variants
- `defaultExtractors.py` — Abstract base classes for extractors

## Typical Training Script Pattern

```python
extractor = extractors.Birdset()
dataset = extractor("HSN")

augmentation_pipeline = ...  # audiomentations Compose
preprocessor = MelSpectrogramPreprocessors(duration=5, augment=augmentation_pipeline)
dataset["train"].set_transform(preprocessor)
dataset["valid"].set_transform(preprocessor)

model = EfficentNet(num_classes=dataset.get_number_species())

args = PyhaTrainingArguments(working_dir="output_dir", run_name="experiment_name")
args.num_train_epochs = 30
args.learning_rate = 0.001

trainer = PyhaTrainer(model=model, dataset=dataset, training_args=args)
trainer.train()
trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
```

## Key Conventions

- **Multilabel classification**: labels are one-hot encoded; multiple species can occur per clip
- **Dataset format**: HuggingFace `DatasetDict` with `Sequence(ClassLabel(...))` for labels
- **Model contract**: `forward()` must accept `audio`, `audio_in`, `labels` (defined in `constants.MODEL_COLUMNS`) and return `{"loss", "logits"}`
- **WandB**: optional; add `--extra wandb` and configure `Logger` in training scripts
- **`uv`**: always use `uv` instead of `pip` or `conda` for dependency management
