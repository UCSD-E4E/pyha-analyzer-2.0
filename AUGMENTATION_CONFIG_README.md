# Audio Augmentation Configuration System

A clean, config-based approach to managing audio augmentations in the PyHa analyzer.

## Overview

The `AugmentationConfig` class provides a declarative way to configure audio augmentations for training. Instead of hardcoding augmentation parameters in your pipeline code, you define them in a configuration object that can be easily customized, saved, and reproduced.

## Quick Start

### Basic Usage

```python
from pyha_analyzer.training_configs import DataConfig, AugmentationConfig
from pyha_analyzer.extractors.birdset_pipeline import BirdSetDataPipeline

# Create your data configuration
data_config = DataConfig(
    region="HSN",
    sampling_rate=32000,
    class_limit=10,
)

# Create your augmentation configuration
aug_config = AugmentationConfig(
    background_noise_path="/path/to/background/noise",
)

# Pass both to the pipeline
pipeline = BirdSetDataPipeline(data_config, aug_config)
```

## AugmentationConfig Parameters

### Background Noise Augmentation

- `enable_background_noise` (bool, default: True)
  - Enable/disable background noise augmentation
  
- `background_noise_path` (str, required if enabled)
  - Path to directory containing background noise audio files
  
- `background_noise_min_snr_db` (float, default: 3.0)
  - Minimum Signal-to-Noise Ratio in dB
  - Lower values = more background noise
  
- `background_noise_max_snr_db` (float, default: 30.0)
  - Maximum Signal-to-Noise Ratio in dB
  
- `background_noise_p` (float, default: 0.5)
  - Probability of applying augmentation (0.0 to 1.0)

### Gain Augmentation

- `enable_gain` (bool, default: True)
  - Enable/disable gain augmentation
  
- `gain_min_gain_db` (float, default: -18.0)
  - Minimum gain in dB (negative = reduce volume)
  
- `gain_max_gain_db` (float, default: 6.0)
  - Maximum gain in dB (positive = increase volume)
  
- `gain_p` (float, default: 0.2)
  - Probability of applying augmentation

### MixItUp Augmentation

- `enable_mixitup` (bool, default: True)
  - Enable/disable MixItUp augmentation (mixes samples from the training set)
  
- `mixitup_min_snr_db` (float, default: 3.0)
  - Minimum SNR for mixed samples
  
- `mixitup_max_snr_db` (float, default: 30.0)
  - Maximum SNR for mixed samples
  
- `mixitup_p` (float, default: 0.7)
  - Probability of applying augmentation

### Polarity Inversion Augmentation

- `enable_polarity_inversion` (bool, default: False)
  - Enable/disable polarity inversion
  
- `polarity_inversion_p` (float, default: 0.1)
  - Probability of applying augmentation

## Configuration Presets

### Baseline (No Augmentations)

Use this for your baseline model to measure the impact of augmentations:

```python
aug_config = AugmentationConfig(
    enable_background_noise=False,
    enable_gain=False,
    enable_mixitup=False,
    enable_polarity_inversion=False,
)
```

### Conservative Augmentation

Light augmentation suitable for small datasets:

```python
aug_config = AugmentationConfig(
    background_noise_path="/path/to/noise",
    background_noise_p=0.2,
    gain_p=0.1,
    enable_mixitup=False,
)
```

### Aggressive Augmentation

Heavy augmentation for better generalization on large datasets:

```python
aug_config = AugmentationConfig(
    background_noise_path="/path/to/noise",
    background_noise_p=0.8,
    background_noise_min_snr_db=1.0,
    gain_p=0.5,
    gain_min_gain_db=-24.0,
    gain_max_gain_db=12.0,
    mixitup_p=0.8,
    enable_polarity_inversion=True,
    polarity_inversion_p=0.2,
)
```

## Common Recipes

### For Limited Data

When you have a small dataset and want to prevent overfitting:

```python
aug_config = AugmentationConfig(
    background_noise_path="/path/to/noise",
    background_noise_p=0.6,
    background_noise_min_snr_db=2.0,
    background_noise_max_snr_db=20.0,
    gain_p=0.4,
    gain_min_gain_db=-15.0,
    gain_max_gain_db=9.0,
    mixitup_p=0.5,
    enable_polarity_inversion=True,
    polarity_inversion_p=0.15,
)
```

### For Noise-Robust Model

Build a model that's robust to noise and SNR variations:

```python
aug_config = AugmentationConfig(
    background_noise_path="/path/to/noise",
    background_noise_p=0.9,
    background_noise_min_snr_db=0.5,  # Very noisy
    background_noise_max_snr_db=15.0,
    gain_p=0.3,
    enable_mixitup=True,
    mixitup_p=0.7,
    enable_polarity_inversion=False,
)
```

### For Speech/Birdsong Focus

Minimal augmentation to preserve signal integrity:

```python
aug_config = AugmentationConfig(
    background_noise_path="/path/to/noise",
    background_noise_p=0.3,
    background_noise_min_snr_db=10.0,
    background_noise_max_snr_db=35.0,
    gain_p=0.1,
    gain_min_gain_db=-6.0,
    gain_max_gain_db=3.0,
    enable_mixitup=False,
    enable_polarity_inversion=False,
)
```

## Using with Training Scripts

Update your training script to accept augmentation configuration:

```python
from pyha_analyzer.training_configs import DataConfig, AugmentationConfig
from pyha_analyzer.extractors.birdset_pipeline import BirdSetDataPipeline

def main(augmentation_config=None):
    data_config = DataConfig(region="HSN")
    
    # Use provided config or create default
    if augmentation_config is None:
        augmentation_config = AugmentationConfig(
            background_noise_path="/path/to/noise"
        )
    
    pipeline = BirdSetDataPipeline(data_config, augmentation_config)
    dataset = pipeline.process_full()
    
    # Train your model...
```

## Inspecting Configuration

### Print Configuration

```python
aug_config = AugmentationConfig(background_noise_path="/path/to/noise")

# Convert to dictionary format
config_dict = aug_config.to_dict()
print(config_dict)
```

Output:
```python
{
    'background_noise': {
        'enabled': True,
        'sounds_path': '/path/to/noise',
        'min_snr_db': 3.0,
        'max_snr_db': 30.0,
        'p': 0.5,
    },
    'gain': {
        'enabled': True,
        'min_gain_db': -18.0,
        'max_gain_db': 6.0,
        'p': 0.2,
    },
    ...
}
```

## Troubleshooting

### AssertionError: No background noise files found

**Error:**
```
AssertionError
assert len(self.sound_file_paths) > 0
```

**Solution:** Make sure `background_noise_path` points to a valid directory with audio files:

```python
import os
bg_path = "/path/to/background/noise"
if os.path.exists(bg_path):
    files = os.listdir(bg_path)
    print(f"Found {len(files)} files in {bg_path}")
else:
    print(f"Path does not exist: {bg_path}")

aug_config = AugmentationConfig(background_noise_path=bg_path)
```

### Required parameter missing

**Error:**
```
ValueError: background_noise_path must be set when enable_background_noise=True
```

**Solution:** Either provide the path or disable the augmentation:

```python
# Option 1: Provide the path
aug_config = AugmentationConfig(
    background_noise_path="/path/to/noise"
)

# Option 2: Disable the augmentation
aug_config = AugmentationConfig(
    enable_background_noise=False
)
```

## Advanced Usage

### Creating Configuration Presets

Define commonly used configurations in a module:

```python
# augmentation_presets.py
from pyha_analyzer.training_configs import AugmentationConfig

BASELINE = AugmentationConfig(
    enable_background_noise=False,
    enable_gain=False,
    enable_mixitup=False,
    enable_polarity_inversion=False,
)

CONSERVATIVE = AugmentationConfig(
    background_noise_path="/path/to/noise",
    background_noise_p=0.2,
    gain_p=0.1,
    enable_mixitup=False,
)

AGGRESSIVE = AugmentationConfig(
    background_noise_path="/path/to/noise",
    background_noise_p=0.8,
    gain_p=0.5,
    mixitup_p=0.8,
    enable_polarity_inversion=True,
)
```

Then use them in your code:

```python
from augmentation_presets import CONSERVATIVE

aug_config = CONSERVATIVE
pipeline = BirdSetDataPipeline(data_config, aug_config)
```

### Comparing Multiple Configurations

```python
configs = {
    "baseline": AugmentationConfig(enable_background_noise=False, ...),
    "conservative": AugmentationConfig(background_noise_p=0.2, ...),
    "aggressive": AugmentationConfig(background_noise_p=0.8, ...),
}

for config_name, aug_config in configs.items():
    pipeline = BirdSetDataPipeline(data_config, aug_config)
    # Train and evaluate model
```

## See Also

- [augmentation_config_examples.py](../augmentation_config_examples.py) - More detailed examples
- [training_configs.py](../pyha_analyzer/training_configs.py) - Configuration classes
- [birdset_pipeline.py](../pyha_analyzer/extractors/birdset_pipeline.py) - Pipeline implementation
