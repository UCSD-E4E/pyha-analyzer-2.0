"""
Training configuration classes for PyHa.

This module provides dataclasses for managing data and training configurations,
enabling reproducible and easily configurable training workflows.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Attributes:
        region: Geographic or dataset region identifier (e.g., "HSN", "NE")
        sampling_rate: Audio sampling rate in Hz (e.g., 32000)
        class_limit: Maximum number of classes/species to include
        event_limit: Maximum number of events per class
        max_event_length: Maximum length of an event in seconds
        batch_size: Batch size for data loading
        train_test_split: Fraction of training data to use for validation (0.0-1.0)
        chunking: Chunking strategy to use for spectrogram preprocessing
    """
    region: str
    sampling_rate: int = 32_000
    class_limit: Optional[int] = None
    event_limit: Optional[int] = None
    max_event_length: Optional[float] = None
    batch_size: int = 32
    train_test_split: float = 0.2
    chunking: ["random", "detected_event_chunking"] = "random"


@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    
    Attributes:
        working_dir: Directory where model checkpoints and outputs will be saved
        run_name: Name identifier for this training run
        project_name: Name of the project (used for logging/tracking)
        num_train_epochs: Number of training epochs
        eval_steps: Number of steps between evaluations
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        dataloader_num_workers: Number of workers for data loading
        logging_steps: Number of steps between logging events
        eval_accumulation_steps: Number of steps to accumulate predictions before evaluation
        learning_rate: Learning rate for the optimizer
    """
    working_dir: str
    run_name: str
    project_name: str
    num_train_epochs: int = 1
    eval_steps: int = 100
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    dataloader_num_workers: int = 16
    logging_steps: int = 10
    eval_accumulation_steps: int = 100
    learning_rate: float = 5e-4


@dataclass
class AugmentationConfig:
    """
    Configuration for audio augmentations.
    
    Controls which augmentations are applied during training and their parameters.
    All augmentations are optional and disabled by default.
    
    Attributes:
        enable_background_noise: Whether to apply AddBackgroundNoise augmentation
        background_noise_path: Path to directory containing background noise files
        background_noise_min_snr_db: Minimum SNR for background noise in dB
        background_noise_max_snr_db: Maximum SNR for background noise in dB
        background_noise_p: Probability of applying background noise (0.0-1.0)
        
        enable_gain: Whether to apply Gain augmentation
        gain_min_gain_db: Minimum gain in dB
        gain_max_gain_db: Maximum gain in dB
        gain_p: Probability of applying gain (0.0-1.0)
        
        enable_mixitup: Whether to apply MixItUp augmentation
        mixitup_min_snr_db: Minimum SNR for MixItUp in dB
        mixitup_max_snr_db: Maximum SNR for MixItUp in dB
        mixitup_p: Probability of applying MixItUp (0.0-1.0)
        
        enable_polarity_inversion: Whether to apply PolarityInversion augmentation
        polarity_inversion_p: Probability of applying polarity inversion (0.0-1.0)
    """
    # Background Noise Augmentation
    enable_background_noise: bool = False
    background_noise_path: Optional[str] = None
    background_noise_min_snr_db: float = 3.0
    background_noise_max_snr_db: float = 30.0
    background_noise_p: float = 0.5
    
    # Gain Augmentation
    enable_gain: bool = False
    gain_min_gain_db: float = -18.0
    gain_max_gain_db: float = 6.0
    gain_p: float = 0.2
    
    # MixItUp Augmentation
    enable_mixitup: bool = False 
    mixitup_min_snr_db: float = 3.0
    mixitup_max_snr_db: float = 30.0
    mixitup_p: float = 0.7
    
    # Polarity Inversion Augmentation
    enable_polarity_inversion: bool = False
    polarity_inversion_p: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return {
            'background_noise': {
                'enabled': self.enable_background_noise,
                'sounds_path': self.background_noise_path,
                'min_snr_db': self.background_noise_min_snr_db,
                'max_snr_db': self.background_noise_max_snr_db,
                'p': self.background_noise_p,
            },
            'gain': {
                'enabled': self.enable_gain,
                'min_gain_db': self.gain_min_gain_db,
                'max_gain_db': self.gain_max_gain_db,
                'p': self.gain_p,
            },
            'mixitup': {
                'enabled': self.enable_mixitup,
                'min_snr_db': self.mixitup_min_snr_db,
                'max_snr_db': self.mixitup_max_snr_db,
                'p': self.mixitup_p,
            },
            'polarity_inversion': {
                'enabled': self.enable_polarity_inversion,
                'p': self.polarity_inversion_p,
            },
        }
