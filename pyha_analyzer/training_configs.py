"""
Training configuration classes for PyHa.

This module provides dataclasses for managing data and training configurations,
enabling reproducible and easily configurable training workflows.
"""

from dataclasses import dataclass, field
from typing import Optional


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
    """
    region: str
    sampling_rate: int = 32_000
    class_limit: Optional[int] = None
    event_limit: Optional[int] = None
    max_event_length: Optional[float] = None
    batch_size: int = 32


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
    per_device_eval_batch_size: int = 32
    dataloader_num_workers: int = 4
    logging_steps: int = 10
    eval_accumulation_steps: int = 100
    learning_rate: float = 5e-5
