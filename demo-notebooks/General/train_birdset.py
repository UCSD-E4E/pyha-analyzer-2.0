"""
BirdSet Training Pipeline

Entry point for BirdSet training workflow with PyHa.

This script orchestrates the modular data pipeline and training process.
Separates concerns: configuration, data loading, and training.

Usage:
    python train_birdset.py 

"""

from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments
from pyha_analyzer.extractors.birdset_pipeline import BirdSetDataPipeline
from pyha_analyzer.metrics.classification_metrics import AudioClassificationMetrics
from pyha_analyzer.models import EfficentNet
from pyha_analyzer.training_configs import DataConfig, TrainingConfig
from pyha_analyzer.preprocessors.birdset_event_decoding import EventDecoding
from pyha_analyzer.preprocessors.augmentations import ComposeAudioLabel, MixItUp
from audiomentations import AddBackgroundNoise, Gain, PolarityInversion
import argparse, os
from safetensors.torch import load_file

import warnings
warnings.filterwarnings("ignore") #AUDIOMENTIONS REALLY NEEDS TO QUIET RESAMPLING WARNINGS

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the output")


def main(
    save: bool
):
    """Main training pipeline orchestration."""
    
    # Configuration
    data_config = DataConfig(
        region="HSN",
        sampling_rate=32_000,
        class_limit=500,
        event_limit=5,
        max_event_length=5.0,
        batch_size=300,
    )
    
    training_config = TrainingConfig(
        working_dir="working_dir",
        run_name="30-train-birdset-chunking-preprocessor-with-aug",
        project_name="egci_bioacoustic_shifts",
        num_train_epochs=1,
        eval_steps=1000,
    )
    
    # Data loading
    data_pipeline = BirdSetDataPipeline(data_config)
    audio_dataset = data_pipeline.process_full()
    
    # Training
    num_classes = audio_dataset.get_number_species()
    model = EfficentNet(num_classes=num_classes)
    # state_dict = load_file('/home/s.dalal.334/models/checkpoint-6750/model.safetensors')
    # model.load_state_dict(state_dict)
    # model.eval()


    training_args = PyhaTrainingArguments(
        working_dir=training_config.working_dir,
        run_name=training_config.run_name,
        project_name=training_config.project_name,
    )

    training_args.num_train_epochs = 30
    training_args.eval_steps = training_config.eval_steps
    training_args.per_device_train_batch_size = (
        training_config.per_device_train_batch_size
    )
    training_args.per_device_eval_batch_size = (
        training_config.per_device_eval_batch_size
    )
    training_args.dataloader_num_workers = training_config.dataloader_num_workers
    training_args.logging_steps = training_config.logging_steps
    training_args.eval_accumulation_steps = training_config.eval_accumulation_steps
    training_args.learning_rate = training_config.learning_rate
    training_args.save_steps = 0.5
    training_args.save_strategy = "steps"
    training_args.output_dir = "/home/s.dalal.334/models/HSN_BirdSet/with_aug"


    compute_metrics = AudioClassificationMetrics([], num_classes=num_classes)

    trainer = PyhaTrainer(
        model=model,
        dataset=audio_dataset,
        metrics=compute_metrics,
        training_args=training_args,
    )
    trainer.train()
    
    if (save):
        save_dir = f"/home/s.dalal.334/models/{training_config.num_train_epochs}-{data_config.region}-no-aug"
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir)
        print(f"Model saved to {save_dir}")
    
    # Evaluation
    results = trainer.evaluate(eval_dataset=audio_dataset["test"], metric_key_prefix="Soundscape")
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(results)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.save:
        print("Saving model")
    else:
        print("not saving model")

    main(args.save)
