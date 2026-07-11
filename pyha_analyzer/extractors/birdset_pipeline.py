"""
BirdSet data pipeline for PyHa-compliant dataset preparation.

Handles loading, preprocessing, and transforming BirdSet data through a
modular pipeline that can be composed and extended.
"""

from copy import copy
from datasets import load_dataset, Audio, DatasetDict, ClassLabel, Sequence

from pyha_analyzer import AudioDataset
from pyha_analyzer.preprocessors.birdset_event_mapper import XCEventMapping
from pyha_analyzer.preprocessors.birdset_utils import smart_sampling, classes_one_hot
from pyha_analyzer.preprocessors.birdset_event_decoding import EventDecoding
from pyha_analyzer.preprocessors.birdset_spectrogram_preprocessors import (
    BirdSetMelSpectrogramPreprocessor,
)

from pyha_analyzer.preprocessors.augmentations import ComposeAudioLabel, MixItUp
from audiomentations import AddBackgroundNoise, Gain, PolarityInversion
from pyha_analyzer.training_configs import DataConfig, AugmentationConfig


class BirdSetDataPipeline:
    """
    Handles all BirdSet data loading and preprocessing steps.
    
    This pipeline encapsulates the complete BirdSet workflow:
    - Load raw dataset
    - Audio normalization and sampling rate casting
    - Event mapping and smart sampling
    - One-hot label encoding
    - Train/valid/test splitting
    - Transform application (event decoding + spectrogram preprocessing)
    
    All steps can be executed via `process_full()` or individually.
    """
    
    def __init__(
        self,
        data_config: DataConfig,
        augmentation_config: AugmentationConfig = None,
    ):
        self.config = data_config
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.raw_dataset = None
        self.processed_dataset = None
        self.audio_dataset = None
        self.chunking = getattr(data_config, "chunking", "random")
    
    def load_raw(
        self
    ) -> DatasetDict:
        """Load raw BirdSet dataset for the specified region."""
        print(f">> Loading BirdSet dataset for region: {self.config.region}")
        self.raw_dataset = load_dataset(
            "DBD-research-group/BirdSet",
            self.config.region,
            trust_remote_code=True,
        )
        return self.raw_dataset
    
    def add_columns(
        self
    ) -> DatasetDict:
        """Add audio_in and labels columns."""
        print(">> Adding audio_in and labels columns.")
        for split in ["train", "test_5s"]:
            self.raw_dataset[split] = self.raw_dataset[split].add_column(
                "audio_in", self.raw_dataset[split]["audio"]
            )
            # self.raw_dataset[split] = self.raw_dataset[split].add_column(
            #     "labels", copy(self.raw_dataset[split]["ebird_code_multilabel"])
            # )
            self.raw_dataset[split] = self.raw_dataset[split].rename_column("ebird_code_multilabel", "labels")
        return self.raw_dataset
    
    def cast_audio(
        self
    ) -> DatasetDict:
        """Cast audio to proper format with sampling rate and mono conversion."""
        print(f">> Casting audio to {self.config.sampling_rate}Hz mono.")
        self.raw_dataset = self.raw_dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.config.sampling_rate,
                mono=True,
                decode=True,
            ),
        )
        return self.raw_dataset
    
    def extract_splits(
        self
    ) -> DatasetDict:
        """Extract only train and test_5s splits."""
        print(">> Extracting train and test_5s splits.")
        self.raw_dataset = DatasetDict(
            {split: self.raw_dataset[split] for split in ["train", "test_5s"]}
        )
        return self.raw_dataset
    
    def event_mapping(
        self
    ) -> DatasetDict:
        """Apply event mapping transformation."""
        print(">> Applying event mapping (train split).")
        event_mapper = XCEventMapping()
        self.raw_dataset["train"] = self.raw_dataset["train"].map(
            event_mapper,
            remove_columns=["audio", "audio_in"],
            batched=True,
            batch_size=self.config.batch_size,
            desc="Train event mapping",
        )
        return self.raw_dataset
    
    def smart_sample(
        self
    ) -> DatasetDict:
        """Apply smart sampling to balance class and event distribution."""
        print(
            f">> Smart sampling (class_limit={self.config.class_limit}, "
            f"event_limit={self.config.event_limit})."
        )
        self.raw_dataset["train"] = smart_sampling(
            dataset=self.raw_dataset["train"],
            label_name="ebird_code",
            class_limit=self.config.class_limit,
            event_limit=self.config.event_limit,
        )
        return self.raw_dataset
    
    def one_hot_encode(
        self
    ) -> DatasetDict:
        """Convert labels to one-hot encoding."""
        print(">> One-hot encoding labels.")
        num_classes = len(self.raw_dataset["train"].features["ebird_code"].names)
        class_list = self.raw_dataset["train"].features["ebird_code"].names
        multilabel = Sequence(ClassLabel(names=class_list))
        
        # self.raw_dataset["train"] = self.raw_dataset["train"].cast_column("labels", multilabel)
        for split in ["train", "test_5s"]:
            self.raw_dataset[split] = self.raw_dataset[split].map(
                classes_one_hot,
                batched=True,
                batch_size=self.config.batch_size,
                load_from_cache_file=False,
                desc=f"One-hot-encoding {split} labels.",
                fn_kwargs={"num_classes": num_classes},
            ).cast_column("labels", multilabel)
        return self.raw_dataset
    
    def train_test_split(
        self
    ) -> DatasetDict:
        """Split training data into train and validation sets."""
        print(
            f">> Splitting train into train/valid "
            f"({self.config.train_test_split*100:.0f}% test)."
        )
        xc_ds = self.raw_dataset["train"].train_test_split(
            test_size=self.config.train_test_split,
            stratify_by_column="ebird_code",
        )
        self.raw_dataset["train"] = xc_ds["train"]
        self.raw_dataset["valid"] = xc_ds["test"]
        return self.raw_dataset
    
    def create_audio_dataset(
        self
    ) -> AudioDataset:
        """Wrap processed dataset in AudioDataset for PyHa compatibility."""
        print(">> Creating AudioDataset.")
        self.audio_dataset = AudioDataset(
            {
                "train": self.raw_dataset["train"],
                "valid": self.raw_dataset["valid"],
                "test": self.raw_dataset["test_5s"],
            },
            f"BirdSet-{self.config.region}",
        )
        return self.audio_dataset
    
    

    def apply_transforms(self) -> AudioDataset:
        """Apply event decoding and spectrogram preprocessing transforms."""
        print(">> Applying event decoder transform.")
        event_decoder = EventDecoding(
            min_len=0,
            max_len=self.config.max_event_length,
            sample_rate=self.config.sampling_rate,
        )

        print(">> Applying spectrogram preprocessing transform.")
        augmentations = self._build_augmentations()

        class AugmentedTransform:
            def __init__(self, augmentations):
                self.augmentations = augmentations

            def __call__(self, audio, sample_rate, label):
                if self.augmentations:
                    augmented_audio, label = self.augmentations(audio, sample_rate, label)
                else:
                    augmented_audio = audio
                return augmented_audio, label

        class ChainedTransform:
            def __init__(self, *transforms):
                self.transforms = transforms

            def __call__(self, batch):
                for transform in self.transforms:
                    batch = transform(batch)
                return batch

        augmenter = AugmentedTransform(augmentations) if augmentations else None
        train_preprocessor = BirdSetMelSpectrogramPreprocessor(
            augment=augmenter,
            chunking=self.chunking,
        )
        test_preprocessor = BirdSetMelSpectrogramPreprocessor()

        train_transform = ChainedTransform(event_decoder, train_preprocessor)
        eval_transform = ChainedTransform(event_decoder, test_preprocessor)

        self.audio_dataset["train"].set_transform(train_transform)
        self.audio_dataset["valid"].set_transform(eval_transform)
        self.audio_dataset["test"].set_transform(test_preprocessor)



        return self.audio_dataset
    
    def _build_augmentations(self) -> ComposeAudioLabel:
        """Build augmentation pipeline based on AugmentationConfig."""
        augmentation_list = []
        aug_config = self.augmentation_config
        
        # Add Background Noise augmentation
        if aug_config.enable_background_noise:
            if aug_config.background_noise_path is None:
                raise ValueError(
                    "background_noise_path must be set when enable_background_noise=True"
                )
            print(f"   - Adding BackgroundNoise augmentation from {aug_config.background_noise_path}")
            augmentation_list.append(
                AddBackgroundNoise(
                    sounds_path=aug_config.background_noise_path,
                    min_snr_db=aug_config.background_noise_min_snr_db,
                    max_snr_db=aug_config.background_noise_max_snr_db,
                    p=aug_config.background_noise_p,
                )
            )
        
        # Add Gain augmentation
        if aug_config.enable_gain:
            print("   - Adding Gain augmentation")
            augmentation_list.append(
                Gain(
                    min_gain_db=aug_config.gain_min_gain_db,
                    max_gain_db=aug_config.gain_max_gain_db,
                    p=aug_config.gain_p,
                )
            )
        
        # Add MixItUp augmentation
        if aug_config.enable_mixitup:
            print("   - Adding MixItUp augmentation")
            augmentation_list.append(
                MixItUp(
                    dataset_ref=self.audio_dataset["train"],
                    min_snr_db=aug_config.mixitup_min_snr_db,
                    max_snr_db=aug_config.mixitup_max_snr_db,
                    p=aug_config.mixitup_p,
                )
            )
        
        # Add Polarity Inversion augmentation
        if aug_config.enable_polarity_inversion:
            print("   - Adding PolarityInversion augmentation")
            augmentation_list.append(
                PolarityInversion(p=aug_config.polarity_inversion_p)
            )
        
        if augmentation_list:
            return ComposeAudioLabel(augmentation_list)
        else:
            print("   - No augmentations enabled")
            return None
    
    def process_full(self) -> AudioDataset:
        """Execute the complete data pipeline."""
        print("\n" + "="*60)
        print("BIRDSET DATA PIPELINE")
        print("="*60 + "\n")
        
        self.load_raw()
        self.add_columns()
        self.cast_audio()
        self.extract_splits()
        if self.chunking == "detected_event_chunking":
            self.event_mapping()
            self.smart_sample()
        self.one_hot_encode()
        self.train_test_split()
        self.create_audio_dataset()
        self.apply_transforms()
        
        print("\n>> Data pipeline complete!")
        return self.audio_dataset
