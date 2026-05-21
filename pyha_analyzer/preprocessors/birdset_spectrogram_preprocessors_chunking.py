"""
BirdSet-style spectrogram preprocessing for PyHa experiments.

This module intentionally lives alongside the existing PyHa preprocessors instead
of replacing them. It keeps the PyHa batch contract:
    batch["audio"], batch["audio_in"], batch["labels"]
while matching the main BirdSet spectrogram preprocessing choices:
    load/resample mono audio, fixed-length crop/pad, mel power spectrogram,
    power-to-dB, optional pad/truncate resize, and mean/std normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import librosa
import numpy as np
import soundfile as sf

from .preprocessors import PreProcessorBase


@dataclass
class BirdSetSpectrogramConfig:
    duration: float = 5.0
    sample_rate: int = 32_000
    n_fft: int = 1024
    hop_length: int = 320
    power: float = 2.0
    n_mels: int = 128
    db_scale: bool = True
    db_ref: float = 1.0
    db_amin: float = 1e-10
    top_db: Optional[float] = 80.0
    target_height: Optional[int] = None
    target_width: Optional[int] = None
    normalize_spectrogram: bool = True
    mean: float = -4.268
    std: float = 4.569
    normalize_waveform: Optional[str] = None
    random_crop: bool = True


class BirdSetSpectrogramPreprocessorWithChunking(PreProcessorBase):
    """
    PyHa-compatible preprocessor using BirdSet-style spectrogram settings.

    Args:
        config: Parameter bundle matching BirdSet defaults.
        augment: Optional waveform augmentation callable with PyHa's existing
            signature: ``audio, sample_rate, label -> audio, label``.
        spectrogram_augments: Optional callable applied after spectrogram
            normalization. It receives and returns a numpy array shaped
            ``(1, n_mels, frames)``.
    """

    def __init__(
        self,
        config: BirdSetSpectrogramConfig | None = None,
        augment: Optional[Callable] = None,
        spectrogram_augments: Optional[Callable] = None,
    ):
        self.config = config or BirdSetSpectrogramConfig()
        self.augment = augment
        self.spectrogram_augments = spectrogram_augments
        super().__init__(name="BirdSetSpectrogramPreprocessor")

    def __call__(self, batch):
        new_audio = []
        new_labels = []

        for item_idx in range(len(batch["audio"])):
            label = batch["labels"][item_idx]
            audio_item = batch["audio"][item_idx]

            if isinstance(audio_item, dict) and "array" in audio_item:
                audio = self._normalize_waveform(audio_item["array"])
            else:
                audio = self._normalize_waveform(audio_item)
                
            audio = self._fixed_length_audio(audio_item["array"])

            if self.augment is not None:
                audio, label = self.augment(audio, self.config.sample_rate, label)

            spectrogram = self._mel_power_spectrogram(audio)
            if self.config.db_scale:
                spectrogram = self._power_to_db(spectrogram)

            spectrogram = self._resize_spectrogram(spectrogram)

            if self.config.normalize_spectrogram:
                spectrogram = (spectrogram - self.config.mean) / self.config.std

            spectrogram = spectrogram[np.newaxis, :, :].astype(np.float32)

            if self.spectrogram_augments is not None:
                spectrogram = self.spectrogram_augments(spectrogram)

            new_audio.append(spectrogram)
            new_labels.append(label)

        batch["audio_in"] = new_audio
        # Keep PyHa's current convention: model-required "audio" mirrors audio_in.
        batch["audio"] = new_audio
        batch["labels"] = np.asarray(new_labels, dtype=np.float32)
        return batch

    def _load_audio(self, audio_item) -> np.ndarray:
        path = audio_item["path"] if isinstance(audio_item, dict) else audio_item
        try:
            audio, sr = sf.read(path)
        except Exception:
            audio, sr = librosa.load(path, sr=None, mono=False)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            # soundfile returns (samples, channels); librosa.to_mono expects
            # (channels, samples).
            audio = librosa.to_mono(audio.T)

        if sr != self.config.sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.config.sample_rate
            )

        return audio.astype(np.float32, copy=False)

    def _fixed_length_audio(self, audio: np.ndarray) -> np.ndarray:
        target_samples = int(round(self.config.duration * self.config.sample_rate))

        if audio.shape[-1] > target_samples:
            if self.config.random_crop:
                start = np.random.randint(0, audio.shape[-1] - target_samples + 1)
            else:
                start = (audio.shape[-1] - target_samples) // 2
            audio = audio[start : start + target_samples]
        elif audio.shape[-1] < target_samples:
            pad = target_samples - audio.shape[-1]
            audio = np.pad(audio, (0, pad), mode="constant")

        return audio

    def _normalize_waveform(self, audio: np.ndarray) -> np.ndarray:
        mode = self.config.normalize_waveform
        if mode is None:
            return audio

        if mode == "instance_normalization":
            return (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)

        if mode == "instance_min_max":
            min_val = audio.min()
            max_val = audio.max()
            return 2 * ((audio - min_val) / (max_val - min_val + 1e-7)) - 1

        if mode == "instance_peak_normalization":
            audio = audio - audio.mean()
            peak = np.max(np.abs(audio))
            return audio if peak == 0 else (audio / peak) * 0.25

        raise ValueError(
            "normalize_waveform must be one of None, 'instance_normalization', "
            "'instance_min_max', or 'instance_peak_normalization'."
        )

    def _mel_power_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        return librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            power=self.config.power,
            n_mels=self.config.n_mels,
            fmin=0.0,
            fmax=self.config.sample_rate / 2,
            htk=True,
            norm=None,
        ).astype(np.float32)

    def _power_to_db(self, spectrogram: np.ndarray) -> np.ndarray:
        if self.config.db_amin <= 0:
            raise ValueError("db_amin must be strictly positive")

        ref = max(abs(self.config.db_ref), self.config.db_amin)
        log_spec = 10.0 * np.log10(np.maximum(spectrogram, self.config.db_amin))
        log_spec -= 10.0 * np.log10(ref)

        if self.config.top_db is not None:
            if self.config.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = np.maximum(log_spec, log_spec.max() - self.config.top_db)

        return log_spec.astype(np.float32)

    def _resize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        pad_value = -80.0 if self.config.db_scale else 0.0

        if self.config.target_height is not None:
            height_delta = self.config.target_height - spectrogram.shape[0]
            if height_delta > 0:
                spectrogram = np.pad(
                    spectrogram,
                    ((0, height_delta), (0, 0)),
                    mode="constant",
                    constant_values=pad_value,
                )
            elif height_delta < 0:
                spectrogram = spectrogram[: self.config.target_height, :]

        if self.config.target_width is not None:
            width_delta = self.config.target_width - spectrogram.shape[1]
            if width_delta > 0:
                spectrogram = np.pad(
                    spectrogram,
                    ((0, 0), (0, width_delta)),
                    mode="constant",
                    constant_values=pad_value,
                )
            elif width_delta < 0:
                spectrogram = spectrogram[:, : self.config.target_width]

        return spectrogram.astype(np.float32, copy=False)


