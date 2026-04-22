import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from .preprocessors import PreProcessorBase


class FastMelSpectrogramPreprocessor(PreProcessorBase):
    """On-the-fly mel-spectrogram preprocessor backed by torchaudio.

    Faster than MelSpectrogramPreprocessors in three ways:
    - torchaudio.load() skips librosa's default 22050 Hz resample step
    - MelSpectrogram transform is initialized once and reused each call
      (torch FFT vs librosa's numpy FFT)
    - Min-max normalization replaces the float→uint8 (PIL) →float roundtrip,
      which both loses precision and allocates an extra buffer
    """

    def __init__(
        self,
        duration: int = 5,
        augment=None,
        spectrogram_augments=None,
        class_list: list = [],
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 256,
        power: float = 2.0,
        n_mels: int = 256,
    ):
        self.duration = duration
        self.augment = augment
        self.spectrogram_augments = spectrogram_augments
        self.sample_rate = sample_rate
        self.n_samples = sample_rate * duration

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            n_mels=n_mels,
        )
        # Cache resamplers keyed by source sample rate so we don't re-instantiate
        # for every file when the dataset has a uniform native rate.
        self._resamplers: dict[int, T.Resample] = {}

        super().__init__(name="FastMelSpectrogramPreprocessor")

    def _get_resampler(self, source_sr: int) -> T.Resample:
        if source_sr not in self._resamplers:
            self._resamplers[source_sr] = T.Resample(source_sr, self.sample_rate)
        return self._resamplers[source_sr]

    def __call__(self, batch: dict) -> dict:
        new_audio = []
        new_labels = []

        for item_idx in range(len(batch["audio"])):
            label = batch["labels"][item_idx]

            # torchaudio.load returns (Tensor[C, T], int) — no resampling by default
            waveform, sr = torchaudio.load(batch["audio"][item_idx]["path"])

            # Mix down to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.sample_rate:
                waveform = self._get_resampler(sr)(waveform)

            # Crop to random window or pad to target length
            waveform = waveform.squeeze(0)
            if waveform.shape[-1] >= self.n_samples:
                start = torch.randint(0, waveform.shape[-1] - self.n_samples + 1, (1,)).item()
                waveform = waveform[start: start + self.n_samples]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.n_samples - waveform.shape[-1]))

            # Audio augmentations (audiomentations expects numpy + sample rate)
            if self.augment is not None:
                waveform_np = waveform.numpy()
                waveform_np, label = self.augment(waveform_np, self.sample_rate, label)
                waveform = torch.from_numpy(waveform_np)

            # Mel spectrogram: (1, n_mels, time)
            mel = self.mel_transform(waveform.unsqueeze(0))

            # Normalize to [0, 1]
            mel_min = mel.min()
            mel_max = mel.max()
            mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)

            mel = mel.numpy().astype(np.float32)

            if self.spectrogram_augments is not None:
                mel = self.spectrogram_augments(mel)

            new_audio.append(mel)
            new_labels.append(label)

        batch["audio_in"] = new_audio
        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)
        return batch
