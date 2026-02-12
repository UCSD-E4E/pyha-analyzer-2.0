import librosa
import numpy as np
import torchvision.transforms as transforms

from .preprocessors import PreProcessorBase


def one_hot_encode(labels, classes):
    one_hot = np.zeros((len(labels), len(classes)))
    for i in range(len(labels)):
        for label in labels[i]:
            one_hot[i, label] = 1
    return one_hot

class MelSpectrogramPreprocessors(PreProcessorBase):
    def __init__(
        self,
        duration=5,
        augment=None,
        spectrogram_augments=None,
        class_list=[],
        n_fft=2048, 
        hop_length=256, 
        power=2.0, 
        n_mels=256,
        dataset_ref=None,
        n_views:int = 1
    ): 
        self.duration = duration
        self.augment = augment
        self.spectrogram_augments = spectrogram_augments

        # Below parameter defaults from https://arxiv.org/pdf/2403.10380 pg 25
        self.n_fft=n_fft
        self.hop_length=hop_length 
        self.power=power
        self.n_mels=n_mels

        self.n_views = n_views #new multiview param

        super().__init__(name="MelSpectrogramPreprocessor")

    def __call__(self, batch):
        new_audio = []
        new_labels = []
        for item_idx in range(len(batch["audio"])):
            label = batch["labels"][item_idx]
            y, sr = librosa.load(path=batch["audio"][item_idx]["path"])
            
            # Select a random 5 second window if not given a 5 second window
            # Padd if less than 5 seconds
            start = 0
            if y.shape[-1] > (sr * self.duration):
                start = np.random.randint(0, y.shape[-1] - (sr * self.duration))
            else:
                y = np.pad(y, (sr * self.duration) - y.shape[-1])

            # ---- MULTI-VIEW LOGIC ----
            if self.augment is not None and self.n_views > 1:
                # print("more than 1 view detected, using contrastive loss")
                # 1) Apply label-changing augmentations ONCE to set label consistently.
                #    We do this by applying full self.augment one time.
                base_audio = y.copy()
                base_label = label
                base_audio, base_label = self.augment(base_audio, sr, base_label)

                # 2) Generate multiple views from base_audio using ONLY audio-only augments
                #    (exclude AudioLabelPreprocessor like MixItUp).
                views = []

                # If the augment is ComposeAudioLabel, it has .augmentations list.
                # We'll apply audio-only transforms from that list.
                audio_only_augments = None
                if hasattr(self.augment, "augmentations"):
                    audio_only_augments = []
                    for aug in self.augment.augmentations:
                        # Exclude label-changing augmenters (e.g., MixItUp) without importing their base class.
                        # MixItUp in your code has a dataset_ref attribute and changes labels.
                        if hasattr(aug, "dataset_ref"):
                            continue
                        # As a fallback, exclude anything whose class name suggests label ops
                        if aug.__class__.__name__ in ("MixItUp", "ComposeAudioLabel", "AudioLabelPreprocessor"):
                            continue
                        audio_only_augments.append(aug)

                for _ in range(self.n_views):
                    v_audio = base_audio.copy()

                    # apply audio-only augments with their own internal randomness
                    if audio_only_augments is not None:
                        for aug in audio_only_augments:
                            v_audio = aug(v_audio, sr)

                    v_mel = self._compute_mel(v_audio, sr, start)  # [1,H,W]
                    views.append(v_mel)

                # stack -> [V, 1, H, W]
                mels = np.stack(views, axis=0).astype(np.float32)
                new_audio.append(mels)
                new_labels.append(base_label)

            else:
                # ---- ORIGINAL SINGLE-VIEW PATH ----
                if self.augment is not None:
                    y_aug = y.copy()
                    y_aug, label = self.augment(y_aug, sr, label)
                else:
                    y_aug = y

                mels = self._compute_mel(y_aug, sr, start)  # [1,H,W]
                new_audio.append(mels)
                new_labels.append(label)
    
        batch["audio_in"] = new_audio
        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)
        return batch
    def _compute_mel(self, y, sr, start):
        """
        Computes a single mel-spectrogram view.
        Returns: np.ndarray float32 with shape [1, H, W] (same format you used before).
        """
        pillow_transforms = transforms.ToPILImage()

        mels = np.array(
            pillow_transforms(
                librosa.feature.melspectrogram(
                    y=y[start : start + (sr * self.duration)],
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    power=self.power,
                    n_mels=self.n_mels,
                )
            ),
            np.float32
        )[np.newaxis, ::] / 255.0  # shape [1, H, W], float32

        if self.spectrogram_augments is not None:
            mels = self.spectrogram_augments(mels)

        return mels

