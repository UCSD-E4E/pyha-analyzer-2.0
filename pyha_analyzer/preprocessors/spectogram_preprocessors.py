import librosa
import numpy as np
import torchvision.transforms as transforms

from .preprocessors import PreProcessorBase


class MelSpectrogramPreprocessors(PreProcessorBase):
    def __init__(
        self, duration=5, augment=None
    ):  # TODO Suppose we pass data augmentations through here?
        self.duration = duration
        self.augment = augment
        super().__init__()

    def __call__(self, batch):
        new_audio = []
        for item_idx in range(len(batch["audio"])):
            y, sr = librosa.load(path=batch["audio"][item_idx]["path"])
            # padding yippeee
            start = 0
            if y.shape[-1] > (sr * self.duration):
                start = np.random.randint(0, y.shape[-1] - (sr * self.duration))
            else:
                y = np.pad(y, (sr * self.duration) - y.shape[-1])

            if self.augment != None:
               y = self.augment(y)

            pillow_transforms = transforms.ToPILImage()
            new_audio.append(
                np.array(
                    pillow_transforms(
                        librosa.feature.melspectrogram(
                            y=y[start : start + (sr * self.duration)], sr=sr
                        )
                    ),
                    np.float32,
                )[np.newaxis, ::]
                / 255
            )
        batch["audio_in"] = new_audio
        batch["audio"] = new_audio
        return batch
