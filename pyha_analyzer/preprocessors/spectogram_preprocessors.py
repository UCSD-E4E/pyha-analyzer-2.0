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
        class_list=[],
        n_fft=2048, 
        hop_length=256, 
        power=2.0, 
        n_mels=256,

    ): 
        self.duration = duration
        self.augment = augment
        self.class_list = class_list

        # Below parameter defaults from https://arxiv.org/pdf/2403.10380 pg 25
        self.n_fft=n_fft
        self.hop_length=hop_length 
        self.power=power
        self.n_mels=n_mels
        super().__init__(name="MelSpectrogramPreprocessor")

    def __call__(self, batch):
        new_audio = []
        for item_idx in range(len(batch["audio"])):
            y, sr = librosa.load(path=batch["audio"][item_idx]["path"])
            
            # Select a random 5 second window if not given a 5 second window
            # Padd if less than 5 seconds
            start = 0
            if y.shape[-1] > (sr * self.duration):
                start = np.random.randint(0, y.shape[-1] - (sr * self.duration))
            else:
                y = np.pad(y, (sr * self.duration) - y.shape[-1])

            # Audio Based Augmentations
            if self.augment != None:
               y = self.augment(y, sr)


            pillow_transforms = transforms.ToPILImage()
            new_audio.append(
                np.array(
                    pillow_transforms(
                        librosa.feature.melspectrogram(
                            y=y[start : start + (sr * self.duration)], sr=sr,
                            n_fft=self.n_fft, 
                            hop_length=self.hop_length, 
                            power=self.power, 
                            n_mels=self.n_mels, 
                        )
                    ),
                    np.float32,
                )[np.newaxis, ::]
                / 255
            )
        batch["audio_in"] = new_audio
        batch["audio"] = new_audio
        batch["labels"] = one_hot_encode(batch["labels"], self.class_list)

        # Handle mutlilabel formatting

        return batch
