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
    ): 
        self.duration = duration
        self.augment = augment
        self.spectrogram_augments = spectrogram_augments

        # Below parameter defaults from https://arxiv.org/pdf/2403.10380 pg 25
        self.n_fft=n_fft
        self.hop_length=hop_length 
        self.power=power
        self.n_mels=n_mels

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

            # Audio Based Augmentations
            if self.augment is not None:
               y, label = self.augment(y, sr, label)


            pillow_transforms = transforms.ToPILImage()
            
            mels = np.array(
                pillow_transforms(
                    librosa.feature.melspectrogram(
                        y=y[start : start + (sr * self.duration)], sr=sr,
                        n_fft=self.n_fft, 
                        hop_length=self.hop_length, 
                        power=self.power, 
                        n_mels=self.n_mels, 
                    )
                ),
                np.float32)[np.newaxis, ::] / 255

            if self.spectrogram_augments is not None:
                mels = self.spectrogram_augments(mels)

            new_audio.append(mels)
            new_labels.append(label)
    
        batch["audio_in"] = new_audio
        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)
        return batch

#Rather than choosing a random 5 second chunk and converting that to mel spectogram, this tries to convert all the consecutive 5 second chunks to mel spectograms to get more data. This is useful for extending Williams et al dataset
class MelSpectrogramPreprocessorsNew(PreProcessorBase):
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
    ): 
        self.duration = duration
        self.augment = augment
        self.spectrogram_augments = spectrogram_augments

        # Below parameter defaults from https://arxiv.org/pdf/2403.10380 pg 25
        self.n_fft=n_fft
        self.hop_length=hop_length 
        self.power=power
        self.n_mels=n_mels

        super().__init__(name="MelSpectrogramPreprocessorsNew")

    def __call__(self, batch):
        new_audio = []
        new_labels = []

        for item_idx in range(len(batch["audio"])):
            label = batch["labels"][item_idx]
            source_path = batch["audio"][item_idx]["path"]
            y, sr = librosa.load(path=source_path)

            chunk_samples = sr * self.duration
            #file_lower = source_path.lower()

            # Case 1: WILLIAMS (get all 5s chunks, pad to multiple of 5s)
            if "Williams_et_al_2024" in source_path:
                total_len = y.shape[-1]
                remainder = total_len % chunk_samples
                if remainder != 0:
                    pad_len = chunk_samples - remainder
                    y = np.pad(y, (0, pad_len))

                num_chunks = len(y) // chunk_samples

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_samples
                    end = start + chunk_samples
                    chunk = y[start:end]

                    if self.augment is not None:
                        chunk, chunk_label = self.augment(chunk, sr, label)
                    else:
                        chunk_label = label

                    mel = librosa.feature.melspectrogram(
                        y=chunk, sr=sr,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        power=self.power,
                        n_mels=self.n_mels,
                    )

                    pillow_transforms = transforms.ToPILImage()
                    mel_image = np.array(pillow_transforms(mel), dtype=np.float32)[np.newaxis, ::] / 255.

                    if self.spectrogram_augments is not None:
                        mel_image = self.spectrogram_augments(mel_image)

                    new_audio.append(mel_image)
                    new_labels.append(chunk_label)
                    # new_metadata.append({
                    #     "source_path": source_path,
                    #     "start_time_sec": chunk_idx * self.duration,
                    #     "end_time_sec": (chunk_idx + 1) * self.duration
                    # })

            # Case 2: Paola, or just anyone not Williams
            else:
                if y.shape[-1] < chunk_samples:
                    y = np.pad(y, (0, chunk_samples - y.shape[-1]))
                    start = 0
                else:
                    start = np.random.randint(0, y.shape[-1] - chunk_samples + 1)

                chunk = y[start: start + chunk_samples]

                if self.augment is not None:
                    chunk, label = self.augment(chunk, sr, label)

                mel = librosa.feature.melspectrogram(
                    y=chunk, sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    power=self.power,
                    n_mels=self.n_mels,
                )

                pillow_transforms = transforms.ToPILImage()
                mel_image = np.array(pillow_transforms(mel), dtype=np.float32)[np.newaxis, ::] / 255.

                if self.spectrogram_augments is not None:
                    mel_image = self.spectrogram_augments(mel_image)

                new_audio.append(mel_image)
                new_labels.append(label)
                # new_metadata.append({
                #     "source_path": source_path,
                #     "start_time_sec": start / sr,
                #     "end_time_sec": (start + chunk_samples) / sr
                # })

        batch["audio_in"] = new_audio
        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)
        #batch["chunk_metadata"] = new_metadata
        return batch
