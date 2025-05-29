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

def find_start_of_segmentat(start, end, meta_duration=8, duration=5):
    event_duration = end - start
    if event_duration < meta_duration:
        pad_s = (meta_duration - duration) / 2
        end = end + pad_s
        start = start - pad_s
        if start < 0:
            start = 0
            end = meta_duration

    actual_start = np.random.uniform(start, end-duration)
    if actual_start < 0:
        actual_start = 0
    # print(actual_start)
    return actual_start

class WaveformPreprocessors(PreProcessorBase):
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
            
            # If the system has detected events,
            # Randomly choose one
            # create a 8 second window around it and randomly sample 5 seconds from it
            events = batch["detected_events"][item_idx]
            
            #TODO THIS SHOULD BE OFFLINE
            start = 0
            if events is not None and len(events) > 0: 
                # print(len(events), events)
                event = events[np.random.choice(range(len(events)), 1)[0]]
                start = find_start_of_segmentat(event[0], event[1], meta_duration=self.duration + 3, duration=self.duration)

            # print(start)

            # Handle out of bound issues
            end_sr = int(start * sr) + int(sr * self.duration)
            if y.shape[-1] <= end_sr:
                y = np.pad(y, end_sr - y.shape[-1])

            # Audio Based Augmentations
            if self.augment != None:
               y, label = self.augment(y, sr, label)

            new_audio.append(y)
            new_labels.append(label)
    
        batch["audio_in"] = new_audio
        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)
        batch["detected_events"] = np.array(new_labels, dtype=np.float32) #why is this replacing labels?
        # print(np.array(new_audio).shape, np.array(new_labels, dtype=np.float32).shape)
        return batch




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
            
            # If the system has detected events,
            # Randomly choose one
            # create a 8 second window around it and randomly sample 5 seconds from it
            events = batch["detected_events"][item_idx]
            
            #TODO THIS SHOULD BE OFFLINE
            start = 0
            if events is not None and len(events) > 0: 
                # print(len(events), events)
                event = events[np.random.choice(range(len(events)), 1)[0]]
                start = find_start_of_segmentat(event[0], event[1], meta_duration=self.duration + 3, duration=self.duration)

            # print(start)

            # Handle out of bound issues
            end_sr = int(start * sr) + int(sr * self.duration)
            if y.shape[-1] <= end_sr:
                y = np.pad(y, end_sr - y.shape[-1])

            # Audio Based Augmentations
            if self.augment != None:
               y, label = self.augment(y, sr, label)


            pillow_transforms = transforms.ToPILImage()
            
            mels = np.array(
                pillow_transforms(
                    librosa.feature.melspectrogram(
                        y=y[int(start * sr) : end_sr], sr=sr,
                        n_fft=self.n_fft, 
                        hop_length=self.hop_length, 
                        power=self.power, 
                        n_mels=self.n_mels, 
                    )
                ),
                np.float32)[np.newaxis, ::] / 255

            if self.spectrogram_augments is not None:
                mels = self.spectrogram_augments(mels)

            # print(mels.shape, int(start * sr), y.shape)
            new_audio.append(mels)
            new_labels.append(label)
    
        batch["audio_in"] = new_audio
        batch["audio"] = new_audio
        batch["labels"] = np.array(new_labels, dtype=np.float32)
        batch["detected_events"] = np.array(new_labels, dtype=np.float32) #why is this replacing labels?
        # print(np.array(new_audio).shape, np.array(new_labels, dtype=np.float32).shape)
        return batch
