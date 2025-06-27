from .preprocessors import PreProcessorBase
from audiomentations import Compose, AddBackgroundNoise #TODO BaseWaveformTransform Didn't exist here
from audiomentations.core.transforms_interface import BaseWaveformTransform
import random
import numpy as np
from copy import copy

class AudiomentationBasePreprocessor(PreProcessorBase):
    def __init__(
        self,
        name: str,
        augmentation
    ):
        self.augmentation = augmentation
        super().__init__(name=name)

    def __call__(self, batch):
        result = self.augmentation(batch)
        return result
    
class AudioLabelPreprocessor(PreProcessorBase, BaseWaveformTransform):
    """
    AudioLabelPreprocessor: Audiomentions like augmentation but has extra parameters for changing the label

    USE CASES: No-Call Mixing, Mixitup, etc

    TODO: This should inherit from Audiomentions API, not PreProcessorBase
    """
    def __init__(
        self,
        name: str,
        augmentation
    ):
        self.augmentation = augmentation
        super().__init__(name=name)

    def __call__(self, data, sr, label):
        result = self.augmentation(data, sr, label)
        return result, label
    
class MixItUp(AudioLabelPreprocessor):
    def __init__(self, dataset_ref, p=0.7, **augmentation_kwargs):
        self.dataset_ref = copy(dataset_ref) # Should be the same split as the dataset. 
                                             # Should be the pre online processor dataset
        self.p=p
        self.augmentation_kwargs = augmentation_kwargs
        super().__init__("MixItUp", None)

    def __call__(self, audio, sr, labels):
        if random.random() < self.p:
            # Preprare Mixing
            mixed_row = self.dataset_ref[random.randint(0, self.dataset_ref.shape[0]-1)] #believe its inclusive
            mixer = AddBackgroundNoise(sounds_path=mixed_row["filepath"], p=1.0, **self.augmentation_kwargs)
            new_labels = mixed_row["labels"]
            
            #Mix data
            audio = mixer(audio, sr)
            labels = np.clip(np.array(new_labels) + np.array(labels), 0, 1)

        return audio, labels



class AudiomentationCompositePreprocessor(PreProcessorBase):
    """
    Composite augmentation preprocessor class, wraps audiomentations.Compose
    """

    def __init__(self, 
                #  augmentation_kwargs: list[dict], # Necessary to have list of kwargs for each augmentation? 
                 augmentations: list, # pre-defined, pre-initiaed augmentations
                 p: float = 1.0,
                 shuffle: bool = False):
        self.p = p
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.num_augmentations = len(augmentations)
        self.compose = Compose(self.augmentations, p=self.p, shuffle=self.shuffle)
        super().__init__(
            name=f"AudiomentationCompose({self.num_augmentations} augmentations)")

    def __call__(self, batch):
        result = self.compose(batch)
        return result
    
class ComposeAudioLabel(PreProcessorBase, Compose):
    """
    Composite augmentation preprocessor class, wraps audiomentations.Compose

    Allows for changing the label of a data point as needed
        EX: 
    """

    def __init__(self, 
                #  augmentation_kwargs: list[dict], # Necessary to have list of kwargs for each augmentation? 
                 augmentations: list, # pre-defined, pre-initiaed augmentations
                 p: float = 1.0,
                 shuffle: bool = False):
        self.p = p
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.transforms = augmentations
        self.num_augmentations = len(augmentations)
        #self.compose = Compose(self.augmentations, p=self.p, shuffle=self.shuffle)
        PreProcessorBase.__init__(self, 
            name=f"ComposeAudioLabelPreprocessor({self.num_augmentations} augmentations)")

    def __call__(self, data, sr, label):
        augmentation = self.augmentations.copy()
        if random.random() < self.p:
            if self.shuffle:
                random.shuffle(augmentation)
            for augmentation in self.augmentations:
                if isinstance(augmentation, AudioLabelPreprocessor):
                    data, label = augmentation(data, sr, label)
                else:
                    data = augmentation(data, sr)
        return data, label
    
class MixUpPreprocessor(AudiomentationBasePreprocessor):
    """
    Mixup augmentation implemented using AddBackgroundNoise
    """

    def __init__(self, 
                 augmentation_kwargs: dict,
                 ):
        self.augmentation_kwargs = augmentation_kwargs
        super().__init__(name="MixUp", 
                         augmentation=AddBackgroundNoise(**self.augmentation_kwargs))

def compose_preprocessors(preprocessors: list[AudiomentationBasePreprocessor],
                          p: float = 1.0,
                          shuffle: bool = False):
    """
    Compose a list of augmentation preprocessors into a single composite preprocessor.
    Args:
        preprocessors (list): List of preprocessors to compose
        p (float): Probability of applying the composite augmentation
        shuffle (bool): Whether to shuffle the order of augmentations with every application
    """
    augmentations_list = []
    for preprocessor in preprocessors:
        augmentations_list.append(preprocessor.augmentation)
    result = AudiomentationCompositePreprocessor(
        augmentations=augmentations_list,
        p=p,
        shuffle=shuffle
    )   
    return result
