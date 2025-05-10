from .preprocessors import PreProcessorBase
from audiomentations import Compose, BaseWaveformTransform, AddBackgroundNoise

class AudiomentationBasePreprocessor(PreProcessorBase):
    def __init__(
        self,
        name: str,
        augmentation: BaseWaveformTransform
    ):
        self.augmentation = augmentation
        super().__init__(name=name)

    def __call__(self, batch):
        result = self.augmentation(batch)
        return result

class AudiomentationCompositePreprocessor(PreProcessorBase):
    """
    Composite augmentation preprocessor class, wraps audiomentations.Compose
    """

    def __init__(self, 
                #  augmentation_kwargs: list[dict], # Necessary to have list of kwargs for each augmentation? 
                 augmentations: list[BaseWaveformTransform], # pre-defined, pre-initiaed augmentations
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
