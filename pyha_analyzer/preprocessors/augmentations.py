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

class AudiomentationComposePreprocessor(PreProcessorBase):
    """
    Composite augmentation preprocessor class, wraps audiomentations.Compose
    """

    def __init__(self, 
                 augmentation_kwargs: list[dict], 
                 augmentations: list[BaseWaveformTransform],
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
    