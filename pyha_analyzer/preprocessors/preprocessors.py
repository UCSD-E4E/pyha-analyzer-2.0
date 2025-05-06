from abc import ABC, abstractmethod

"""
Base class for preprocessing

Could be online or offline, but always follows this class
"""

class PreProcessorBase(ABC):
    def init(self, *args, **kargs):
        pass

    @abstractmethod
    def __call__(self, batch):
        pass

class AudiomentionsExampleWrapper(PreProcessorBase):
    def init(self, augmentation_kwargs, ):
        self.augmentation_kwargs = augmentation_kwargs

    @abstractmethod
    def __call__(self, batch):
        pass