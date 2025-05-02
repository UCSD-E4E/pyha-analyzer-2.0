## Trans Rights
from abc import ABC, abstractmethod
import typing
from .. import AudioDataset

class DefaultExtractor(ABC):
    @abstractmethod
    def __init__(self, extractor_name:str):
        self.name = extractor_name
        
    @abstractmethod    
    def __call__(self) -> AudioDataset:
        pass

    def get_provendence(self) -> str:
        return f"Extractor: {self.name}"
       