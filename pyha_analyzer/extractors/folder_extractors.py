from defaultExtractors import DefaultExtractor
from datasets import load_dataset
from pyha_analyzer import AudioDataset
from abc import ABC, abstractmethod
from pathlib import Path

class FolderExtractor(DefaultExtractor):
    @abstractmethod
    def __init__(self):
        pass

    def __call__(self, data_dir):
        ds = load_dataset("audiofolder", data_dir=data_dir)
        return AudioDataset(ds)