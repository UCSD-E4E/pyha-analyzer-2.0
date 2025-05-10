## Trans Rights
from abc import ABC, abstractmethod
from .. import AudioDataset
from pathlib import Path


class DefaultExtractor(ABC):
    @abstractmethod
    def __init__(self, 
                 extractor_name: str):
        self.name = extractor_name

    @abstractmethod
    def __call__(self) -> AudioDataset:
        pass

    def get_provenance(self) -> str:
        return f"Extractor: {self.name}"


class FolderExtractor(DefaultExtractor):
    @abstractmethod
    def __call__(self, data_dir, meta_dir) -> AudioDataset:
        pass

    def verify_directories(self, data_dir, meta_path) -> bool:
        """Verify that the data directory and metadata file exist and are valid.
        Raises:
            FileNotFoundError: If the data directory or metadata file does not exist.
        """
        data_dir = Path(data_dir)
        meta_path = Path(meta_path)
        directory_exists = data_dir.exists() and data_dir.is_dir()
        meta_exists = meta_path.exists() and meta_path.is_file()
        if not directory_exists:
            raise FileNotFoundError(f"Directory {data_dir} does not exist.")
        if not meta_exists:
            raise FileNotFoundError(f"Metadata file not found in {data_dir}.")
        return True
