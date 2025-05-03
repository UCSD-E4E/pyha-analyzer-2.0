## Trans Rights
from abc import ABC, abstractmethod
import typing
from .. import AudioDataset
from pathlib import Path

class DefaultExtractor(ABC):
    @abstractmethod
    def __init__(self, extractor_name:str):
        self.name = extractor_name
        
    @abstractmethod    
    def __call__(self) -> AudioDataset:
        pass

    def get_provendence(self) -> str:
        return f"Extractor: {self.name}"
       
class FolderExtractor(DefaultExtractor):
    @abstractmethod
    def __init__(self, 
                 extractor_name:str, 
                 data_dir: str,
                 meta_path: str):
        super().__init__(extractor_name)
        self.data_dir = Path(data_dir)
        self.meta_path = Path(meta_path)
        self.verify_directories()

    @abstractmethod
    def __call__(self, data_dir) -> AudioDataset:
        pass
    
    def verify_directories(self) -> bool:
        """Verify that the data directory and metadata file exist and are valid.
        Raises:
            FileNotFoundError: If the data directory or metadata file does not exist.
        """
        directory_exists = self.data_dir.exists() and self.data_dir.is_dir()
        meta_exists = self.meta_path.exists() and self.meta_path.is_file()
        if not directory_exists:
            raise FileNotFoundError(f"Directory {self.data_dir} does not exist.")
        if not meta_exists:
            raise FileNotFoundError(f"Metadata file not found in {self.data_dir}.")
    
    def get_data_dir(self) -> Path:
        return self.data_dir
    
    def get_meta_path(self) -> Path:
        return self.meta_path