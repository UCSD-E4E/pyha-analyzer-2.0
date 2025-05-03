from pathlib import Path
from datasets import load_dataset
from pyha_analyzer.extractors.defaultExtractors import FolderExtractor
from pyha_analyzer.dataset import AudioDataset

class Peru132Extractor(FolderExtractor):
    def __init__(self):
        super().__init__("Peru132")
    
    def __call__(self, data_dir):
        self.verify_directories(data_dir)
        ds = load_dataset("audiofolder", data_dir=data_dir)
        return AudioDataset(ds, self.get_provenance())
    
    def verify_directories(self, data_dir):
        meta_path = Path(data_dir) / "metadata.csv"
        return super().verify_directories(data_dir, meta_path)
    
if __name__ == "__main__":
    extractor = Peru132Extractor("data/Peru132")
    dataset = extractor()
    print(dataset)
    print(dataset.get_provedence())