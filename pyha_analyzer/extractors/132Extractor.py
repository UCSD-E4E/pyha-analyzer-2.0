from pathlib import Path
from datasets import load_dataset
from pyha_analyzer.extractors.defaultExtractors import FolderExtractor
from pyha_analyzer.dataset import AudioDataset

class Peru132Extractor(FolderExtractor):
    def __init__(self, 
                 data_dir: str):
        meta_path = Path(data_dir) / "metadata.csv"
        super().__init__("Peru132", data_dir, meta_path)
    
    def __call__(self):
        ds = load_dataset("audiofolder", data_dir=self.data_dir)
        return AudioDataset(ds, self.name)
    
if __name__ == "__main__":
    extractor = Peru132Extractor("data/Peru132")
    dataset = extractor()
    print(dataset)
    print(dataset.get_provedence())