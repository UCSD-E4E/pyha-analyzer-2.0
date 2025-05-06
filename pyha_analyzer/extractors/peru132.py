from pathlib import Path

from datasets import load_dataset
import pandas as pd

from pyha_analyzer.extractors.defaultExtractors import FolderExtractor
from pyha_analyzer.dataset import AudioDataset


class Peru132Extractor(FolderExtractor):
    def __init__(
        self,
        offset_col: str = "OFFSET",
        duration_col: str = "DURATION",
        file_name_col: str = "IN FILE",
        label_col: str = "MANUAL ID",
    ):
        self.offset_col = offset_col
        self.duration_col = duration_col
        self.file_name_col = file_name_col
        self.label_col = label_col

        super().__init__("Peru132")

    def __call__(self, data_dir):
        self.verify_directories(data_dir)
        meta_path = Path(data_dir) / "metadata.csv"
        self.process_metadata(meta_path)

        ds = load_dataset("audiofolder", data_dir=data_dir)
        return AudioDataset(ds, self.get_provenance())

    def verify_directories(self, data_dir):
        meta_path = Path(data_dir) / "metadata.csv"
        return super().verify_directories(data_dir, meta_path)

    def process_metadata(self, meta_path):
        return pd.read_csv(meta_path)


if __name__ == "__main__":
    extractor = Peru132Extractor("data/Peru132")
    dataset = extractor()
    print(dataset)
    print(dataset.get_provedence())

# if going_to_crash:
#    dont()
