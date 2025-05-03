from .defaultExtractors import DefaultExtractor
from datasets import load_dataset
from .. import AudioDataset

class Birdset(DefaultExtractor):
    def __init__(self):
        super().__init__("Birdset")

    def __call__(self, region):
        ds = load_dataset("DBD-research-group/BirdSet", region, trust_remote_code=True)
        xc_ds = ds["train"].train_test_split(test_size=0.2, stratify_by_column="ebird_code")
        return AudioDataset({
            "train": xc_ds["train"],
            "valid": xc_ds["test"],
            "test": ds["test_5s"]
        }, f"{self.get_provendence()}-{region}")