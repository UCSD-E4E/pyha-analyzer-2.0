from .defaultExtractors import DefaultExtractor
from datasets import load_dataset
from .. import AudioDataset

class Birdset(DefaultExtractor):
    def __init__(self):
        super().__init__("Birdset")

    def __call__(self, region):
        ds = load_dataset("DBD-research-group/BirdSet", region, trust_remote_code=True)
        #TODO Clean Features
        return AudioDataset(ds)