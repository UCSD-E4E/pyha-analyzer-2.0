from datasets import DatasetDict
from .constants import DEFAULT_COLUMNS

class AudioDataset(DatasetDict):
    def __init__(self,ds:DatasetDict, provenance:str):

        #TODO Feature Checker
        self.provenance = provenance
        self.validate_format(ds)
        super().__init__(ds)

    def get_provenance(self) -> str:
        return self.provenance
    
    def validate_format(self, ds:DatasetDict):
        for split in ds.keys():
            dataset = ds[split]
            for column in DEFAULT_COLUMNS:
                assert column in dataset.features, f"The column `{column}` is missing from dataset split `{split}`. Required by system"
    
## TODO: Features to add that maybe useful
##  Summary Statistics System
##  Audio Player for demos?
##  Concatenate System (might be built into DatasetDict)