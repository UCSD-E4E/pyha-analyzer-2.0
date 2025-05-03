from datasets import DatasetDict

class AudioDataset(DatasetDict):
    def __init__(self,ds:DatasetDict, provenance:str):

        #TODO Feature Checker
        self.provenance = provenance
        super().__init__(ds)

    def get_provenance(self) -> str:
        return self.provenance
    
## TODO: Features to add that maybe useful
##  Summary Statistics System
##  Audio Player for demos?
##  Concatenate System (might be built into DatasetDict)