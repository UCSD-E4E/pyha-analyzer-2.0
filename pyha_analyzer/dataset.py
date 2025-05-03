from datasets import DatasetDict

class AudioDataset(DatasetDict):
    def __init__(self,ds:DatasetDict, provedence:str):

        #TODO Feature Checker
        self.provedence = provedence
        super().__init__(ds)

    def get_provedence(self) -> str:
        return self.provedence
    
## TODO: Features to add that maybe useful
##  Summary Statistics System
##  Audio Player for demos?
##  Concatenate System (might be built into DatasetDict)