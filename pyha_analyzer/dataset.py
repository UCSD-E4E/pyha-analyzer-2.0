from datasets import DatasetDict, ClassLabel
from .constants import DEFAULT_COLUMNS

# TODO Add required splits
class AudioDataset(DatasetDict):
    def __init__(self, ds: DatasetDict, provenance: str):
        # TODO Feature Checker
        self.provenance = provenance
        self.validate_format(ds)
        super().__init__(ds)

    def get_provenance(self) -> str:
        return self.provenance

    def validate_format(self, ds: DatasetDict):
        for split in ds.keys():
            dataset = ds[split]
            for column in DEFAULT_COLUMNS:
                assert column in dataset.features, (
                    f"The column `{column}` is missing from dataset split `{split}`. Required by system"
                )

    def get_number_species(self):
        return self["train"].features["labels"].num_classes 

    def get_class_labels(self):
        """
        Returns a new ClassLabel Object to make mapping easier between datasets
        """
        return ClassLabel(names=self["train"].features["labels"].names)


## TODO: Features to add that maybe useful
##  Summary Statistics System
##  Audio Player for demos?
##  Concatenate System (might be built into DatasetDict)
