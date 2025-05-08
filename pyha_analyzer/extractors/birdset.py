from .defaultExtractors import DefaultExtractor
from copy import copy
from datasets import load_dataset, ClassLabel
from .. import AudioDataset


class Birdset(DefaultExtractor):
    def __init__(self):
        super().__init__("Birdset")

    def __call__(self, region):
        ds = load_dataset("DBD-research-group/BirdSet", region, trust_remote_code=True)
        class_label = ClassLabel(names=ds["train"].features["ebird_code"].names)

        for split in ds.keys():
            ds[split] = ds[split].add_column("audio_in", ds[split]["audio"])
            ds[split] = (
                ds[split]
                .add_column("labels", copy(ds[split]["ebird_code"]))
                .cast_column("labels", class_label)
            )

        xc_ds = ds["train"].train_test_split(
            test_size=0.2, stratify_by_column="ebird_code"
        )
        return AudioDataset(
            {"train": xc_ds["train"], "valid": xc_ds["test"], "test": ds["test_5s"]},
            f"{self.get_provenance()}-{region}",
        )
