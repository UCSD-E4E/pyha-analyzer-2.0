from .defaultExtractors import DefaultExtractor
from copy import copy
from datasets import load_dataset, ClassLabel, Sequence
from .. import AudioDataset
import numpy as np

def one_hot_encode(labels, classes):
    one_hot = np.zeros(len(classes))
    for label in labels:
        one_hot[label] = 1
    return np.array(one_hot, dtype=float)

def one_hot_encode_ds_wrapper(row, class_list):
    row["labels"] = one_hot_encode(row["labels"], class_list)
    return row

class Birdset(DefaultExtractor):
    def __init__(self):
        super().__init__("Birdset")

    def __call__(self, region):
        ds = load_dataset("DBD-research-group/BirdSet", region, trust_remote_code=True)
        class_list = ds["train"].features["ebird_code"].names
        mutlilabel_class_label =  Sequence(ClassLabel(names=class_list))

        for split in ["train", "test_5s"]:
            ds[split] = ds[split].add_column("audio_in", ds[split]["audio"])
            ds[split] = (
                ds[split]
                .add_column("labels", copy(ds[split]["ebird_code_multilabel"]))
                # .cast_column("labels", mutlilabel_class_label)
            )

            ds[split] = ds[split].map(
                lambda row: one_hot_encode_ds_wrapper(row, class_list)
                ).cast_column("labels", mutlilabel_class_label)

        xc_ds = ds["train"].train_test_split(
            test_size=0.2, stratify_by_column="ebird_code" #still works since not mutlilabel
        )
        return AudioDataset(
            {"train": xc_ds["train"], "valid": xc_ds["test"], "test": ds["test_5s"]},
            f"{self.get_provenance()}-{region}",
        )
