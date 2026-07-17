from pathlib import Path

from datasets import load_dataset, Sequence, ClassLabel, Audio
import pandas as pd
import ast

from pyha_analyzer.extractors.defaultExtractors import FolderExtractor
from pyha_analyzer.dataset import AudioDataset

import numpy as np

from copy import copy


def one_hot_encode(labels, classes):
    one_hot = np.zeros(len(classes))
    for label in labels:
        one_hot[label] = 1
    return np.array(one_hot, dtype=float)

def one_hot_encode_ds_wrapper(row, class_list):
    row["labels"] = one_hot_encode(row["labels"], class_list)
    return row

class SamExtractor(FolderExtractor):
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

        super().__init__("Sam")

    def __call__(self, data_dir, split="test"):
        

        ds = load_dataset("audiofolder", data_dir=data_dir)

        birdset = load_dataset("DBD-research-group/BirdSet", "HSN", trust_remote_code=True)

        print(birdset)


        print('------------------')
        print(ds["train"][0])


        class_list = birdset["train"].features["ebird_code"].names
        mutlilabel_class_label =  Sequence(ClassLabel(names=class_list))

        # ds["test"] = ds["test"].add_column("audio_in", ds["test"]["audio"])
        ds["train"] = ds["train"].cast_column("audio", Audio(decode=False))
        ds["train"] = ds["train"].map(lambda x: {"audio_in": x["audio"]})
        ds["train"] = ds["train"].rename_column("ebird_code_multilabel", "labels")

        ds["train"] = ds["train"].map(lambda x: {"labels": ast.literal_eval(x["labels"])})

        ds["train"] = ds["train"].cast_column("audio", Audio(decode=False))

        ds["train"] = ds["train"].map(lambda x: {"filepath": x["audio"]["path"]})

        # ds["test"] = ds["test"].cast_column("audio", Audio(decode=False))
        ds["train"] = ds["train"].cast_column("audio", Audio(decode=False))

        print('------------------')
        # print(ds["test"][0])
        print(ds["train"][0])

        ds["train"] = ds["train"].map(
            lambda row: one_hot_encode_ds_wrapper(row, class_list)
            ).cast_column("labels", mutlilabel_class_label)
        

        print(ds)

        print(ds["train"][0])

        return AudioDataset(ds, self.get_provenance())

    def verify_directories(self, data_dir):
        meta_path = Path(data_dir) / "metadata.csv"
        return super().verify_directories(data_dir, meta_path)

    def process_metadata(self, meta_path):
        return pd.read_csv(meta_path)


if __name__ == "__main__":
    extractor = SamExtractor()
    dataset = extractor("/home/s.dalal.334/SAM/sam_audio_files/test/sam")
    print(dataset)
    print(dataset.get_provenance())

# if going_to_crash:
#    dont()