from .defaultExtractors import DefaultExtractor
from datasets import ClassLabel, Sequence, Audio
from .. import AudioDataset
import os
from datasets import Dataset


def parse_config(config_path):
    metadata = {}
    with open(config_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                if ((key.strip()== "Device ID") or (key.strip() == "Sample rate (Hz)")):
                    metadata[key.strip()] = val.strip()
    return metadata


def extract_features(wav, label):
    if label==0:
        oneHotEncodedLabel = [0,1] #Non_Degraded_Reef
    else:
        oneHotEncodedLabel = [1,0] #Degraded_Reef

    return {
        "sample_rate": 48000,
        "labels": oneHotEncodedLabel,
        "filepath": str(wav.path),
        "audio": str(wav.path),
        "audio_in": {"array": str(wav.path), "sampling_rate": 48000},
    }


class CoralReef(DefaultExtractor):
    def __init__(self):
        super().__init__("CoralReef")

    def __call__(self, audio_path):
        all_data = []
        #audio_path= "/home/s.kamboj.400/unzipped-coral"
        for state in os.scandir(audio_path):
            for month in os.scandir(state.path):
                label= int(state.name == "Degraded_Reef") # 1 for Degraded_Reef, 0 for Non_Degraded_Reef

                # count=0
                count=0
                for wav in os.scandir(month.path):
                    if (wav.name.endswith(".TXT")):
                        continue
                    curr_data = extract_features(wav, label) 
                    all_data.append(curr_data)
                    #comment out the next 3 lines to get all the data
                    # count+=1
                    # if (count> 50):
                    #     break
                    count+=1
                    if (count> 200):
                        break

        ds = Dataset.from_list(all_data)
        class_list = ["Degraded_Reef" , "Non_Degraded_Reef"]
        
        split_ds = ds.train_test_split(test_size=0.3) # train is 70%, valid + test is 30%
        valid_test = split_ds["test"].train_test_split(test_size=0.7) #test is 70% of the 30% split
        
        mutlilabel_class_label = Sequence(ClassLabel(names=class_list))

        split_ds["train"]= split_ds["train"].cast_column("labels", mutlilabel_class_label)
        valid_test["train"] = valid_test["train"].cast_column("labels", mutlilabel_class_label)
        valid_test["test"]= valid_test["test"].cast_column("labels", mutlilabel_class_label)

        split_ds["train"]= split_ds["train"].cast_column("audio", Audio(48000))
        valid_test["train"] = valid_test["train"].cast_column("audio", Audio(48000))
        valid_test["test"]= valid_test["test"].cast_column("audio", Audio(48000))

        return AudioDataset(
            {"train": split_ds["train"], "valid": valid_test["train"], "test": valid_test["test"]},
            "null"
        )