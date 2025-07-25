from .defaultExtractors import DefaultExtractor
from datasets import ClassLabel, Sequence, Audio
from .. import AudioDataset
import os
from datasets import Dataset
import datasets
import pandas as pd
import wave
import random
import fnmatch


def parse_config(config_path):
    metadata = {}
    with open(config_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                if ((key.strip()== "Device ID") or (key.strip() == "Sample rate (Hz)")):
                    metadata[key.strip()] = val.strip()
    return metadata


def extract_features(wav, label, site):
    if label==0:
        oneHotEncodedLabel = [0,1] #Non_Degraded_Reef
    elif (label==1):
        oneHotEncodedLabel = [1,0] #Degraded_Reef
    else: #if label is 2
        #oneHotEncodedLabel = [0,0,1] #Unknown
        return
        
    with wave.open(wav, "rb") as wave_file:
        try:
            sample_rate = wave_file.getframerate()
        except Exception as e:
            print("Exception ", e)
            return
        
    return {
        "sample_rate": sample_rate,
        "labels": oneHotEncodedLabel,
        "filepath": str(wav),
        "audio": str(wav),
        "audio_in": {"array": str(wav), "sampling_rate": sample_rate},
        "site": site
    }


class MultiCoralReef(DefaultExtractor):
    def __init__(self):
        super().__init__("CoralReef")

    def __call__(self, audio_path, sampling=False):
        # Constants

        # Organize into buckets
        buckets = {
            ('Paola', 0): [],
            ('Paola', 1): [],
            ('Williams_et_al_2024', 0): [],
            ('Williams_et_al_2024', 1): [],
        }

        for root, dirs, files in os.walk(audio_path):
            for file in files:
                if not file.lower().endswith(".wav"):
                    continue

                file_path = os.path.join(root, file)

                # Detect dataset
                if "Paola" in file_path:
                    dataset = "Paola"
                elif "Williams_et_al_2024" in file_path:
                    dataset = "Williams_et_al_2024"
                else:
                    continue  # skip others like Lin

                # Detect label
                if "Non_Degraded_Reef" in file_path:
                    label = 0
                    site = "Non_Degraded_Reef"
                elif "Degraded_Reef" in file_path:
                    label = 1
                    site = "Degraded_Reef"
                else:
                    continue

                buckets[(dataset, label)].append((file_path, label, site))

        # 1: Since Williams dataset is much smaller than Paola, determine its size to make sure we know Paola's
        w0 = len(buckets[('Williams_et_al_2024', 0)])
        w1 = len(buckets[('Williams_et_al_2024', 1)])

        # balance degraded and non-degraded to make it equal. this means # paola might be greater than # williams for either degraded or non-degraded but hopefully not by too much
        max_len = max(w0, w1)
        # Step 2: Each dataset in William's dataset is about 60 seconds long and we are breaking every 5 second chunk into a spectogram so one williams file, contributes 12 spectograms. ensure paola, where one file contributed only one spectogram, is 12x as much as williams
        if (max_len-w0)==0:
            p0_target = int((60/5) * max_len)
            p1_target = int((60/5) * (max_len-w1))
        else: #max_len-w1 = 0
            p0_target = int((60/5) * (max_len-w0))
            p1_target = int((60/5) * max_len)


        # # Step 3: Check if enough Paola data
        # if len(buckets[('Paola', 0)]) < p0_target:
        #     raise ValueError(f"Not enough Paola non-degraded files: found {len(buckets[('Paola', 0)])}, need {p0_target}")
        # if len(buckets[('Paola', 1)]) < p1_target:
        #     raise ValueError(f"Not enough Paola degraded files: found {len(buckets[('Paola', 1)])}, need {p1_target}")


        # Step 4: Sample Paola files
        random.seed(42)
        p0_samples = random.sample(buckets[('Paola', 0)], p0_target)
        p1_samples = random.sample(buckets[('Paola', 1)], p1_target)

        # Step 5: Gather all samples
        w0_samples = buckets[('Williams_et_al_2024', 0)]
        w1_samples = buckets[('Williams_et_al_2024', 1)]
        sampled= p0_samples + w0_samples + p1_samples

        # Step 7: Feature extraction
        all_data = []
        for file_path, label, site in sampled:
            try:
                curr_data = extract_features(file_path, label, site)
                if curr_data is not None:
                    all_data.append(curr_data)
            except (wave.Error, EOFError) as e:
                print(f"Skipping {file_path} due to error: {e}")
                continue

        # Summary
        print(f"Loaded: {len(all_data)} samples")
        print(f"  Paola Non-Degraded: {len(p0_samples)}")
        print(f"  Paola Degraded:     {len(p1_samples)}")
        print(f"  Williams Non-Degraded: {len(w0_samples)}")
        print(f"  Williams Degraded:     {len(w1_samples)}")
        print(f"  Total: {len(all_data)}")


        ds = Dataset.from_list(all_data)
        #class_list = ["Degraded_Reef" , "Non_Degraded_Reef", "Unknown"]
        class_list = ["Degraded_Reef" , "Non_Degraded_Reef"]
        
        ds = ds.class_encode_column('site')
        
        
        if sampling:
            
            filt_datasets = []
            
            label_column = 'site'
            
            labels = set(ds[label_column])
            
            for label in labels:
                label_dataset = ds.filter(lambda x: x[label_column] == label)
                
                filt_datasets.append(label_dataset.shuffle(seed=42).select([i for i in range(25)]))
                
            balanced_dataset = datasets.concatenate_datasets(filt_datasets)
            
            balanced_dataset = balanced_dataset.shuffle(seed=42)
                
            ds = balanced_dataset
        
        split_ds = ds.train_test_split(test_size=0.3, stratify_by_column='site') # train is 70%, valid + test is 30%
        valid_test = split_ds["test"].train_test_split(test_size=0.7, stratify_by_column='site') #test is 70% of the 30% split
        
        mutlilabel_class_label = Sequence(ClassLabel(names=class_list))

        split_ds["train"]= split_ds["train"].cast_column("labels", mutlilabel_class_label)
        valid_test["train"] = valid_test["train"].cast_column("labels", mutlilabel_class_label)
        valid_test["test"]= valid_test["test"].cast_column("labels", mutlilabel_class_label)

        # split_ds["train"]= split_ds["train"].cast_column("audio", Audio(48000))
        # valid_test["train"] = valid_test["train"].cast_column("audio", Audio(48000))
        # valid_test["test"]= valid_test["test"].cast_column("audio", Audio(48000))
        # keep it at variable sampling rate, rather than hard coding at 48000
        split_ds["train"] = split_ds["train"].cast_column("audio", Audio())
        valid_test["train"] = valid_test["train"].cast_column("audio", Audio())
        valid_test["test"] = valid_test["test"].cast_column("audio", Audio())
                
        return AudioDataset(
                    {"train": split_ds["train"], "valid": valid_test["train"], "test": valid_test["test"]},
                    "null"
                )