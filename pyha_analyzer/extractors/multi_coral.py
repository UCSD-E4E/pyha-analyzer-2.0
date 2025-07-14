from .defaultExtractors import DefaultExtractor
from datasets import ClassLabel, Sequence, Audio
from .. import AudioDataset
import os
from datasets import Dataset
import datasets
import pandas as pd
import wave


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
        oneHotEncodedLabel = [0,1,0] #Non_Degraded_Reef
    elif (label==1):
        oneHotEncodedLabel = [1,0,0] #Degraded_Reef
    else: #if label is 2
        oneHotEncodedLabel = [0,0,1] #Unknown
        
    with wave.open(wav.path, "rb") as wave_file:
        try:
            sample_rate = wave_file.getframerate()
        except Exception as e:
            print("Exception ", e)
            return
        
    return {
        "sample_rate": sample_rate,
        "labels": oneHotEncodedLabel,
        "filepath": str(wav.path),
        "audio": str(wav.path),
        "audio_in": {"array": str(wav.path), "sampling_rate": sample_rate},
        "site": site
    }


class MultiCoralReef(DefaultExtractor):
    def __init__(self):
        super().__init__("CoralReef")

    def __call__(self, audio_path, sampling=False):
        all_data = []
        #audio_path= "/home/s.kamboj.400/unzipped-coral"
        count = 0
        for dataset in os.scandir(audio_path):
            if not dataset.is_dir():
                continue
            for state in os.scandir(dataset.path):
                if not state.is_dir():
                    continue
                print(state.name)
                for month in os.scandir(state.path):
                    if not month.is_dir():
                        continue
                    if dataset.name == 'Paola':
                        label = int(state.name == "Degraded_Reef") # 1 for Degraded_Reef, 0 for Non_Degraded_Reef
                    elif dataset.name == 'Lin_et_al_2021': # all of Lin's data set is Non-degraded
                        label = 0
                    #use the script to separate Indonesai into healthy and non-healthy
                    elif dataset.name=='Williams_et_al_2024': 
                        #label = int(state.name == "Degraded_Reef") # Cleared all unknowns to make it back to binary
                        if (state.name=="Degraded_Reef"):
                            label=1
                        elif (state.name=="Non_Degraded_Reef"):
                            label = 0
                        else:
                            label = 2 #2 is for unknown
                    else:
                        label=2
                    
                    site = state.name
                    

                    
                    # count=0
                    for wav in os.scandir(month.path):
                        #if (wav.name.endswith(".TXT")):
                        if not wav.name.lower().endswith(".wav"):
                            continue
                        try:
                            curr_data = extract_features(wav, label, site) 
                        except (wave.Error, EOFError) as e:
                            print(f"Skipping file {wav.path} due to WAV error: {e}")
                            continue
                        if curr_data is not None:
                            all_data.append(curr_data)
                            #comment out the next 3 lines to get all the data
                            # count+=1
                            # if (count> 50):
                            #     break
                            if site not in ['Degraded_Reef', 'Non_Degraded_Reef']:
                                count += 1
                            
        print('count:', count)
        
        ds = Dataset.from_list(all_data)
        class_list = ["Degraded_Reef" , "Non_Degraded_Reef", "Unknown"]
        #class_list = ["Degraded_Reef" , "Non_Degraded_Reef"]
        
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