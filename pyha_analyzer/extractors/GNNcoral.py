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
import sys
import re
from datetime import datetime


def extract_features(wav, label, site, dataset, raw_sounds):
    match = re.search(r"(\d{8})_(\d{6})", wav)
    if match:
        date=datetime.strptime(match.group(1), "%Y%m%d").date()
        #got info about rainy and dry season from pavones, costa rica
        month=date.month #between 1-12
        if (month >=5 and month <=11): #may to november
            season="Rainy"
        else:
            season="Dry"

        time = datetime.strptime(match.group(2), "%H%M%S").time()
        currentHour= time.hour #from 0 to 23
        if (currentHour >= 0 and currentHour <= 12): #between midnight and 12 PM
            timeOfDay="Morning"
        elif(currentHour>12 and currentHour <=17): #between 12 and 5 PM
            timeOfDay="Afternoon"
        else: # from 5 PM to midnight
            timeOfDay="Night"
    else:
        print("No date and time found in filename.")
        date, time, season, timeOfDay=None

    #make the list of sounds an array instead 
    sounds= raw_sounds.strip("{}").replace("'", "")
    sound_list = [s.strip() for s in sounds.split(",") if s.strip()] 

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
        #"site": site,
        #"dataset": dataset,
        "season": season,
        "timeOfDay": timeOfDay,
        "sounds": set(sound_list) #make sure no repeats
    }


class GNNCoralReef(DefaultExtractor):
    def __init__(self):
        super().__init__("GNNCoralReef")

    def __call__(self, csv_path, sampling=False):


        # Constants

        # Organize into buckets
        buckets = {
            ('Paola', 0): [],
            ('Paola', 1): [],
        }
        df = pd.read_csv(csv_path)

        # Now you can loop through the list of filepaths
        #for file_path in filepaths:
        for _, row in df.iterrows():
            file_path = row['filepath']
            raw_sounds = row['sounds']  # string like "{'bioph_cascading_saw', 'bioph_croak'}"

            # Detect dataset
            if "Paola" in file_path:
                dataset = "Paola"
            elif "Williams_et_al_2024" in file_path:
                dataset = "Williams_et_al_2024"
            # elif "Lin_et_al_2021" in file_path:
            #     dataset = "Lin_et_al_2021"
            else:
                continue  # skip others 

            # Detect label
            if "Non_Degraded_Reef" in file_path:
                label = 0
                site = "Non_Degraded_Reef"
            elif "Degraded_Reef" in file_path:
                label = 1
                site = "Degraded_Reef"
            else:
                continue # skip files without labels

            buckets[(dataset, label)].append((file_path, label, site, raw_sounds))
                
        # Define how many spectrograms each dataset contributes per file
        dataset_multipliers = {
            #'Williams_et_al_2024': 12,
            'Paola': 1,
            #'Lin_et_al_2021': 1
        }

        #does not do equal split when it comes to lin et al. 
        min_size=sys.maxsize
        for (dataset, label), items in buckets.items():
            size = len(items) * dataset_multipliers[dataset] #number of spectogrgams that the current dataset + label contributetes
            if (size >0 and size<min_size):
                min_size= size
                #datasetOfMinSize= dataset

        #print(f"min size is {min_size} from dataset {datasetOfMinSize}")
        
        random.seed(42)
        sampled = []
        for (dataset, label), items in buckets.items():
            #due to multiplicative factors, it may not be a perfect split 
            numToSample = min (int(min_size / dataset_multipliers[dataset]), len(items))
            sampled += random.sample(items, numToSample)
            print(f"sampling {numToSample} from ({dataset}, {label})")

        
        all_data = []
        for file_path, label, site, raw_sounds in sampled:
            try:
                curr_data = extract_features(file_path, label, site, dataset, raw_sounds)
                if curr_data is not None:
                    all_data.append(curr_data)
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")
                continue

        # # Summary
        print(f"Loaded: {len(all_data)} samples")
       


        ds = Dataset.from_list(all_data)
        #class_list = ["Degraded_Reef" , "Non_Degraded_Reef", "Unknown"]
        class_list = ["Degraded_Reef" , "Non_Degraded_Reef"]

        
        split_ds = ds.train_test_split(test_size=0.3) # train is 70%, valid + test is 30%
        valid_test = split_ds["test"].train_test_split(test_size=0.7) #test is 70% of the 30% split
        mutlilabel_class_label = Sequence(ClassLabel(names=class_list))

        split_ds["train"]= split_ds["train"].cast_column("labels", mutlilabel_class_label)
        valid_test["train"] = valid_test["train"].cast_column("labels", mutlilabel_class_label)
        valid_test["test"]= valid_test["test"].cast_column("labels", mutlilabel_class_label)

        # keep it at variable sampling rate, rather than hard coding at 48000
        split_ds["train"] = split_ds["train"].cast_column("audio", Audio())
        valid_test["train"] = valid_test["train"].cast_column("audio", Audio())
        valid_test["test"] = valid_test["test"].cast_column("audio", Audio())
                
        return AudioDataset(
                    {"train": split_ds["train"], "valid": valid_test["train"], "test": valid_test["test"]},
                    "null"
                )