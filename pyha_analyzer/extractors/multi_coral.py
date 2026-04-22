from .defaultExtractors import DefaultExtractor
from datasets import ClassLabel, Sequence, Audio
from .. import AudioDataset
import os
from datasets import Dataset
import wave
import random
import sys


def parse_config(config_path):
    metadata = {}
    with open(config_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                if ((key.strip()== "Device ID") or (key.strip() == "Sample rate (Hz)")):
                    metadata[key.strip()] = val.strip()
    return metadata


def extract_features(wav, label, site, dataset):
    if label==0:
        oneHotEncodedLabel = [0,1] #Non_Degraded_Reef
    elif (label==1):
        oneHotEncodedLabel = [1,0] #Degraded_Reef
    else: #if label is 2
        #oneHotEncodedLabel = [0,0,1] #Unknown
        return

    # if dataset == "Paola":
    #     oneHotEncodedLabel = [1, 0, 0]
    # elif dataset == "Williams_et_al_2024":
    #     oneHotEncodedLabel = [0, 1, 0]
    # elif dataset == "Lin_et_al_2021":
    #     oneHotEncodedLabel = [0, 0, 1]
    # else:
    #     return
    
        
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
        "site": site,
        "dataset": dataset
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
            # ('Lin_et_al_2021', 0): [],
            # ('Lin_et_al_2021', 1): []
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

                buckets[(dataset, label)].append((file_path, label, site, dataset))
                
        # Define how many spectrograms each dataset contributes per file
        dataset_multipliers = {
            'Williams_et_al_2024': 12,
            'Paola': 1,
            #'Lin_et_al_2021': 1
        }

        #does not do equal split when it comes to lin et al. 
        min_size=sys.maxsize
        for (dataset, label), items in buckets.items():
            size = len(items) * dataset_multipliers[dataset] #number of spectogrgams that the current dataset + label contributetes
            if (size >0 and size<min_size):
                min_size= size
                datasetOfMinSize= dataset

        print(f"min size is {min_size} from dataset {datasetOfMinSize}")
        
        random.seed(42)
        sampled = []
        for (dataset, label), items in buckets.items():
            #due to multiplicative factors, it may not be a perfect split 
            numToSample = min (int(min_size / dataset_multipliers[dataset]), len(items))
            sampled += random.sample(items, numToSample)
            print(f"sampling {numToSample} from ({dataset}, {label})")

        
        #make sure that same number of degraded and non-degraded considering multiplicative factors
        #reason why u do this: lin, et, al 1 causes discrepancy between 



        # # Step 1: Get Williams data sizes
        # w0 = len(buckets[('Williams_et_al_2024', 0)])
        # w1 = len(buckets[('Williams_et_al_2024', 1)])

        # # Step 2: Pick min of w0 and w1 for balancing
        # min_w = min(w0, w1)

        # # Step 3: Sample Williams equally
        # random.seed(42)
        # w0_samples = random.sample(buckets[('Williams_et_al_2024', 0)], min_w)
        # w1_samples = random.sample(buckets[('Williams_et_al_2024', 1)], min_w)

        # # Step 4: Sample same number from Paola for each label
        # p0_available = len(buckets[('Paola', 0)])
        # p1_available = len(buckets[('Paola', 1)])
        # #because min_w produces 12 spectograms per clip since it is 60 seconds long and each spectogram is for 5 seconds
        # p_sample_size = min(int(min_w*(60/5)), p0_available, p1_available)

        # p0_samples = random.sample(buckets[('Paola', 0)], p_sample_size)
        # p1_samples = random.sample(buckets[('Paola', 1)], p_sample_size)

        # # Step 5: Combine samples
        # sampled = w0_samples + w1_samples + p0_samples + p1_samples

        
        # Step 7: Feature extraction
        all_data = []
        for file_path, label, site, dataset in sampled:
            try:
                curr_data = extract_features(file_path, label, site, dataset)
                if curr_data is not None:
                    all_data.append(curr_data)
            except (wave.Error, EOFError) as e:
                print(f"Skipping {file_path} due to error: {e}")
                continue

        # # Summary
        print(f"Loaded: {len(all_data)} samples")
        # print(f"  Paola Non-Degraded: {len(p0_samples)}")
        # print(f"  Paola Degraded:     {len(p1_samples)}")
        # print(f"  Williams Non-Degraded: {len(w0_samples)}")
        # print(f"  Williams Degraded:     {len(w1_samples)}")
        # print("")
        # print(f"  Paola Total: {len(buckets[('Paola', 0)]) + len(buckets[('Paola', 1)])}")
        # print(f"  Williams Total: {len(buckets[('Williams_et_al_2024', 0)]) + len(buckets[('Williams_et_al_2024', 1)])}")
        # print(f"  Non-degraded total: {len(buckets[('Paola', 0)]) + len(buckets[('Williams_et_al_2024', 0)])}")
        # print(f"  Degraded total: {len(buckets[('Paola', 1)]) + len(buckets[('Williams_et_al_2024', 1)])}")
        
        # print(f"Loaded: {len(all_data)} samples")
        # print(f"    Paola Total: {len(buckets[('Paola', 0)]) + len(buckets[('Paola', 1)])}")
        # print(f"    Williams Total: {len(buckets[('Williams_et_al_2024', 0)]) + len(buckets[('Williams_et_al_2024', 1)])}")
        # print(f"    Lin Total: {len(buckets[('Lin_et_al_2021', 0)]) + len(buckets[('Lin_et_al_2021', 1)])}")


        ds = Dataset.from_list(all_data)
        #class_list = ["Degraded_Reef" , "Non_Degraded_Reef", "Unknown"]
        class_list = ["Degraded_Reef" , "Non_Degraded_Reef"]
        #class_list = ["Paola" , "Williams_et_al_2024", "Lin_et_al_2021"]
        
        # ds = ds.class_encode_column('site')
        
        # if sampling:
            
        #     filt_datasets = []
            
        #     label_column = 'site'
            
        #     labels = set(ds[label_column])
            
        #     for label in labels:
        #         label_dataset = ds.filter(lambda x: x[label_column] == label)
                
        #         filt_datasets.append(label_dataset.shuffle(seed=42).select([i for i in range(25)]))
                
        #     balanced_dataset = datasets.concatenate_datasets(filt_datasets)
            
        #     balanced_dataset = balanced_dataset.shuffle(seed=42)
                
        #     ds = balanced_dataset
        
        # split_ds = ds.train_test_split(test_size=0.3, stratify_by_column='site') # train is 70%, valid + test is 30%
        # valid_test = split_ds["test"].train_test_split(test_size=0.7, stratify_by_column='site') #test is 70% of the 30% split
        
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