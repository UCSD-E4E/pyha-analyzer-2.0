import re
from .defaultExtractors import DefaultExtractor
from datasets import load_dataset, DatasetDict, ClassLabel, Sequence, Audio
from .. import AudioDataset
import os
import librosa

from pathlib import Path
from datetime import datetime
import numpy as np
from datasets import Dataset

#LABEL_MAP = ["Degraded_Reef" , "Non_Degraded_Reef"]

def parse_config(config_path):
    metadata = {}
    with open(config_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                if ((key.strip()== "Device ID") or (key.strip() == "Sample rate (Hz)")): #or (key.strip() == "Gain")):
                    metadata[key.strip()] = val.strip()
    return metadata

def one_hot_encode(labels, classes):
    one_hot = np.zeros(len(classes))
    for label in labels:
        one_hot[label] = 1
    return np.array(one_hot, dtype=float)

def one_hot_encode_ds_wrapper(row, class_list):
    row["labels"] = one_hot_encode(row["labels"], class_list)
    return row

def extract_features(wav, config, label):
    #raw_audio_data is a 2640000 size array, does not resample the audio
    #print(f"Extracting features from {wav.path}")
    raw_audio_data, sampling_rate = librosa.load(wav.path, sr=None) 
    #if sampling rate is not equal to the config["Sample rate (Hz)"], resample the audio (should not be the case, due to sr=None on previous line)
    if sampling_rate != int(config["Sample rate (Hz)"]):
        print(f"Sampling rate mismatch for {wav.path}. Expected {config['Sample rate (Hz)']}, got {sampling_rate}. Resampling...")
        raw_audio_data = librosa.resample(raw_audio_data, orig_sr=sampling_rate, target_sr=int(config["Sample rate (Hz)"]))
    
    match = re.search(r"(\d{8})_(\d{6})", wav.name)
    if match:
        date = datetime.strptime(match.group(1), "%Y%m%d").date()
        time = datetime.strptime(match.group(2), "%H%M%S").time()
    else:
        print("No date and time found in filename.")
        date = None
        time = None
    
    S = np.abs(librosa.stft(raw_audio_data))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sampling_rate)

    max_freq = float(np.max(freqs))
    min_freq = float(np.min(freqs))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=raw_audio_data, sr=sampling_rate)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=raw_audio_data, sr=sampling_rate)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=raw_audio_data)))
    rms = float(np.mean(librosa.feature.rms(y=raw_audio_data)))
    
    # make everything positive so that you can find all the upper and lower peaks
    envelope = np.abs(raw_audio_data)
    # TODO : Experiment with the peak picking parameters
    peaks= librosa.util.peak_pick(envelope, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    num_peaks = int(len(peaks))
    if label==0:
        oneHotEncodedLabel = [0,1]
    else:
        oneHotEncodedLabel = [1,0]

    return {
        "sample_rate": sampling_rate,
        "device_id": config.get("Device ID", "Unknown"),
        "duration": librosa.get_duration(y=raw_audio_data, sr=sampling_rate),
        "date": date,
        "time": time,
        "labels": oneHotEncodedLabel,
        "filepath": str(wav.path),
        "audio": str(wav.path),
        #"audio_in": {"array": raw_audio_data, "sampling_rate": sampling_rate},
        "audio_in": {"array": str(wav.path), "sampling_rate": sampling_rate},
        "max_freq": max_freq,
        "min_freq": min_freq,
        "mean_freq": centroid,
        "peak_freq": float(freqs[np.argmax(S_db.mean(axis=1))]),
        "variance_freq": bandwidth, #variance of the frequency distribution
        "num_peaks": num_peaks,
        "spectral_flatness": flatness,
        "rms_energy": rms
    }


class CoralReef(DefaultExtractor):
    def __init__(self):
        super().__init__("CoralReef")

    def __call__(self, audio_path):
        all_data = []
        #audio_path= "/home/s.kamboj.400/unzipped-coral"
        for state in os.scandir(audio_path):
            for month in os.scandir(state.path):
                #label = LABEL_MAP[state.name]  # Extract from parent dir name
                label= int(state.name == "Degraded_Reef") # 1 for Degraded_Reef, 0 for Non_Degraded_Reef
                config_path = next(Path(month.path).rglob("CONFIG.TXT"), None)
                if not config_path:
                    # TODO : Ask Sean if we should skip folders without a config file. For now, we are skipping them.
                    print(f"Config file not found in {state.name}, skipping.")
                    continue
                config = parse_config(config_path)

                #count=0
                for wav in os.scandir(month.path):
                    if (wav.name.endswith(".TXT")):
                        continue
                    #print(f"Processing {wav.name} with label {label}")
                    curr_data = extract_features(wav, config, label)
                    all_data.append(curr_data)
                    # count+=1
                    # if (count> 6):
                    #     break
        
        ds = Dataset.from_list(all_data)
        # ds = ds.class_encode_column("labels")
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

        # split_ds["train"] = split_ds["train"].map(
        #         lambda row: one_hot_encode_ds_wrapper(row, class_list)
        #         ).cast_column("labels", class_list=ClassLabel(names=class_list))
        
        # return DatasetDict({
        #     "train": split_ds["train"], #biggest piece
        #     "valid": valid_test["train"], #smallest piece
        #     "test": valid_test["test"] 
        # })
        return AudioDataset(
            {"train": split_ds["train"], "valid": valid_test["train"], "test": valid_test["test"]},
            "null"
        )

        







