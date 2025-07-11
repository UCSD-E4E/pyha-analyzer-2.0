import os
import soundfile as sf
from .defaultExtractors import DefaultExtractor
from datasets import ClassLabel, Sequence, Audio
from .. import AudioDataset
from datasets import Dataset

#bad_files = []
def get_wav_sampling_rate(file_path):
    try:
        info = sf.info(file_path)
        return info.samplerate
    #about 350 WAV files are corrupt
    except RuntimeError as e:
        #bad_files.append(file_path)
        return None

def extract_features(wav, label):
    #labels are {'Data', 'LA_Sand_Forrest_ULTRASONIC_03-05-25', 'A Zoom F3_03-05-25', 'Location A Sand Forrest'}
    if label==0:
        oneHotEncodedLabel = [1,0]
    # elif label =='LA_Sand_Forrest_ULTRASONIC_03-05-25':
    #     oneHotEncodedLabel = [0,1,0,0]
    # elif label =='A Zoom F3_03-05-25':
    #     oneHotEncodedLabel = [0,0,1,0]
    # elif label =='Location A Sand Forrest':
    #     oneHotEncodedLabel = [0,0,0,1]
    # else: #invalid label so return nothing
    #     return

    sample_rate = get_wav_sampling_rate(wav)
    #This means file is likely corrupted
    if (sample_rate==None):
        return

    return {
        "sample_rate": sample_rate,
        "labels": oneHotEncodedLabel,
        "filepath": str(wav),
        "audio": str(wav),
        "audio_in": {"array": str(wav), "sampling_rate": sample_rate},
    }


class Music(DefaultExtractor):
    def __init__(self):
        super().__init__("Music")

    def __call__(self, audio_path):
        all_data = []

        for root, _, files in os.walk(audio_path):
            for filename in files:
                if filename.endswith(".wav") or filename.endswith(".WAV"):
                    curr_wav=os.path.join(root, filename)
                    # splitArray = str(curr_wav).split("/")
                    # label = splitArray[6]
                    label=0
                    curr_data = extract_features(curr_wav, label) 
                    #ignore all corrupt files, as determined by an inability to open them and get their sampling rate
                    if curr_data is not None:
                        all_data.append(curr_data)
        # #print all bad files into a txt file
        # with open("bad_wav_files.txt", "w") as f:
        #     for path in bad_files:
        #         f.write(path + "\n")

        ds = Dataset.from_list(all_data)
        class_list = ["Random", "Random 2"]

        split_ds = ds.train_test_split(test_size=0.3) # train is 70%, valid + test is 30%
        valid_test = split_ds["test"].train_test_split(test_size=0.7) #test is 70% of the 30% split

        mutlilabel_class_label = Sequence(ClassLabel(names=class_list))

        split_ds["train"]= split_ds["train"].cast_column("labels", mutlilabel_class_label)
        valid_test["train"] = valid_test["train"].cast_column("labels", mutlilabel_class_label)
        valid_test["test"]= valid_test["test"].cast_column("labels", mutlilabel_class_label)

        #Audio() grabs sample rate on its own and is not hard coded
        split_ds["train"]= split_ds["train"].cast_column("audio", Audio())
        valid_test["train"] = valid_test["train"].cast_column("audio", Audio())
        valid_test["test"]= valid_test["test"].cast_column("audio", Audio())

        return AudioDataset(
            {"train": split_ds["train"], "valid": valid_test["train"], "test": valid_test["test"]},
            "null"
        )