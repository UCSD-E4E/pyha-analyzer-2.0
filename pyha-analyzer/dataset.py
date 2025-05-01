import datasets
from datasets import load_dataset

class AudioDataset():
    def __init__(self, data_path="", dataset_name="", extractor=None):
        self.data = None # HF dataset object
        self.meta = None
        self.train = False
        # pseudocode:
        # if input is path
        #   if input is directory
        #      self.data = load_dataset("audiofolder", data_dir=data_path)
        #      self.meta = extractor(csv file input)
        #      update self.data with metadata
        #   elif input is a csv file
        #      use csv to load dataset
        #      possibly generate metadata csv and use filepaths to find root directory for audiofolder
        # elif input is the name of an existing dataset
        #   self.data = load_dataset(dataset_name)
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    