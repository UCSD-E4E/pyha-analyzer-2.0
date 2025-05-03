import datasets
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import numpy as np

class AudioDataset(): #Probably should inherit the hugging face datasets
    def __init__(self,
                 data_path: str,
                 train=False,
                 species=[],
                 dataset_name = ""):
        
        # QUESTION: Will we load from a csv every time?
        # Techically with a hugging face dataset we could load from other file types
        self.data_dir = Path(data_path)
        self.meta_dir = self.data_dir / "metadata.csv" # metadata csv file must be named metadata
        self.train = train
        
        # Ensure paths exist
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found.")
        if not self.meta_dir.exists():
            # TODO: add reminder about naming convention if csvs found in data directory
            raise FileNotFoundError(f"Metadata csv not found at {self.meta_dir}.")
        
        # Load in metadata and data
        self.meta = pd.read_csv(self.meta_dir)
        # self.verify_metadata()

        ## For a frist draft, this might be alright. 
        ## But we may want the ability to convert from a diffrent huggingface dataset to AudioDataset
        ## Say Birdset Hugging face data objects
        self.data = load_dataset("audiofolder", data_dir=self.data_dir)
    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def train_test_split(self):
        # pseudocode:
        # self.dataset = self.dataset.train_test_split()
        # returns none
        pass
    
    def verify_metadata(self):
        """Verify metadata csv for hfds
        Necessary? Huggingface has the error handling within itself
        """
        assert "file_name" in self.meta.columns, "'File_name' not found in metadata columns."
    
def kaleidoscope_extractor(data_dir, csv_path):
    # pseudocode
    # meta = pd.read_csv(csv_path)
    # [format csv into hf metadata csv]
    # - change column names
    # - add event onset/offset from offset and duration
    # - output: one row per file
    # meta.to_csv(data_dir/metadata.csv)
    # dataset = AudioDataset(data_dir,)
    pass

def species_wise_valid_split(self, dataset):
    """
    Due to class imbalances, species can often be dropped from naive random test-train splits
    Handles the creation of spliting the test and train datasets
    TODO: Should this live in the trainer?
    This should live outside for consistent trainings. We can save sepearte datasets.
    """
    index = []
    for label in dataset["labels"].names:
        class_count = len(dataset.filter(lambda x: x["labels"] == label))
        index.extend(list(
            np.random.choice([0, 1], class_count, p=[1 - self.split_p, self.split_p])))
    return (
        dataset[index == 0], # Train
        dataset[index == 1] # Validation
    )

def main():
    print("welcome to dataset.py")
    
if __name__ == "__main__":
    main()