import pandas as pd
from tqdm import tqdm
from collections import Counter
import random
from datasets import DatasetDict
import torch

def smart_sampling(
        dataset: DatasetDict, 
        label_name: str, 
        class_limit: int, 
        event_limit: int
    ) -> DatasetDict:
    """
        Function to remove redundant samples created in XCEventMapping()

        Args:
            dataset: `DatasetDict`
                Dataset from which redundant samples should be removed
            label_name: `str`
                Field to use to find class
            class_limit: `int`
                Max number of samples of a specific class
            event_limit: `int`
                Max number of events in one sample
    """
    def _unique_identifier(x, labelname):
        file = x["filepath"]
        label = x[labelname]
        return {"id": f"{file}-{label}"}

    class_limit = class_limit if class_limit else -float("inf")
    dataset = dataset.map(
        lambda x: _unique_identifier(x, label_name),
        desc="sampling: unique-identifier",
        load_from_cache_file=False
    )
    # Only load necessary columns into pandas to avoid loading large audio/spectrogram data
    df = pd.DataFrame(dataset.select_columns(["id", label_name]))
    print(df)
    path_label_count = df.groupby(["id", label_name], as_index=False).size()
    path_label_count = path_label_count.set_index("id")
    class_sizes = df.groupby(label_name).size()

    for label in tqdm(class_sizes.index, desc="sampling"):
        current = path_label_count[path_label_count[label_name] == label]
        total = current["size"].sum()
        most = current["size"].max()

        while total > class_limit or most != event_limit:
            largest_count = current["size"].value_counts()[current["size"].max()]
            n_largest = current.nlargest(largest_count + 1, "size")
            to_del = n_largest["size"].max() - n_largest["size"].min()

            idxs = n_largest[n_largest["size"] == n_largest["size"].max()].index
            if (
                total - (to_del * largest_count) < class_limit
                or most == event_limit
                or most == 1
            ):
                break
            for idx in idxs:
                current.at[idx, "size"] = current.at[idx, "size"] - to_del
                path_label_count.at[idx, "size"] = (
                    path_label_count.at[idx, "size"] - to_del
                )

            total = current["size"].sum()
            most = current["size"].max()
    
    event_counts = Counter(dataset["id"])

    all_file_indices = {label: [] for label in event_counts.keys()}
    for idx, label in enumerate(dataset["id"]):
        all_file_indices[label].append(idx)

    limited_indices = []
    for file, indices in all_file_indices.items():
        limit = path_label_count.loc[file]["size"]
        limited_indices.extend(random.sample(indices, limit))

    dataset = dataset.remove_columns("id")
    return dataset.select(limited_indices)

def classes_one_hot(
    batch: dict, 
    num_classes: int
) -> dict:
    """
    Converts class labels to one-hot encoding.

    This method takes a batch of data and converts the class labels to one-hot encoding.
    The one-hot encoding is a binary matrix representation of the class labels.

    Args:
        `batch`: dict
            A batch of data. The batch should be a dictionary where the keys are the field names and the values are the field data.
        `num_classes`: int
            Total number of classes in dataset to one-hot encode

    Returns:
        dict: The batch with the "labels" field converted to one-hot encoding. The keys are the field names and the values are the field data.
    """
    label_list = [y for y in batch["labels"]]
    class_one_hot_matrix = torch.zeros(
        (len(label_list), num_classes), dtype=torch.float
    )

    for class_idx, idx in enumerate(label_list):
        class_one_hot_matrix[class_idx, idx] = 1

    class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
    return {"labels": class_one_hot_matrix}