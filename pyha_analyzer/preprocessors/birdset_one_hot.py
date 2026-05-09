import torch

def classes_one_hot(batch, num_classes):
  """
  Converts class labels to one-hot encoding.

  This method takes a batch of data and converts the class labels to one-hot encoding.
  The one-hot encoding is a binary matrix representation of the class labels.

  Args:
      batch (dict): A batch of data. The batch should be a dictionary where the keys are the field names and the values are the field data.

  Returns:
      dict: The batch with the "labels" field converted to one-hot encoding. The keys are the field names and the values are the field data.
  """
  label_list = [y for y in batch["ebird_code_multilabel"]]
  class_one_hot_matrix = torch.zeros(
      (len(label_list), num_classes), dtype=torch.float
  )

  for class_idx, idx in enumerate(label_list):
      class_one_hot_matrix[class_idx, idx] = 1

  class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
  return {"ebird_code_multilabel": class_one_hot_matrix}