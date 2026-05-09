
import pandas as pd

from tqdm import tqdm

from collections import Counter

import random

def smart_sampling(dataset, label_name, class_limit, event_limit):
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
  print('reached')
  df = pd.DataFrame(dataset)
  print(df)
  path_label_count = df.groupby(["id", label_name], as_index=False).size()
  path_label_count = path_label_count.set_index("id")
  class_sizes = df.groupby(label_name).size()

  print('here')
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