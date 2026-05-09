from datasets import load_dataset, Audio, DatasetDict

from pyha_analyzer.preprocessors.birdset_event_mapper import XCEventMapping

from pyha_analyzer.preprocessors.smart_sampling import smart_sampling

birdset_data = load_dataset("DBD-research-group/BirdSet", "HSN", trust_remote_code=True)

sampling_rate = 32_000

birdset_data = birdset_data.cast_column(
  column="audio",
  feature=Audio(
      sampling_rate=sampling_rate,
      mono=True,
      decode=True,
  ),
)

birdset_data = DatasetDict(
    {split: birdset_data[split] for split in ["train", "test_5s"]}
)

event_mapper = XCEventMapping()

print(">> Mapping train data.")
birdset_data["train"] = birdset_data["train"].map(
    event_mapper,
    remove_columns=["audio"],
    batched=True,
    batch_size=300,
    desc="Train event mapping",
    load_from_cache_file=False
)

# defaults for HSN BirdSet

class_limit = 500
event_limit = 5 

birdset_data["train"] = birdset_data["train"].remove_columns(["audio"])

print(birdset_data)

birdset_data = birdset_data.rename_column("ebird_code_multilabel", "labels")


birdset_data["train"] = smart_sampling(
  dataset=birdset_data["train"],
  label_name="ebird_code",
  class_limit=class_limit,
  event_limit=event_limit,
)

print(birdset_data)