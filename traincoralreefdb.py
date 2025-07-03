import lancedb
from lancedb.pydantic import Vector
import torch
import tqdm
import pandas as pd
import pyarrow as pa
from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
from pyha_analyzer.models.demo_CNN import ResnetConfig, ResnetModel
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors
from pyha_analyzer.models import EfficentNet
import torch


coralreef_extractor = extractors.CoralReef()
coral_ads = coralreef_extractor("/home/s.kamboj.400/unzipped-coral")
# print(coral_ads)

preprocessor = MelSpectrogramPreprocessors(duration=5, class_list=coral_ads["train"].features["labels"].feature.names)

coral_ads["train"].set_transform(preprocessor)
coral_ads["valid"].set_transform(preprocessor)
coral_ads["test"].set_transform(preprocessor)


model = EfficentNet(num_classes=2)

args = PyhaTrainingArguments(
    working_dir="working_dir"
)
args.num_train_epochs = 1
args.eval_steps = 20

trainer = PyhaTrainer(
    model=model,
    dataset=coral_ads,
    training_args=args
)
trainer.train()

print(trainer.evaluate(eval_dataset=coral_ads["test"], metric_key_prefix="Soundscape"))


#Connect to LanceDB and save the embeddings
uri = "database/coral_reef_db.lance"
db = lancedb.connect(uri)

#delete the table for testing purposes, so that the same embeddings are not re-inserted because id is simply a key not a primary key, so embeddings can get reinserted
if "coral_embeddings" in db.table_names():
    db.drop_table("coral_embeddings")


schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("vector_embedding", pa.list_(pa.float32(), list_size=1280)),
    pa.field("label", pa.string()),
    pa.field("audio_path", pa.string())
])
table = db.create_table("coral_embeddings", schema=schema)

#add info into vector database
for split in ["train", "valid"]:
    dataset = coral_ads[split]
    for idx in range(len(dataset)):
        item = dataset[idx]
        embedding=model.get_embedding(**item) 
        
        if ((item["labels"] == [0.0, 1.0]).all()):
            currLabel= "Non_Degraded_Reef"
        elif ((item["labels"] == [1.0, 0.0]).all()):
            currLabel= "Degraded_Reef"
        else:
            currLabel = None
            print("Unknown label", item["labels"])

        metadata = {
            "id": f"{split}_{idx}",
            "vector_embedding": embedding.tolist(),
            "label": currLabel, 
            "audio_path": item["filepath"]
        }

        table.add([metadata])


#query the database to verify the embeddings
for currAudioIdx in ["test"]:
    dataset = coral_ads[currAudioIdx]

    for idx in range(len(dataset)):
        item = dataset[idx]
        embedding=model.get_embedding(**item) 
        query_vector = embedding.tolist()
        results = table.search(query_vector).limit(1).to_pandas()
        print("The most similar to ", item["filepath"], " is ", results["audio_path"].tolist()[0]) # see just the audio path to the most simlar embeddings


del model
del trainer
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
