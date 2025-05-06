from transformers import Trainer, TrainingArguments
from .dataset import AudioDataset
from .constants import DEFAULT_COLUMNS
from .models.base_model import BaseModel

"""
Intended to wrap around the normal hugging face trainer

In case there are any project spefific configs we need to add here

For example: We may want to log git hashes, so that can be included
"""


class PyhaTrainer(Trainer):
    def __init__(
        self,
        model: BaseModel,
        dataset: AudioDataset,
        training_args=None,
        data_collator=None,
        preprocessor=None,
    ):
        assert issubclass(type(model), BaseModel), (
            "PyhaTrainer Only Work with BaseModel. Please have model inherit from BaseModel"
        )

        self.dataset = dataset

        if training_args is None:
            training_args = TrainingArguments("working_dir")
        training_args.label_names = DEFAULT_COLUMNS

        super().__init__(
            model,
            training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            data_collator=data_collator,
            processing_class=preprocessor,
        )

    def evaluate(self, dataset=None):
        if dataset is None:
            dataset = self.dataset["test"]

        super().evaluate(dataset)
