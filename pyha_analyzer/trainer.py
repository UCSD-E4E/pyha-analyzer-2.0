from transformers import Trainer, TrainingArguments
from .dataset import AudioDataset
from .constants import DEFAULT_COLUMNS
from .models.base_model import BaseModel
from .logging.logging import Logger
from .metrics.evaluate import ComputeMetricsBase
from .metrics.classification_metrics import AudioClassificationMetrics

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
        logger: Logger=None,
        metrics: ComputeMetricsBase = None,
        training_args=None,
        data_collator=None,
        preprocessor=None,
    ):
        assert issubclass(type(model), BaseModel), (
            "PyhaTrainer Only Work with BaseModel. Please have model inherit from BaseModel"
        )

        self.dataset = dataset

        ## DEFINES METRICS FOR DETERMINING HOW GOOD MODEL IS
        if metrics is not None:
            self.compute_metrics = metrics
        else:
            self.compute_metrics = AudioClassificationMetrics(None, class_size=10) #TODO CHANGE

        ## HANDLES DEFAULT ARGUMENTS FOR HUGGING FACE TRAINER
        if training_args is None:
            training_args = TrainingArguments("working_dir")
        training_args.label_names = DEFAULT_COLUMNS
        self.training_args = training_args

        if logger is not None:
            self = logger.modify_training_args(self)

        super().__init__(
            model,
            training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            data_collator=data_collator,
            processing_class=preprocessor,
        )

        if logger is not None:
            self = logger.modify_trainer(self)


    def evaluate(self, dataset=None):
        if dataset is None:
            dataset = self.dataset["test"]

        super().evaluate(dataset)
