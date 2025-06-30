from transformers import Trainer, TrainingArguments, IntervalStrategy
from .logging.wandb import WANDBLogging
from .dataset import AudioDataset
from .constants import DEFAULT_COLUMNS, DEFAULT_PROJECT_NAME, DEFAULT_RUN_NAME
from .models.base_model import BaseModel
from .metrics.evaluate import ComputeMetricsBase
from .metrics.classification_metrics import AudioClassificationMetrics

"""
Intended to wrap around the normal hugging face trainer

In case there are any project spefific configs we need to add here

For example: We may want to log git hashes, so that can be included
"""

class PyhaTrainingArguments(TrainingArguments):
    """
    Subclassing training arugments because there are some arugments that
    probably should remaintain consistent during training

    Note there are many TrainingArguments, please refer back to hugging face documentation for all settings
    """

    def __init__(self, 
                 working_dir: str,
                 run_name: str=DEFAULT_RUN_NAME,
                 project_name: str=DEFAULT_PROJECT_NAME):
        super().__init__(working_dir)
        
        self.run_name = run_name
        self.project_name = project_name
        self.label_names = DEFAULT_COLUMNS
        self.logging_strategy = IntervalStrategy.STEPS
        self.logging_steps = 10
        self.eval_strategy = IntervalStrategy.STEPS
        self.eval_steps = 100

        self.per_device_train_batch_size = 64
        self.per_device_eval_batch_size = 32
        self.dataloader_num_workers = 4

        # In inferance, model saves all predictions on GPU by default
        # So in soundscape evals, this can be expensive
        # This setting has it give predictions back to CPU every X steps
        # should make things cheaper
        # https://discuss.huggingface.co/t/cuda-out-of-memory-during-evaluation-but-training-is-fine/1783/12
        self.eval_accumulation_steps = 100


class PyhaTrainer(Trainer):
    def __init__(
        self,
        model: BaseModel,
        dataset: AudioDataset,
        metrics: ComputeMetricsBase = None,
        training_args: PyhaTrainingArguments = None,
        data_collator=None,
        preprocessor=None,
    ):
        assert issubclass(type(model), BaseModel), (
            "PyhaTrainer only works with BaseModel. Please have model inherit from BaseModel"
        )
        
        ## HANDLES DEFAULT ARGUMENTS FOR HUGGING FACE TRAINER
        self.training_args = (training_args if training_args 
                              else PyhaTrainingArguments("working_dir"))

        self.wandb_logger = WANDBLogging(self.training_args.project_name)
        self.dataset = dataset

        ## DEFINES METRICS FOR DETERMINING HOW GOOD MODEL IS
        # if metrics is not None:
        #     self.compute_metrics = metrics
        # else:

        # Will create default metrics such as cMAP and AUROC
        num_classes = self.dataset.get_number_species()

        compute_metrics = AudioClassificationMetrics([], num_classes=num_classes)

        super().__init__(
            model,
            training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            data_collator=data_collator,
            processing_class=preprocessor,
            compute_metrics=compute_metrics,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="valid"):
        # print(eval_dataset)
        if eval_dataset is None:
            eval_dataset = self.dataset["valid"]
            metric_key_prefix = "valid"

        if ignore_keys is None:
            # is this the best place for this?
            # there maybe a training_arg that defines this by default. Should be changed there...
            ignore_keys = ["audio", "audio-in"]

        super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
