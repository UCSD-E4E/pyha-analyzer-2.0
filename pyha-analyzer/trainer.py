from transformers import Trainer
import numpy as np

"""
Intended to wrap around the normal hugging face trainer

In case there are any project spefific configs we need to add here

For example: We may want to log git hashes, so that can be included
"""
class PyhaTrainer(Trainer):
    def __init__(self,
                 model, 
                 training_args, 
                 dataset, 
                 test_dataset=None,
                 data_collator=None, 
                 preprocessor=None):
        
        self.dataset = dataset

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