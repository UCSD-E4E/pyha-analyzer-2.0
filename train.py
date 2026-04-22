"""Example code for training a model on birdset.

Make a copy of this file and edit it to train your own model on birdset or your own dataset!
Make the copy named experiment_*.py 
"""

from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors
from pyha_analyzer.models import EfficentNet

region = "PER"
experiment_parameters = [
    {
        "augmentation": None,
        "run_name": "baseline-EfficentNet-PERU",
        "region": region
    },
]

for parameters in experiment_parameters:

    birdset_extactor = extractors.Birdset()
    hsn_ads = birdset_extactor(parameters["region"])

    preprocessor = MelSpectrogramPreprocessors(
        duration=5, 
        augment=parameters["augmentation"],
        class_list=hsn_ads["train"].features["labels"].feature.names
    )

    test_preprocessor = MelSpectrogramPreprocessors(
        duration=5, 
        augment=None,
        class_list=hsn_ads["train"].features["labels"].feature.names
    )

    hsn_ads["train"].set_transform(preprocessor)
    hsn_ads["valid"].set_transform(test_preprocessor)
    hsn_ads["test"].set_transform(test_preprocessor)

    model = EfficentNet(num_classes=len(hsn_ads["train"].features["ebird_code"].names))

    args = PyhaTrainingArguments(
        working_dir="model_checkpoints/" + parameters["run_name"],
    )
    args.num_train_epochs = 20
    args.eval_steps = 500
    args.learning_rate = 0.001
    args.run_name = parameters["run_name"] + " " + parameters["region"]

    trainer = PyhaTrainer(
        model=model,
        dataset=hsn_ads,
        training_args=args
    )
    trainer.train()
    trainer.evaluate(eval_dataset=hsn_ads["test"], metric_key_prefix="Soundscape")

    del model
    del trainer

