from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
from pyha_analyzer.models.demo_CNN import ResnetConfig, ResnetModel
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors
from pyha_analyzer.models import EfficentNet


birdset_extactor = extractors.Birdset()
hsn_ads = birdset_extactor("HSN")
hsn_ads

preprocessor = MelSpectrogramPreprocessors(duration=5, class_list=hsn_ads["train"].features["labels"].feature.names)

hsn_ads["train"].set_transform(preprocessor)
hsn_ads["valid"].set_transform(preprocessor)
hsn_ads["test"].set_transform(preprocessor)

resnet50d_config = ResnetConfig(
    num_classes=len(hsn_ads["train"].features["ebird_code"].names), input_channels=1
)

model = EfficentNet(num_classes=len(hsn_ads["train"].features["ebird_code"].names))

args = PyhaTrainingArguments(
    working_dir="working_dir",
    run_name="logging_test",
    project_name="pyha_analyzer_2.0",
)
args.num_train_epochs = 30
args.eval_steps = 30

trainer = PyhaTrainer(
    model=model,
    dataset=hsn_ads,
    training_args=args
)
#trainer.train()
trainer.evaluate(eval_dataset=hsn_ads["test"], metric_key_prefix="Soundscape")

