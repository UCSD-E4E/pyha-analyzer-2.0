from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
from pyha_analyzer.models.demo_CNN import ResnetConfig, ResnetModel
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors
from pyha_analyzer.models import EfficentNet


coralreef_extractor = extractors.CoralReef()
coral_ads = coralreef_extractor("/home/s.kamboj.400/unzipped-coral")
print(coral_ads)

preprocessor = MelSpectrogramPreprocessors(duration=5, class_list=coral_ads["train"].features["labels"].names)

coral_ads["train"].set_transform(preprocessor)
coral_ads["valid"].set_transform(preprocessor)
coral_ads["test"].set_transform(preprocessor)

resnet50d_config = ResnetConfig(
    #num_classes=len(coral_ads["train"].features["labels"].names), input_channels=1
    num_classes=2, input_channels=1
)

#model = EfficentNet(num_classes=len(coral_ads["train"].features["labels"].names))
model = EfficentNet(num_classes=2)

args = PyhaTrainingArguments(
    working_dir="working_dir"
)
args.num_train_epochs = 1
args.eval_steps = 5

# TODO : This causes an error, likely because the model is meant for multiclass but this is binary? So the labels are not formatted correctly? Needs to get debugged
trainer = PyhaTrainer(
    model=model,
    dataset=coral_ads,
    training_args=args
)
trainer.train()
trainer.evaluate(eval_dataset=coral_ads["test"], metric_key_prefix="Soundscape")

