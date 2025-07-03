from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
from pyha_analyzer.models.demo_CNN import ResnetConfig, ResnetModel
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors
from pyha_analyzer.models import EfficentNet


coralreef_extractor = extractors.CoralReef()
coral_ads = coralreef_extractor("/home/s.kamboj.400/unzipped-coral")

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

print(coral_ads["test"])
print(trainer.evaluate(eval_dataset=coral_ads["test"], metric_key_prefix="Soundscape"))

result = trainer.predict(coral_ads["test"], metric_key_prefix="Soundscape")
loss_scalar, logits = result.predictions
print("Logits:\n", logits)

  

