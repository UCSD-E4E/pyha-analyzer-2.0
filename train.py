from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors, preprocessors
from pyha_analyzer.models.demo_CNN import ResnetConfig, ResnetModel
from audiomentations import Compose, AddColorNoise

# Dataset
birdset_extactor = extractors.Birdset()
hsn_ads = birdset_extactor("HSN")

# Preprocessor (TODO HAVE CHECK FOR TRAINING TO DISABLE SOME AUGMENTATIONS)
augment = Compose([
    AddColorNoise(min_snr_db=1, max_snr_db=5, min_f_decay=-3.01, max_f_decay=-3.0, p=1),
])
preprocessor = preprocessors.MelSpectrogramPreprocessors(duration=5)
hsn_ads.set_transform(preprocessor)

# Model
resnet50d_config = ResnetConfig(
    num_classes=len(hsn_ads["train"].features["ebird_code"].names), input_channels=1
)
model = ResnetModel(resnet50d_config)

# Train
args = PyhaTrainingArguments(
    working_dir="working_dir"
)
args.num_train_epochs = 10

trainer = PyhaTrainer(
    model=model,
    dataset=hsn_ads,
    training_args=args
)
trainer.train()