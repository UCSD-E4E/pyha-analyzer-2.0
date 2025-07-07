import torch
from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
from pyha_analyzer.models import EfficentNet
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors

def load_model(weights_path):
    model = EfficentNet(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(preprocessor, audio_path):
    return audio_path.set_transform(preprocessor)


model=load_model("./coral_model_4146a1f.pt") #replace with your model path
preprocessor = MelSpectrogramPreprocessors(duration=5, class_list=['Degraded_Reef', 'Non_Degraded_Reef'])
#audio_path= preprocess_image(preprocessor, "../unzipped-coral/Degraded_Reef/coral_reef_audio.wav") #replace with your audio path

#print(model)
print(model.model.efficientnet.extract_features(input_tensor))


