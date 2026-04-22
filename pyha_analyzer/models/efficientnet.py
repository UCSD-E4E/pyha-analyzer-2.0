## Based on https://github.com/DBD-research-group/BirdSet/blob/main/birdset/modules/models/efficientnet.py#L10
from transformers import AutoConfig, EfficientNetForImageClassification
from torch import nn
from .base_model import Model, ModelInput, ModelOutput, has_required_inputs
import torch


class EfficientNetInputs(ModelInput):
    @classmethod
    def from_dict(cls, some_input: dict):
        labels = some_input["labels"]
        spectrogram = some_input.get("spectrogram") or some_input.get("audio_in")
        waveform = some_input.get("waveform") or some_input.get("audio")
        assert spectrogram is not None or waveform is not None
        return cls(labels, spectrogram=spectrogram, waveform=waveform)


class EfficentNet(Model, nn.Module):
    def __init__(
        self,
        num_channels: int = 1,
        num_classes: int = None,
    ):
        super().__init__()
        self.input_format = EfficientNetInputs
        self.num_channels = num_channels
        self.num_classes = num_classes

        config = AutoConfig.from_pretrained(
            "google/efficientnet-b1",
            num_labels=self.num_classes,
            num_channels=self.num_channels,
            problem_type="multi_label_classification"
        )
        self.model = EfficientNetForImageClassification(config)

    @has_required_inputs()
    def forward(self, x: EfficientNetInputs) -> ModelOutput:
        output = self.model(pixel_values=x.spectrogram, labels=x.labels)
        return ModelOutput(logits=output.logits, loss=output.loss, labels=x.labels)

    def get_embedding(self, audio_batch):
        self.model.eval()
        with torch.no_grad():
            device = next(self.model.parameters()).device
            audio_batch = audio_batch.to(device)
            output = self.model.efficientnet(pixel_values=audio_batch, return_dict=True)
            last_hidden = output.last_hidden_state  # [B, 1280, H, W]
            pooled = torch.nn.functional.adaptive_avg_pool2d(last_hidden, output_size=(1, 1))
            return pooled.squeeze(-1).squeeze(-1)  # [B, 1280]
