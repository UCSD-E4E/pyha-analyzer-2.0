## Based on https://github.com/DBD-research-group/BirdSet/blob/main/birdset/modules/models/efficientnet.py#L10

from transformers import AutoConfig, EfficientNetForImageClassification
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from torch import nn
from typing import List
from .base_model import BaseModel, has_required_inputs

class EfficentNet(nn.Module, BaseModel): #BaseModel
    def __init__(
        self,
        num_channels: int = 1,
        num_classes: int = None,
    ):
        super().__init__()
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
    def forward(self, labels, **kwrgs):
        input = kwrgs["audio_in"]
        output = self.model(pixel_values=input, labels=labels)
        
        return {
            "loss": output.loss,
            "logits": output.logits
        }