## Based on https://github.com/DBD-research-group/BirdSet/blob/main/birdset/modules/models/efficientnet.py#L10
from transformers import AutoConfig, EfficientNetForImageClassification
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from torch import nn
from typing import List
from .base_model import BaseModel, has_required_inputs
import torch

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
    
    def get_embedding(self, **kwrgs):
        # The model we are using is EfficientNetForImageClassification, which wraps the base EfficientNet and then adds global average pooling, dropout, and a final linear classification layer. 
        # To get the vector embedding, we ran just the EfficientNet portion then used pytorch to do global average pooling. Then, we squeezed the dimensions and got a vector embedding of size 1280
        self.model.eval()
        with torch.no_grad():
            input = torch.tensor(kwrgs["audio_in"], dtype=torch.float32)
            if input.ndim == 3:
                input = input.unsqueeze(0)

            device = next(self.model.parameters()).device
            input = input.to(device)

            #Grab output of EfficientNet model
            output = self.model.efficientnet(pixel_values=input, return_dict=True)
            last_hidden = output.last_hidden_state
            # Apply global average pooling
            embedding = torch.nn.functional.adaptive_avg_pool2d(last_hidden, output_size=(1, 1))
            embedding = embedding.squeeze(-1).squeeze(-1).squeeze(0)  # squeeze to get expected shape: [1280]
            return embedding 
         