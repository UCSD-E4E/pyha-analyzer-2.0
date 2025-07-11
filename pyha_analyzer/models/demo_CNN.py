# https://huggingface.co/docs/transformers/en/custom_models
from transformers import PretrainedConfig, PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from torch import nn
from typing import List
from .base_model import BaseModel, has_required_inputs


class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(
                f"`block_type` must be 'basic' or bottleneck', got {block_type}."
            )
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(
                f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}."
            )

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)


BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}


class ResnetModel(PreTrainedModel, BaseModel):
    config_class = ResnetConfig

    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        BaseModel.__init__(self)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

        self.loss_func = nn.BCEWithLogitsLoss()

    @has_required_inputs()
    # TODO Bug, when we are preprocessing live, we need to have audio defined here
    # A solution could be we change this to kwargs and use has_required_inputs..
    def forward(self, audio, audio_in, labels=None):
        logits = self.model.forward(audio_in)
        return_dict = {"logits": logits}
        if labels is not None:
            return_dict["loss"] = self.loss_func(logits, labels)
        return return_dict
