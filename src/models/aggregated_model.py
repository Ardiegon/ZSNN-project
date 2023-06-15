import torch
import torch.nn as nn
from src.models.huggingface import UNet2DModelAdapted
from src.models.conditional_model import ConditionModel


class AggregatedModel(nn.Module):
    def __init__(self, **config):
        super().__init__()
        generation_config = config["image_generator"]
        conditional_config = config["conditional_model"]
        final_layer_config = config["final_layer"]
        checkpoint_path = generation_config.pop("checkpoint_path")

        self.image_generation_model = UNet2DModelAdapted(**generation_config)
        self.condition_model = ConditionModel(
            **conditional_config
        )
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path)["state_dict"]
            self.image_generation_model.load_state_dict(state_dict)
        self.final_convolution = nn.Conv2d(
            generation_config["out_channels"]+conditional_config["out_channels"],
            generation_config["out_channels"],
            **final_layer_config)

    def forward(self, input, timestamp, class_labels):
        img_out = self.image_generation_model(input, timestamp)
        cond_out = self.condition_model(input, timestamp, class_labels)
        concat_out = torch.concat([img_out, cond_out], dim=1)
        return self.final_convolution(concat_out)
