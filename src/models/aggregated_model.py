import torch
import torch.nn as nn
from huggingface import UNet2DModelAdapted
from conditional_model import ConditionModel


class AggregatedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        out_channel = config.get["out_channels"]

        self.image_generation_model = UNet2DModelAdapted(**config)
        self.condition_model = ConditionModel(
            config,
            dict_size=config.get("dict_size", 10),
            n_classes=config.get("n_classes", 5)
        )
        checkpoint_path = config.get("checkpoint_path")
        if config.get("state_dict_path"):
            state_dict = torch.load(checkpoint_path)["state_dict"]
            self.image_generation_model.load_state_dict(state_dict)
        self.final_convolution = nn.Conv2d(2*out_channel, out_channel, **config.get("final_layer", {}))

    def forward(self, input, timestamp, class_labels):
        img_out = self.image_generation_model(input, timestamp)
        cond_out = self.condition_model(input, timestamp, class_labels)
        concat_out = torch.concat([img_out, cond_out], dim=1)
        return self.final_convolution(concat_out)
