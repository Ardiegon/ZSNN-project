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
        self.n_classes = self.condition_model.n_classes
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path)
            self.image_generation_model.load_state_dict(state_dict)
        self.final_convolution = nn.Conv2d(
            generation_config["out_channels"]+conditional_config["out_channels"],
            conditional_config["out_channels"],
            **final_layer_config)

    def forward(self, input, timestamp, class_labels):
        img_gen_input = input[:, 0:1, :, :]
        img_out = self.image_generation_model(img_gen_input, timestamp)
        cond_out = self.condition_model(input, timestamp, class_labels)
        concat_out = torch.concat([img_out, cond_out], dim=1)
        return self.final_convolution(concat_out)

    def get_l2_reg_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return l2_loss
