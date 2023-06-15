import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ConditionModel(nn.Module):
    def __init__(self, dict_size=10, n_classes=5, **config):
        super().__init__()
        self.n_classes = n_classes
        self.embeddings = nn.Embedding(dict_size, n_classes)
        self.model = UNet2DModel(**config)

    def forward(self, input, timestamp, class_labels):
        bs, ch, w, h = input.shape
        class_cond = self.embeddings(class_labels)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)

        net_input = torch.cat((input, class_cond), 1)
        return self.model(net_input, timestamp).sample
