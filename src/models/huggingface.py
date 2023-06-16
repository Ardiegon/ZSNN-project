import torch
from diffusers import UNet2DModel

class UNet2DModelAdapted(UNet2DModel):
    def forward(self, sample, timestep, class_labels=None, return_dict=True):
        output_class = super().forward(sample, timestep, None, return_dict)
        return output_class.sample
    
    def get_l2_reg_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return l2_loss