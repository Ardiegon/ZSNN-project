from diffusers import UNet2DModel

class UNet2DModelAdapted(UNet2DModel):
    def forward(self, sample, timestep, class_labels = None, return_dict = True):
        output_class = super().forward(sample, timestep, class_labels, return_dict)
        return output_class.sample