import torch
import torch.nn.functional as F

class NoiseAdder():
    def __init__(self, all_timestamps) -> None:
        self.all_timestamps = all_timestamps
        self.sqrt_recip_alphas, \
        self.sqrt_alphas_cumprod, \
        self.sqrt_one_minus_alphas_cumprod, \
        self.posterior_variance \
            = self.calculate_alpha_factors()

    def calculate_alpha_factors(self):
        betas = torch.linspace(0.0001, 0.02, self.all_timestamps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        return sqrt_recip_alphas, sqrt_alphas_cumprod, \
            sqrt_one_minus_alphas_cumprod, posterior_variance

    def get_index_from_list(self, vals, timestamp, image_shape):
        batch_size = timestamp.shape[0]
        out = vals.gather(-1, timestamp.cpu())
        return out.reshape(batch_size, *((1,) * (len(image_shape) - 1))).to(timestamp.device)

    def __call__(self, image, timestamp, device="cpu"):
        """ 
        Simulates image with noise for timestamp
        """
        noise = torch.randn_like(image)
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, timestamp, image.shape
            )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, timestamp, image.shape
        )
        return sqrt_alphas_cumprod_t.to(device) * image.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
