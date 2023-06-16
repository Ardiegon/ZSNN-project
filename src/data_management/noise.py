import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from configs.general import IMG_SIZE
from data_management import show_tensor_image
from models.sharpener import sharpen_image

class NoiseAdder():
    def __init__(self, all_timestamps, device) -> None:
        self.device = device
        self.all_timestamps = all_timestamps
        
        self.betas, \
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
        return betas.to(self.device), sqrt_recip_alphas.to(self.device), sqrt_alphas_cumprod.to(self.device), \
            sqrt_one_minus_alphas_cumprod.to(self.device), posterior_variance.to(self.device)

    def get_index_from_list(self, vals, timestamp, image_shape):
        batch_size = timestamp.shape[0]
        out = vals.gather(-1, timestamp)
        return out.reshape(batch_size, *((1,) * (len(image_shape) - 1))).to(self.device)

    def __call__(self, image, timestamp, device):
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


    @torch.no_grad()
    def sample_timestep(self, model, x, t, y=None):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        if y is not None:
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t, y) / sqrt_one_minus_alphas_cumprod_t
            )
        else:
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
            )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_plot_image(self, model, path):
        img_size = IMG_SIZE
        num_images = 10
        base_img = torch.randn((1, 1, img_size, img_size), device=self.device)
        stepsize = int(self.all_timestamps/num_images)
        n_classes = 1
        if hasattr(model, "n_classes"):
            n_classes = model.n_classes
        fig, axs = plt.subplots(n_classes, num_images)
        n_classes = torch.arange(0, n_classes, device=self.device)
        backward_iter = num_images - 1
        for cond_class in n_classes:
            img = base_img
            for i in range(0,self.all_timestamps)[::-1]:
                t = torch.full((1,), i, device=self.device, dtype=torch.long)
                label = torch.unsqueeze(cond_class, dim=0)
                img = self.sample_timestep(model, img, t, label)
                img = torch.clamp(img, -1.0, 1.0)
                if i % stepsize == 0:
                    show_tensor_image(img.detach().cpu(), ax=axs[int(cond_class), backward_iter])
                    axs[int(cond_class), backward_iter].text(0.5,0.5, str(int(i/stepsize)+1))
                    backward_iter -= 1
            backward_iter = num_images - 1
        fig.savefig(path)
        plt.clf()
        plt.cla()
        plt.close()

    @torch.no_grad()
    def sample_plot_image_old(self, model, path):
        img_size = IMG_SIZE
        img = torch.randn((1, 1, img_size, img_size), device=self.device)
        num_images = 10
        stepsize = int(self.all_timestamps/num_images)
        fig, axs = plt.subplots(1, num_images)
        backward_iter = 9
        for i in range(0,self.all_timestamps)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.sample_timestep(model, img, t)
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                output  = show_tensor_image(img.detach().cpu(), ax=axs[backward_iter])
                if i == 0:
                    plt.imsave(path.replace(".png", "_last.png"), sharpen_image(output))
                axs[backward_iter].text(0.5,0.5, str(int(i/stepsize)+1))
                backward_iter -= 1
        fig.savefig(path)
        plt.clf()
        plt.cla()
        plt.close()
