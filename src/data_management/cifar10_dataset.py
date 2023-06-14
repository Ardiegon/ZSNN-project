import torch.utils.data
import torchvision.datasets
from torchvision import transforms

from configs.general import IMG_SIZE

def get_dataset():
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) 
    ])
    data = torchvision.datasets.CIFAR10(root="cifar10", download=True, 
                                         transform=image_transforms)
    return MyCifar([data])

class MyCifar(torch.utils.data.ConcatDataset):
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return img
