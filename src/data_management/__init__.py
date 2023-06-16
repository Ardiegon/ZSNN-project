import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from torchvision import transforms

import data_management.cocodataset as main
import data_management.additional_dataset as additional
import data_management.cifar10_dataset as cifar10
import data_management.butterflies_dataset as butterflies

class DatasetTypes(Enum):
    MAIN = "main"
    ADDITIONAL = "additional"
    CIFAR10 = "cifar10"
    BUTTERFLIES = "butterflies"

def get_dataset(dataset_type):
    map_type_to_dataset = {
        DatasetTypes.MAIN: main.get_dataset,
        DatasetTypes.ADDITIONAL: additional.get_dataset,
        DatasetTypes.CIFAR10: cifar10.get_dataset,
        DatasetTypes.BUTTERFLIES: butterflies.get_dataset

    }
    return map_type_to_dataset[dataset_type]()

def show_tensor_image(image, write_here_instead = "", ax = None):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().detach().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    if write_here_instead:
        plt.imsave(write_here_instead, reverse_transforms(image))
    else:    
        if ax is not None:
            ax.imshow(reverse_transforms(image), cmap='gray')
            ax.axis("off")
        else:
            plt.imshow(reverse_transforms(image), cmap='gray')
            plt.show()
    return reverse_transforms(image)
    

