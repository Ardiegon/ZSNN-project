import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from torchvision import transforms

import data_management.cocodataset as main
import data_management.additional_dataset as additional

class DatasetTypes(Enum):
    MAIN = "main"
    ADDITIONAL = "additional"

def get_dataset(dataset_type):
    map_type_to_dataset = {
        DatasetTypes.MAIN: main.get_dataset,
        DatasetTypes.ADDITIONAL: additional.get_dataset
    }
    return map_type_to_dataset[dataset_type]()

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().detach().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    plt.show()
    

