import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import CocoDetection
from torchvision import transforms
from configs.path import DATASET_DIR_PATH, ANNOTATIONS_PATH
    
def get_dataset():
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) 
    ])
    return CocoDetection(root=DATASET_DIR_PATH, annFile=ANNOTATIONS_PATH, transform=image_transforms)

def show_tensor_image(image, ):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    plt.show()
    
