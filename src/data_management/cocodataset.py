from typing import Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import CocoDetection
from torchvision import transforms
from configs.path import MAIN_TRAIN_DATASET_DIR, MAIN_ANNOTATIONS_PATH
from configs.general import IMG_SIZE
    
def get_dataset():
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) 
    ])
    return CocoDetectionImageOnly(root=MAIN_TRAIN_DATASET_DIR, annFile=MAIN_ANNOTATIONS_PATH, transform=image_transforms)

class CocoDetectionImageOnly(CocoDetection):
    def __getitem__(self, index: int) -> Any:
        image, _ = super().__getitem__(index)
        return image
