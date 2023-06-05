from typing import Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import CocoDetection
from torchvision import transforms
from configs.path import MAIN_TRAIN_DATASET_DIR, MAIN_ANNOTATIONS_PATH
    
def get_dataset():
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) 
    ])
    return CocoDetectionImageOnly(root=MAIN_TRAIN_DATASET_DIR, annFile=MAIN_ANNOTATIONS_PATH, transform=image_transforms)

class CocoDetectionImageOnly(CocoDetection):
    def __getitem__(self, index: int) -> Any:
        image, _ = super().__getitem__(index)
        return image
