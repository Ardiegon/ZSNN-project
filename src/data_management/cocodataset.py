from typing import Any, Tuple
import numpy as np
import torch
import cv2 as cv

from torchvision.datasets import CocoDetection
from torchvision import transforms
from configs.path import MAIN_TRAIN_DATASET_DIR, MAIN_ANNOTATIONS_PATH
from configs.general import IMG_SIZE, SCALE_FACTOR
    
def get_dataset():
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    return CocoDetectionImageOnly(root=MAIN_TRAIN_DATASET_DIR, annFile=MAIN_ANNOTATIONS_PATH, transform=image_transforms)

class CocoDetectionImageOnly(CocoDetection):
    def __getitem__(self, index: int) -> Any:
        image, metadata = super().__getitem__(index)
        mask = np.zeros(image.shape[1:])
        if metadata:
            points = []
            segmentation_points = metadata[0]["segmentation"]
            for segment in segmentation_points:
                segment = np.float32(segment).reshape(-1, 1, 2)/SCALE_FACTOR
                points.append(segment.astype("int32"))
            cv.fillPoly(mask, pts=points, color=255)
            label = metadata[0]["category_id"] + 1
        else:
            label = 0
        label = torch.tensor(label).to("cuda")
        mask = np.expand_dims(mask, axis=0)//255
        return image, label, torch.tensor(mask).to(torch.float32)

