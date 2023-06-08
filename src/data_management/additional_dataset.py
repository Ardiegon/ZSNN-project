import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from configs.path import ADDITIONAL_TRAIN_DATASET_DIR
from configs.general import IMG_SIZE

def get_dataset():
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) 
    ])
    return SimpleImageDataset(root=ADDITIONAL_TRAIN_DATASET_DIR, transform=image_transforms)


class SimpleImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.image_list = os.listdir(root)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image