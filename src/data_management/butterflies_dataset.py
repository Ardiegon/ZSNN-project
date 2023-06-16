from datasets import load_dataset
from torchvision import transforms

from configs.general import IMG_SIZE

def get_dataset():
    dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset = load_dataset(dataset_name)['train']
    image_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) 
    ])
    dataset = [(image_transforms(image.convert("RGB")),0) for image in dataset["image"]]
    return dataset