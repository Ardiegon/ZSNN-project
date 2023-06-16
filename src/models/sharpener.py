# WARNING! Broken due to bad checkpoint saved in cloud, possibly leave to retrain in future

from PIL import Image

def sharpen_image( image):
    return image.resize((800, 800), resample=Image.Resampling.NEAREST).convert("L")

