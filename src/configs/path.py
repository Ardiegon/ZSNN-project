import os 

DATASET_DIR_PATH = "path/to/your/dataset"
TRAIN_DATASET_DIR = os.path.join(DATASET_DIR_PATH, "train")
ANNOTATIONS_PATH = os.path.join(TRAIN_DATASET_DIR, "_annotations.coco.json")

MODEL_CONFIGS_PATH = os.path.join("src", "configs", "models")
CHECKPOINTS_PATH = os.path.join("src", "checkpoints")