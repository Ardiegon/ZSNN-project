import os 

MAIN_DATASET_DIR_PATH = "Track2\\data\\Console_sliced"
MAIN_TRAIN_DATASET_DIR = os.path.join(MAIN_DATASET_DIR_PATH, "train")
MAIN_ANNOTATIONS_PATH = os.path.join(MAIN_TRAIN_DATASET_DIR, "_annotations.coco.json")

ADDITIONAL_DATASET_DIR_PATH = "Additional\\data"
ADDITIONAL_TRAIN_DATASET_DIR = os.path.join(ADDITIONAL_DATASET_DIR_PATH, "train")

MODEL_CONFIGS_PATH = os.path.join("src", "configs", "models")
CHECKPOINTS_PATH = os.path.join("src", "checkpoints")