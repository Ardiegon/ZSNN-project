import os 

DATASET_DIR_PATH = os.path.join("Track2","data", "Console_sliced", "train")
ANNOTATIONS_PATH = os.path.join(DATASET_DIR_PATH, "_annotations.coco.json") 

MODEL_CONFIGS_PATH = os.path.join("src", "configs", "models")
CHECKPOINTS_PATH = os.path.join("src", "checkpoints")