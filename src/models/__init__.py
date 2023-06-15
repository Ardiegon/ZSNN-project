import os
import torch.nn as nn

from models.dummy import *
from models.generators import *
from models.aggregated_model import *


from utils import read_config
from configs.path import MODEL_CONFIGS_PATH

def get_default_config_path(model_name):
    path = os.path.join(MODEL_CONFIGS_PATH, "default_"+model_name+".json")
    if not os.path.isfile(path):
        message = f"Default config for {model_name} was not found. Please specify path to existing one."
        raise RuntimeError(message)
    return path
    
def _robust_issubclass(val, class_):
    try:
        return issubclass(val, class_)
    except TypeError:
        pass

def _models():
    return {name:val for name, val in globals().items() if _robust_issubclass(val, nn.Module)}


def get_model(model_name, config_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    checkpoint_path = None
    if not config_path:
        config_path = get_default_config_path(model_name)
    config = read_config(config_path)
    if "checkpoint_path" in config.keys():
        checkpoint_path = config.pop("checkpoint_path")
    model = _models()[model_name](**config)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=torch.device(device=device))
        model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    print(_models())