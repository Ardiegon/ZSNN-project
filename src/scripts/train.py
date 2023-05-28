import os
import torch 
import argparse

from torch.utils.data import DataLoader

from models import get_model
from data_management.cocodataset import get_dataset, show_tensor_image
from utils import get_current_time

from configs.path import CHECKPOINTS_PATH

BATCH_SIZE = 64

def initialize_opts(args):

    train_id = args.model+get_current_time()
    curr_checkpoints_dir = os.path.join(CHECKPOINTS_PATH, train_id)
    os.makedirs(curr_checkpoints_dir, exist_ok=True)

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = get_model(args.model, config_path = args.config)
    
    return {
        "model": model,
        "dataloader": dataloader,
        "checkpoints_dir": curr_checkpoints_dir
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument("-c", "--config", type=str, default="", help="Path to model config")
    return parser.parse_args()

#draft
def main(args):
    opts = initialize_opts(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
