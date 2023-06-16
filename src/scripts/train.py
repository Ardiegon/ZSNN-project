import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch 
import argparse
import logging
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam

from models import get_model
from data_management import DatasetTypes, get_dataset, show_tensor_image
from data_management.noise import NoiseAdder
from utils import get_current_time

from configs.path import CHECKPOINTS_PATH
from configs.general import MAX_TIMESTAMPS, BATCH_SIZE


def wasserstein_distance(real, pred):
    return torch.tensor([torch.abs(real[i] - pred[i]).sum() for i in range(len(real))]).mean()


def get_loss(model, noise_adder, batch, timestep, device, regularization = False):
    image, label, mask = batch
    input_img = torch.concat([image, mask], dim=1)
    image_noisy, noise = noise_adder(input_img, timestep, device)
    noise_pred = model(image_noisy, timestep, label)
    if regularization:
        reg_loss = model.get_l2_reg_loss()
        loss = F.mse_loss(noise, noise_pred) + 0.1 * reg_loss
    else:
        loss = F.mse_loss(noise, noise_pred)
    return loss


def initialize_opts(args):
    tag = args.tag if args.tag=="" else "_"+args.tag+"_"
    train_id = args.model+tag+get_current_time()
    curr_checkpoints_dir = os.path.join(CHECKPOINTS_PATH, train_id)
    os.makedirs(curr_checkpoints_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"

    logging.basicConfig(filename=os.path.join(curr_checkpoints_dir, "alog.txt"),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

    dataset = get_dataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = get_model(args.model, config_path = args.config).to(device)

    noise_adder = NoiseAdder(MAX_TIMESTAMPS, device)
    
    return {
        "model": model,
        "noise_adder": noise_adder,
        "dataloader": dataloader,
        "checkpoints_dir": curr_checkpoints_dir,
        "device": device
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument("-c", "--config", type=str, default="", help="Path to model config")
    parser.add_argument("-d", "--dataset", type=DatasetTypes, choices=list(DatasetTypes), default=DatasetTypes.MAIN, help= "Chose training dataset")
    parser.add_argument("-e", "--device", type=str, default="cuda", help="Device to train onto")
    parser.add_argument("-t", "--tag", default = "", type=str, help ="unique name of this training process to name checkpoint after")
    parser.add_argument("-po", "--plot-old", action="store_true", help ="use olderplotting while training")
    parser.add_argument("-rl", "--reg_loss", action="store_true", help ="Use L2 Regularization")
    parser.add_argument("--epochs", type=int, default = 100, help="Number of epochs")
    return parser.parse_args()

def main(args):
    opts = initialize_opts(args)
    model = opts["model"]
    noise_adder = opts["noise_adder"]
    dataloader = opts["dataloader"]
    device = opts["device"]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params}")
    print(f"device: {device}")
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = args.epochs

    for epoch in tqdm(range(epochs), desc="Epochs", position=0):
        for batch in tqdm(dataloader, desc="Batches", position=1):
            optimizer.zero_grad()

            timestamp = torch.randint(0, MAX_TIMESTAMPS, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, noise_adder, batch, timestamp, device, args.reg_loss)
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch} Loss: {loss.item()} ")
        if args.plot_old:
            noise_adder.sample_plot_image_old(model, os.path.join(opts["checkpoints_dir"], f"results_{epoch}.png"))
        else:
            noise_adder.sample_plot_image(model, os.path.join(opts["checkpoints_dir"], f"results_{epoch}.png"))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(opts["checkpoints_dir"], f"model_{epoch}.pth"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
