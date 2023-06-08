import argparse
import torch
from src.data_management import show_tensor_image
from src.models import get_model
from data_management.noise import NoiseAdder
from configs.general import MAX_TIMESTAMPS
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument("-c", "--config", type=str, default="", help="Path to model config")
    parser.add_argument("-l", "--label", type=int, default=0, help="Choose label of created image")
    parser.add_argument("-e", "--device", type=str, default="cuda", help="Device to train onto")
    parser.add_argument("-t", "--tag", default="", type=str, help="unique name of this training process to name checkpoint after")
    return parser.parse_args()


def main(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    model = get_model(args.model, config_path=args.config, device=device).to(device)

    noise_adder = NoiseAdder(MAX_TIMESTAMPS, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params}")
    print(f"device: {device}")

    x = torch.randn(1, 1, 128, 128).to(device)
    y = torch.tensor(args.label).flatten().to(device)

    for timestamp in tqdm(range(noise_adder.all_timestamps-1, -1, -1), desc="Timestamp", position=0):
        with torch.no_grad():
            x = noise_adder.sample_timestep(model, t=torch.tensor([timestamp], device=device), x=x, y=y)
    show_tensor_image(x)


if __name__ == "__main__":
    args = parse_args()
    main(args)
