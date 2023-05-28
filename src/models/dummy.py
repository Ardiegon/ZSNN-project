import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(3, 3, self.kernel_size)

    def forward(self, x):
        return (self.conv(x))