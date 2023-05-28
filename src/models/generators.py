import torch.nn as nn
from src.models.blocks import *

class SimpleAttentionUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = Encoder(3, 64)
        self.e2 = Encoder(64, 128)
        self.e3 = Encoder(128, 256)

        self.b1 = Conv(256, 512)

        self.d1 = Decoder([512, 256], 256)
        self.d2 = Decoder([256, 128], 128)
        self.d3 = Decoder([128, 64], 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)
        return output
    
if __name__ == "__main__":
    model = SimpleAttentionUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))