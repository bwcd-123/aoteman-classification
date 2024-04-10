import torch
from torch import nn


class AddNoise(nn.Module):
    """
    Add noise to images.

    Input is Tensor(float), std=0.01.

    Parameters:

    noise_scale: float, default=0.01
        Standard deviation of the noise.
    """
    def __init__(self, noise_scale=0.01, *args, **kwargs) -> None:
        super(AddNoise, self).__init__(*args, **kwargs)
        self.noise_scale = noise_scale


    def forward(self, x):
        noise = torch.normal(0., self.noise_scale, x.shape)
        print(111, x.shape, noise.shape)
        return x + noise
